"""Text2SQL 单元测试模块。

这个文件验证第三阶段结构化查询能力，包括 Schema Catalog 检索、SQL 生成、安全校验、
只读执行和 SQL Evidence 输出。
"""

from app.text2sql.executor import ReadOnlySQLiteExecutor
from app.text2sql.service import Text2SQLService, get_text2sql_service
from app.text2sql.generator import LLMSQLGenerator, RuleBasedSQLGenerator
from app.text2sql.schema import build_default_schema_catalog, load_schema_catalog_from_postgres
from app.text2sql.validator import SQLValidator
import httpx
import pytest


def test_text2sql_group_by_cuisine() -> None:
    # 按菜系统计数量应生成 GROUP BY SQL，并返回每个菜系的计数结果。
    result = get_text2sql_service().query("统计一下每个菜系的菜谱数量")

    assert result.sql is not None
    assert "GROUP BY cuisine" in result.sql
    assert "川菜" in result.answer
    assert "recipe_count" in result.answer
    assert result.raw_evidence[0]["source_type"].value == "sql"


def test_text2sql_rejects_write_sql() -> None:
    # SQL 校验器必须拒绝写操作，确保 LLM 只能生成候选语句，最终执行由程序校验决定。
    validator = SQLValidator(allowed_tables={"recipes"}, max_rows=10)
    result = validator.validate("DROP TABLE recipes")

    assert result.allowed is False
    assert "只允许 SELECT" in result.reason or "写操作" in result.reason


def test_default_schema_catalog_contains_recipes() -> None:
    # 未配置 PostgreSQL 时，Text2SQL 仍保留本地 fallback schema，保证开发环境可运行。
    catalog = build_default_schema_catalog()

    assert catalog.tables[0].name == "recipes"
    assert any(column.name == "cuisine" for column in catalog.tables[0].columns)


def test_schema_catalog_rejects_unsafe_table_name() -> None:
    # schema catalog 表名来自环境变量，必须拒绝带分号等注入风险的值。
    try:
        load_schema_catalog_from_postgres("postgresql://unused", table_name="schema_catalog;drop")
    except ValueError as exc:
        assert "非法 schema catalog 表名" in str(exc)
    else:
        raise AssertionError("unsafe schema catalog table name should be rejected")


def test_llm_sql_generator_returns_candidate_sql(monkeypatch) -> None:
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"sql":"SELECT COUNT(*) AS recipe_count FROM recipes","reason":"统计菜谱数量"}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("app.text2sql.generator.httpx.post", fake_post)
    catalog = build_default_schema_catalog()
    generator = LLMSQLGenerator(
        base_url="http://llm.local/v1",
        model="sql-test",
        api_key="test-key",
        fallback=RuleBasedSQLGenerator(),
        max_retries=0,
    )

    result = generator.generate("统计菜谱数量", catalog.retrieve("统计菜谱数量", top_k=2))

    assert captured["url"] == "http://llm.local/v1/chat/completions"
    assert captured["payload"]["temperature"] == 0.0
    assert result.sql == "SELECT COUNT(*) AS recipe_count FROM recipes"
    assert result.reason.startswith("llm_text2sql")


def test_llm_sql_generator_falls_back_on_failure(monkeypatch) -> None:
    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("llm unavailable")

    monkeypatch.setattr("app.text2sql.generator.httpx.post", failing_post)
    catalog = build_default_schema_catalog()
    generator = LLMSQLGenerator(
        base_url="http://llm.local/v1",
        model="sql-test",
        fallback=RuleBasedSQLGenerator(),
        max_retries=0,
    )

    result = generator.generate("按菜系统计菜谱数量", catalog.retrieve("按菜系统计菜谱数量", top_k=2))

    assert "SELECT" in result.sql
    assert "llm_text2sql_failed_fallback" in result.reason


def test_llm_sql_generator_strict_mode_does_not_fallback(monkeypatch) -> None:
    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("llm unavailable")

    monkeypatch.setattr("app.text2sql.generator.httpx.post", failing_post)
    catalog = build_default_schema_catalog()
    generator = LLMSQLGenerator(
        base_url="http://llm.local/v1",
        model="sql-test",
        fallback=RuleBasedSQLGenerator(),
        fallback_on_error=False,
        max_retries=0,
    )

    with pytest.raises(RuntimeError, match="text2sql llm request failed"):
        generator.generate("按菜系统计菜谱数量", catalog.retrieve("按菜系统计菜谱数量", top_k=2))


def test_text2sql_service_retries_llm_after_validation_failure(monkeypatch) -> None:
    prompts = []

    def fake_post(url, **kwargs):
        prompts.append(kwargs["json"]["messages"][1]["content"])
        sql = "DELETE FROM recipes" if len(prompts) == 1 else "SELECT COUNT(*) AS recipe_count FROM recipes"
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": f'{{"sql":"{sql}","reason":"validation retry test"}}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("app.text2sql.generator.httpx.post", fake_post)
    catalog = build_default_schema_catalog()
    service = Text2SQLService(
        schema_catalog=catalog,
        sql_generator=LLMSQLGenerator(
            base_url="http://llm.local/v1",
            model="sql-test",
            fallback=RuleBasedSQLGenerator(),
            max_retries=0,
        ),
        sql_validator=SQLValidator(allowed_tables={table.name for table in catalog.tables}, max_rows=20),
        executor=ReadOnlySQLiteExecutor(),
        schema_top_k=2,
        max_validation_retries=1,
    )

    result = service.query("数据库里有多少道菜")

    assert "recipe_count" in result.answer
    assert len(prompts) == 2
    assert "DELETE FROM recipes" in prompts[1]
    assert "校验失败原因" in prompts[1]
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["validation_attempts"] == 2
    assert metadata["validation_retry_count"] == 1
