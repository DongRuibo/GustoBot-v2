"""Text2SQL 服务编排模块。

这个文件串起第三阶段 Text2SQL 主链路：Schema Catalog 检索、候选 SQL 生成、
SQL AST/规则安全校验、只读执行、结果格式化和 SQL Evidence 输出。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from app.cache.semantic import stable_json_hash
from app.core.config import settings
from app.kb.embeddings import EmbeddingProvider, HashEmbeddingProvider, OpenAICompatibleEmbeddingProvider
from app.models import EvidenceSource
from app.text2sql.executor import (
    ReadOnlyPostgreSQLExecutor,
    ReadOnlySQLiteExecutor,
    SQLExecutionResult,
    SQLExecutor,
)
from app.text2sql.generator import (
    GeneratedSQL,
    LLMSQLGenerator,
    RuleBasedSQLGenerator,
    SQLGenerationFeedback,
    SQLGenerator,
)
from app.text2sql.schema import (
    SchemaCatalog,
    SchemaMatch,
    build_default_schema_catalog,
    load_schema_catalog_from_postgres,
)
from app.text2sql.validator import SQLValidationResult, SQLValidator


@dataclass(slots=True)
class Text2SQLQueryResult:
    # Text2SQLQueryResult 是节点层消费的内部结果，包含答案草稿、证据和执行过程关键信息。
    answer: str
    raw_evidence: list[dict[str, Any]] = field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Text2SQLPreparedQuery:
    # 预处理结果包含生成和校验后的 SQL，但还没有访问结构化数据库。
    generated_sql: GeneratedSQL
    validation: SQLValidationResult
    schema_matches: list[SchemaMatch]
    validation_attempts: int
    semantic_cache_allowed: bool = False
    semantic_cache_disabled_reason: str | None = None


class Text2SQLService:
    # Text2SQLService 是结构化查询链路入口，上层只调用 query，不直接拼 SQL 或访问数据库。
    def __init__(
        self,
        *,
        schema_catalog: SchemaCatalog,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        executor: SQLExecutor,
        schema_top_k: int,
        max_validation_retries: int = 0,
    ) -> None:
        self.schema_catalog = schema_catalog
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.executor = executor
        self.schema_top_k = schema_top_k
        self.max_validation_retries = max(0, max_validation_retries)

    def prepare_query(self, question: str) -> Text2SQLPreparedQuery:
        schema_matches = self.schema_catalog.retrieve(question, top_k=self.schema_top_k)
        feedback: SQLGenerationFeedback | None = None
        generated_sql: GeneratedSQL | None = None
        validation: SQLValidationResult | None = None
        validation_attempts = 0
        for attempt in range(self.max_validation_retries + 1):
            generated_sql = self.sql_generator.generate(question, schema_matches, feedback)
            validation = self.sql_validator.validate(generated_sql.sql)
            validation_attempts = attempt + 1
            if validation.allowed or attempt >= self.max_validation_retries:
                break
            feedback = SQLGenerationFeedback(
                invalid_sql=generated_sql.sql,
                validation_reason=validation.reason,
                attempt=attempt + 1,
            )
        if generated_sql is None or validation is None:
            raise RuntimeError("text2sql generation failed")
        disabled_reason = None if validation.allowed else "sql_validation_failed"
        return Text2SQLPreparedQuery(
            generated_sql=generated_sql,
            validation=validation,
            schema_matches=schema_matches,
            validation_attempts=validation_attempts,
            semantic_cache_allowed=disabled_reason is None,
            semantic_cache_disabled_reason=disabled_reason,
        )

    def execute_prepared(self, prepared: Text2SQLPreparedQuery) -> Text2SQLQueryResult:
        if not prepared.validation.allowed:
            return Text2SQLQueryResult(
                answer=f"生成的 SQL 没有通过安全校验：{prepared.validation.reason}",
                raw_evidence=[
                    self._to_raw_evidence(
                        prepared.generated_sql,
                        prepared.validation,
                        None,
                        prepared.schema_matches,
                        validation_attempts=prepared.validation_attempts,
                    )
                ],
                sql=prepared.validation.sql,
            )

        execution_result = self.executor.execute(prepared.validation.sql)
        return Text2SQLQueryResult(
            answer=self._format_answer(execution_result),
            raw_evidence=[
                self._to_raw_evidence(
                    prepared.generated_sql,
                    prepared.validation,
                    execution_result,
                    prepared.schema_matches,
                    validation_attempts=prepared.validation_attempts,
                )
            ],
            sql=prepared.validation.sql,
            rows=execution_result.rows,
        )

    def query(self, question: str) -> Text2SQLQueryResult:
        return self.execute_prepared(self.prepare_query(question))

    def schema_fingerprint(self) -> str:
        payload = {
            "tables": [
                {
                    "name": table.name,
                    "updated_at": table.updated_at,
                    "module": table.module,
                    "columns": [
                        {
                            "name": column.name,
                            "data_type": column.data_type,
                        }
                        for column in table.columns
                    ],
                }
                for table in self.schema_catalog.tables
            ]
        }
        return stable_json_hash(payload)

    def _format_answer(self, execution_result: SQLExecutionResult) -> str:
        # 当前不调用 LLM 解释 SQL 结果，而是把表格结果压缩成可读文本，避免解释层编造。
        if not execution_result.rows:
            return "SQL 查询已执行，但没有返回结果。"

        formatted_rows = []
        for row in execution_result.rows:
            cells = [f"{column}={row[column]}" for column in execution_result.columns]
            formatted_rows.append("，".join(cells))
        return "SQL 查询结果：" + "；".join(formatted_rows)

    def _to_raw_evidence(
        self,
        generated_sql: GeneratedSQL,
        validation: SQLValidationResult,
        execution_result: SQLExecutionResult | None,
        schema_matches: list[SchemaMatch],
        validation_attempts: int = 1,
    ) -> dict[str, Any]:
        rows = execution_result.rows if execution_result else []
        return {
            "source_type": EvidenceSource.SQL,
            "content": json.dumps(
                {
                    "sql": validation.sql or generated_sql.sql,
                    "rows": rows,
                    "validation": validation.reason,
                },
                ensure_ascii=False,
                default=str,
            ),
            "score": 1.0 if validation.allowed else 0.0,
            "source_id": "text2sql_result",
            "metadata": {
                "generated_sql": generated_sql.sql,
                "generation_reason": generated_sql.reason,
                "generation_mode": _generation_mode(generated_sql.reason),
                "validated_sql": validation.sql,
                "validation_allowed": validation.allowed,
                "validation_reason": validation.reason,
                "validation_attempts": validation_attempts,
                "validation_retry_count": max(0, validation_attempts - 1),
                "schema_matches": [
                    {"table": match.table.name, "score": match.score} for match in schema_matches
                ],
                "row_count": len(rows),
                "executor_type": self.executor.executor_type,
                "readonly": True,
            },
        }


_service: Text2SQLService | None = None
_service_lock = Lock()


def get_text2sql_service() -> Text2SQLService:
    # 惰性单例让 Text2SQL 链路只在被调用时初始化示例数据库，减少应用启动副作用。
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = _build_service()
    return _service


def reset_text2sql_service_for_tests(service: Text2SQLService | None = None) -> None:
    # 测试隔离用函数，避免全局 SQLite 状态影响用例之间的可重复性。
    global _service
    with _service_lock:
        _service = service


def _build_service() -> Text2SQLService:
    text2sql_dsn = settings.text2sql_postgres_dsn or settings.postgres_dsn
    if settings.strict_external_stores and not text2sql_dsn:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_TEXT2SQL_POSTGRES_DSN 或 GUSTOBOT_POSTGRES_DSN，不能使用 SQLite fallback。")
    if settings.strict_external_stores and (
        not settings.text2sql_llm_base_url
        or not settings.text2sql_llm_model
        or not settings.text2sql_llm_api_key
    ):
        raise RuntimeError("生产环境必须配置 GUSTOBOT_TEXT2SQL_LLM_BASE_URL、GUSTOBOT_TEXT2SQL_LLM_API_KEY 和 GUSTOBOT_TEXT2SQL_LLM_MODEL，不能使用规则 SQL 生成器。")
    if text2sql_dsn:
        schema_catalog = load_schema_catalog_from_postgres(
            text2sql_dsn,
            table_name=settings.text2sql_schema_table,
            embedding_provider=_build_schema_embedding_provider(),
        )
        if not schema_catalog.tables:
            if settings.strict_external_stores:
                raise RuntimeError("生产环境 schema_catalog 为空，不能回退到默认示例 schema。")
            schema_catalog = build_default_schema_catalog()
        executor: SQLExecutor = ReadOnlyPostgreSQLExecutor(text2sql_dsn)
    else:
        schema_catalog = build_default_schema_catalog()
        executor = ReadOnlySQLiteExecutor()

    allowed_tables = {table.name for table in schema_catalog.tables}
    rule_generator = RuleBasedSQLGenerator()
    sql_generator: SQLGenerator = rule_generator
    if settings.text2sql_llm_base_url and settings.text2sql_llm_model:
        sql_generator = LLMSQLGenerator(
            base_url=settings.text2sql_llm_base_url,
            api_key=settings.text2sql_llm_api_key,
            model=settings.text2sql_llm_model,
            timeout_seconds=settings.text2sql_llm_timeout_seconds,
            temperature=settings.text2sql_llm_temperature,
            fallback=rule_generator,
            fallback_on_error=not settings.strict_external_stores,
        )
    return Text2SQLService(
        schema_catalog=schema_catalog,
        sql_generator=sql_generator,
        sql_validator=SQLValidator(
            allowed_tables=allowed_tables,
            max_rows=settings.text2sql_max_rows,
        ),
        executor=executor,
        schema_top_k=settings.text2sql_schema_top_k,
        max_validation_retries=settings.text2sql_llm_max_validation_retries,
    )


def _build_schema_embedding_provider() -> EmbeddingProvider:
    if not settings.strict_external_stores:
        return HashEmbeddingProvider(dimension=64)
    provider = settings.kb_embedding_provider.strip().lower()
    if provider not in {"openai", "openai-compatible", "openai_compatible"}:
        raise RuntimeError("生产环境 Text2SQL schema catalog 必须使用真实 embedding provider，不能使用 hash。")
    if not settings.kb_embedding_base_url or not settings.kb_embedding_model or not settings.kb_embedding_api_key:
        raise RuntimeError("生产环境 Text2SQL schema catalog 必须配置真实 embedding base_url、model 和 api_key。")
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.kb_embedding_base_url,
        api_key=settings.kb_embedding_api_key,
        model=settings.kb_embedding_model,
        dimension=settings.kb_embedding_dimension,
        timeout_seconds=settings.kb_embedding_timeout_seconds,
    )


def _generation_mode(reason: str) -> str:
    if reason.startswith("llm_text2sql_failed_fallback") or reason.startswith("llm_text2sql_empty_fallback"):
        return "llm_fallback_rule"
    if reason.startswith("llm_text2sql"):
        return "llm"
    return "rule"
