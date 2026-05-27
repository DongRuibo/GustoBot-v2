"""Schema Catalog 模块。

这个文件定义表、字段和 schema 检索结果，用于解决“大量表结构不能全部塞给模型”的问题。
当前用本地示例表和 hash embedding 做最小可运行检索，后续可以接 pgvector 存 schema embedding。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.kb.embeddings import EmbeddingProvider, HashEmbeddingProvider, cosine_similarity, tokenize_for_retrieval


@dataclass(slots=True)
class ColumnSchema:
    # ColumnSchema 描述一个字段的技术信息和业务含义，Text2SQL 生成 SQL 时会优先使用这些字段说明。
    name: str
    data_type: str
    comment: str
    sample_values: tuple[Any, ...] = ()


@dataclass(slots=True)
class TableSchema:
    # TableSchema 是 Schema Catalog 的核心对象，包含表名、业务说明、字段、模块和样例信息。
    name: str
    comment: str
    columns: list[ColumnSchema]
    business_meaning: str
    module: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_search_text(self) -> str:
        # 检索文本把表名、字段名、注释、样例值和业务含义合并起来，模拟 schema embedding 入库内容。
        column_text = " ".join(
            f"{column.name} {column.data_type} {column.comment} {' '.join(map(str, column.sample_values))}"
            for column in self.columns
        )
        return f"{self.name} {self.comment} {self.business_meaning} {self.module} {column_text}"


@dataclass(slots=True)
class SchemaMatch:
    # SchemaMatch 是 Schema Catalog 检索结果，score 越高说明这张表越可能和用户问题相关。
    table: TableSchema
    score: float


class SchemaCatalog:
    # SchemaCatalog 负责从大量表结构中召回少数相关表；当前只有示例表，但接口按大规模场景设计。
    def __init__(self, tables: list[TableSchema], embedding_provider: EmbeddingProvider) -> None:
        self.tables = tables
        self.embedding_provider = embedding_provider
        self._table_embeddings = {
            table.name: embedding_provider.embed(table.to_search_text()) for table in tables
        }

    def retrieve(self, question: str, *, top_k: int) -> list[SchemaMatch]:
        query_embedding = self.embedding_provider.embed(question)
        query_tokens = set(tokenize_for_retrieval(question))
        matches: list[SchemaMatch] = []
        for table in self.tables:
            vector_score = cosine_similarity(query_embedding, self._table_embeddings[table.name])
            table_tokens = set(tokenize_for_retrieval(table.to_search_text()))
            overlap_score = len(query_tokens & table_tokens) / max(len(query_tokens), 1)
            # 向量分负责宽召回，关键词重叠负责把明显命中的业务表排到前面。
            score = max(0.0, min(1.0, vector_score * 0.65 + overlap_score * 0.35))
            matches.append(SchemaMatch(table=table, score=score))
        return sorted(matches, key=lambda item: item.score, reverse=True)[:top_k]


def build_default_schema_catalog() -> SchemaCatalog:
    # 示例 Schema Catalog 模拟菜谱业务结构化表，后续可以由数据库 introspection 或文件入库自动生成。
    recipes = TableSchema(
        name="recipes",
        comment="菜谱主表，保存菜名、菜系、难度、耗时和热度等结构化字段。",
        business_meaning="用于回答菜谱数量统计、菜系统计、排名、平均耗时和趋势类问题。",
        module="recipe_analytics",
        updated_at="2026-04-27",
        columns=[
            ColumnSchema("recipe_id", "integer", "菜谱唯一编号", (1, 2, 3)),
            ColumnSchema("name", "text", "菜谱名称", ("宫保鸡丁", "麻婆豆腐")),
            ColumnSchema("cuisine", "text", "菜系名称", ("川菜", "粤菜", "闽菜")),
            ColumnSchema("difficulty", "text", "制作难度", ("简单", "中等", "困难")),
            ColumnSchema("cooking_time_minutes", "integer", "烹饪耗时，单位分钟", (15, 25, 90)),
            ColumnSchema("popularity", "integer", "热度分，用于排名分析", (82, 95, 76)),
            ColumnSchema("created_year", "integer", "菜谱录入年份，用于趋势分析", (2024, 2025, 2026)),
        ],
    )
    return SchemaCatalog(
        tables=[recipes],
        embedding_provider=HashEmbeddingProvider(dimension=64),
    )


def load_schema_catalog_from_postgres(
    dsn: str,
    *,
    table_name: str = "schema_catalog",
    embedding_provider: EmbeddingProvider | None = None,
) -> SchemaCatalog:
    """从 PostgreSQL schema catalog 表读取 Text2SQL 可见表结构。

    这个函数只读取经过 catalog 白名单暴露的结构，不直接把数据库所有表暴露给 SQL 生成器。
    如果 catalog 为空，调用方应回退到默认 catalog 或提示先执行初始化脚本。
    """
    safe_table_name = _validate_catalog_table_name(table_name)
    psycopg, dict_row = _ensure_postgres_driver()
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT table_name,
                       table_comment,
                       business_meaning,
                       module,
                       columns,
                       metadata,
                       updated_at
                FROM {safe_table_name}
                ORDER BY table_name
                """
            )
            rows = cursor.fetchall()

    tables = [_table_schema_from_catalog_row(row) for row in rows]
    return SchemaCatalog(
        tables=tables,
        embedding_provider=embedding_provider or HashEmbeddingProvider(dimension=64),
    )


def _table_schema_from_catalog_row(row: dict[str, Any]) -> TableSchema:
    columns_payload = row.get("columns") or []
    columns = [
        ColumnSchema(
            name=str(item.get("name", "")),
            data_type=str(item.get("data_type", "text")),
            comment=str(item.get("comment", "")),
            sample_values=tuple(item.get("sample_values", ())),
        )
        for item in columns_payload
        if isinstance(item, dict) and item.get("name")
    ]
    return TableSchema(
        name=str(row["table_name"]),
        comment=str(row.get("table_comment") or ""),
        columns=columns,
        business_meaning=str(row.get("business_meaning") or ""),
        module=str(row.get("module") or ""),
        updated_at=str(row.get("updated_at") or ""),
        metadata=dict(row.get("metadata") or {}),
    )


def _validate_catalog_table_name(table_name: str) -> str:
    # 表名来自环境变量，只允许普通标识符，避免 schema catalog 查询产生注入风险。
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(f"非法 schema catalog 表名：{table_name}")
    return table_name


def _ensure_postgres_driver():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "已配置 Text2SQL PostgreSQL，但当前环境缺少 psycopg。"
            "请安装 psycopg[binary]，或取消 GUSTOBOT_TEXT2SQL_POSTGRES_DSN 使用本地 fallback。"
        ) from exc
    return psycopg, dict_row
