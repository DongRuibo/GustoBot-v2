"""SQL 安全校验模块。

这个文件负责校验 LLM 或规则生成的候选 SQL 是否允许执行。
它只允许单条 SELECT 查询，并拦截写操作、DDL、多语句、注释和未知表，保证最终执行由程序控制。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


WRITE_OR_DDL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|merge|replace|grant|revoke|vacuum|attach|detach|pragma)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class SQLValidationResult:
    # 校验结果会进入 Text2SQL Evidence，方便追踪 SQL 为什么被允许或拒绝。
    allowed: bool
    sql: str
    reason: str


class SQLValidator:
    # SQLValidator 是 Text2SQL 的安全闸门：生成器只给候选 SQL，是否执行必须由这里决定。
    def __init__(self, *, allowed_tables: set[str], max_rows: int) -> None:
        self.allowed_tables = {table.lower() for table in allowed_tables}
        self.max_rows = max_rows

    def validate(self, sql: str) -> SQLValidationResult:
        stripped = sql.strip()
        if not stripped:
            return SQLValidationResult(False, "", "SQL 为空，拒绝执行。")

        if "--" in stripped or "/*" in stripped or "*/" in stripped:
            return SQLValidationResult(False, stripped, "SQL 包含注释，拒绝执行。")

        semicolon_count = stripped.count(";")
        if semicolon_count > 1 or (semicolon_count == 1 and not stripped.endswith(";")):
            return SQLValidationResult(False, stripped, "SQL 包含多语句风险，拒绝执行。")

        normalized = stripped[:-1].strip() if stripped.endswith(";") else stripped
        if not normalized.lower().startswith("select "):
            return SQLValidationResult(False, normalized, "只允许 SELECT 只读查询。")

        if WRITE_OR_DDL_PATTERN.search(normalized):
            return SQLValidationResult(False, normalized, "SQL 命中写操作或 DDL 关键字，拒绝执行。")

        referenced_tables = _extract_referenced_tables(normalized)
        if not referenced_tables:
            return SQLValidationResult(False, normalized, "无法识别 FROM/JOIN 表名，拒绝执行。")

        unknown_tables = referenced_tables - self.allowed_tables
        if unknown_tables:
            return SQLValidationResult(
                False,
                normalized,
                f"SQL 引用了未在 Schema Catalog 中允许的表：{', '.join(sorted(unknown_tables))}。",
            )

        limited_sql = _ensure_limit(normalized, self.max_rows)
        return SQLValidationResult(True, limited_sql, "SQL 通过只读安全校验。")


def _extract_referenced_tables(sql: str) -> set[str]:
    # 简化版表名提取只处理当前生成器会产生的 FROM/JOIN 形态；复杂 SQL 后续建议接 sqlglot。
    matches = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", sql, flags=re.IGNORECASE)
    return {match.lower() for match in matches}


def _ensure_limit(sql: str, max_rows: int) -> str:
    # 没有 LIMIT 的查询统一追加行数上限，避免误生成大结果集拖慢服务。
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {max_rows}"

