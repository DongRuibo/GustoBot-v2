"""SQL 只读执行模块。

这个文件提供 Text2SQL 的执行层：正式链路使用 PostgreSQL 只读事务，
未配置数据库时使用 SQLite 内存示例库作为本地 fallback。
所有执行器都只接收经过 SQLValidator 校验后的 SELECT 查询。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol


@dataclass(slots=True)
class SQLExecutionResult:
    # SQLExecutionResult 保存执行结果和列名，答案生成和 Evidence 都会使用这些信息。
    columns: list[str]
    rows: list[dict[str, Any]]


class SQLExecutor(Protocol):
    # Text2SQL 执行器协议：候选 SQL 必须先通过 SQLValidator，再由这里只读执行。
    executor_type: str

    def execute(self, sql: str) -> SQLExecutionResult:
        ...


class ReadOnlySQLiteExecutor:
    # SQLite 执行器只用于本地最小闭环；真实生产环境应使用 PostgreSQL 只读账号连接。
    executor_type = "sqlite_memory_readonly"

    def __init__(self) -> None:
        self._lock = RLock()
        self._connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._seed()
        self._connection.execute("PRAGMA query_only = ON")

    def execute(self, sql: str) -> SQLExecutionResult:
        with self._lock:
            cursor = self._connection.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [description[0] for description in cursor.description or []]
        return SQLExecutionResult(columns=columns, rows=rows)

    def _seed(self) -> None:
        # 示例数据只用于本地开发和测试，覆盖菜系数量、热度排名、平均耗时和趋势统计。
        self._connection.execute(
            """
            CREATE TABLE recipes (
                recipe_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                cuisine TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                cooking_time_minutes INTEGER NOT NULL,
                popularity INTEGER NOT NULL,
                created_year INTEGER NOT NULL
            )
            """
        )
        self._connection.executemany(
            """
            INSERT INTO recipes
                (recipe_id, name, cuisine, difficulty, cooking_time_minutes, popularity, created_year)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (1, "宫保鸡丁", "川菜", "中等", 25, 96, 2024),
                (2, "麻婆豆腐", "川菜", "简单", 18, 94, 2024),
                (3, "鱼香肉丝", "川菜", "中等", 30, 88, 2025),
                (4, "白灼虾", "粤菜", "简单", 12, 82, 2025),
                (5, "叉烧", "粤菜", "中等", 90, 89, 2026),
                (6, "佛跳墙", "闽菜", "困难", 180, 91, 2026),
                (7, "白菜炖豆腐", "家常菜", "简单", 20, 76, 2026),
            ],
        )
        self._connection.commit()


class ReadOnlyPostgreSQLExecutor:
    # PostgreSQL 执行器用于 v2 正式 Text2SQL 链路。生产环境建议搭配数据库只读账号。
    executor_type = "postgres_readonly"

    def __init__(self, dsn: str, *, statement_timeout_ms: int = 5000) -> None:
        self.dsn = dsn
        self.statement_timeout_ms = statement_timeout_ms

    def execute(self, sql: str) -> SQLExecutionResult:
        psycopg, dict_row = self._ensure_driver()
        with psycopg.connect(self.dsn, row_factory=dict_row) as connection:
            with connection.cursor() as cursor:
                # 即使上游校验只允许 SELECT，也在数据库事务层再加只读约束。
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute(f"SET LOCAL statement_timeout = {int(self.statement_timeout_ms)}")
                cursor.execute(sql)
                rows = [dict(row) for row in cursor.fetchall()]
                columns = list(rows[0].keys()) if rows else [
                    column.name for column in (cursor.description or [])
                ]
        return SQLExecutionResult(columns=columns, rows=rows)

    @staticmethod
    def _ensure_driver():
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError(
                "已配置 Text2SQL PostgreSQL，但当前环境缺少 psycopg。"
                "请安装 psycopg[binary]，或取消 GUSTOBOT_TEXT2SQL_POSTGRES_DSN 使用 SQLite fallback。"
            ) from exc
        return psycopg, dict_row
