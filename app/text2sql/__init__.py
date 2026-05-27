"""Text2SQL 子包。

这个子包实现第三阶段的结构化数据问答能力，负责 Schema Catalog 检索、
候选 SQL 生成、安全校验、只读执行和 SQL Evidence 输出。
配置 PostgreSQL DSN 时使用正式只读链路；未配置时使用本地 SQLite 示例表作为 fallback。
"""
