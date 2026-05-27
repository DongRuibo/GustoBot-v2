# KB 原始资料目录

这个目录用于放置 GustoBot-v2 自己要读取和入库的知识库资料。

推荐流程：

1. 先把旧项目或外部来源中的资料复制到这个目录。
2. 运行 `python scripts/prepare_kb_data.py --dry-run` 检查能读出多少文档。
3. 配好 embedding / PostgreSQL + pgvector 后，运行 `python scripts/prepare_kb_data.py` 正式入库。

脚本会读取 `txt`、`md`、`json`、`jsonl`、`csv`、`xlsx` 文件。
表格类文件会按“每一行一篇文档”展开，方便后续做来源追踪和增量更新。
