# P0 DashScope Strict 替换实验

这份说明用于验证 P0 生产路径不再命中 HashEmbedding、内存存储、seed 图谱、规则 SQL、关键词 rerank、文件占位解析等演示实现。

## 1. 配置

默认开发启动仍使用 `docker-compose.yml`。P0 验收请叠加 `docker-compose.p0.yml`：

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml up -d --build api
```

P0 override 会设置：

- `GUSTOBOT_ENV=prod`
- `GUSTOBOT_STRICT_EXTERNAL_STORES=true`
- `GUSTOBOT_KB_EMBEDDING_PROVIDER=openai-compatible`
- `GUSTOBOT_KB_EMBEDDING_MODEL=text-embedding-v4`
- `GUSTOBOT_KB_EMBEDDING_DIMENSION=1024`
- `GUSTOBOT_KB_RERANK_MODEL=qwen3-rerank`

真实 DashScope Key 只放本地 `.env`，不要提交。v2 会复用旧项目变量：

- `LLM_*` -> Router / Text2SQL / Answer
- `VISION_BASE_URL` / `VISION_API_KEY` -> 图片理解 Vision
- `EMBEDDING_*` -> KB embedding
- `RERANK_*` -> KB rerank

图片理解链路会调用 OpenAI-compatible `/chat/completions` 并要求模型返回文本/JSON，P0 override 默认使用
`qwen3-vl-plus`。如果本地 `.env` 里 `VISION_MODEL` 或 `IMAGE_GENERATION_MODEL` 是 `qwen-image-2.0-pro`，
请把它继续留给图片生成/编辑；P0 图片理解如需改模型，请设置 `VISION_UNDERSTANDING_MODEL=qwen-vl-plus`
或 `qwen-vl-max` 等支持 `image_url` 输入的视觉理解模型。

## 2. 数据准备

建议在 API 容器内执行，避免宿主机和容器网络地址不一致：

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/import_real_recipe_data.py --reset
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/import_ingredient_taxonomy.py
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/prepare_kb_data.py --reset
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/import_real_recipe_data.py --ingest-kb
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/bootstrap_neo4j_from_postgres.py --reset
```

其中 `prepare_kb_data.py --reset` 会用 DashScope `text-embedding-v4` 重建本地 KB 文档向量；`bootstrap_neo4j_from_postgres.py --reset` 只从 PostgreSQL 真实菜谱数据重建 Neo4j 图谱。

## 3. Readiness 检查

只检查配置和旧变量映射：

```powershell
python scripts/check_p0_dashscope_readiness.py --config-only
```

检查配置和运行时 provider/store：

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/check_p0_dashscope_readiness.py
```

通过时应看到：

- KB: `postgres_pgvector` + `openai-compatible` + `http`
- GraphRAG: `neo4j`
- Text2SQL: `postgres_readonly` + `LLMSQLGenerator` + `schema_embedding_provider=openai-compatible`
- Cache: `redis`
- Session/Upload: PostgreSQL 存储类

输出只包含 provider/model/store 类型和 key 是否已配置，不会打印 API Key 或数据库 DSN。

P0 override 启动 API 时也会执行同一套 readiness 门禁。`GUSTOBOT_ENV=prod` 且
`GUSTOBOT_STRICT_EXTERNAL_STORES=true` 时，如果命中 hash、memory、seed、sqlite、rule、keyword
或错误 Vision 模型，API 会在启动阶段 fail-fast。

## 4. 失败注入

失败注入脚本只临时覆盖子进程环境变量，不修改 `.env`，也不会停止 Docker 服务。建议在 API 容器内运行：

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/p0_failure_injection.py
```

也可以只运行指定 case：

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/p0_failure_injection.py --cases missing_model_keys bad_redis_runtime
```

当前覆盖的 P0 注入项包括：模型 key 缺失、hash embedding、embedding 维度错误、rerank base_url 缺失、
Vision 误用 qwen-image、PostgreSQL/Neo4j/Redis 运行时地址不可达。

## 5. Smoke 验收

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/docker_smoke.py --expect-embedding-provider openai-compatible --expect-reranker-success --expect-answer-mode llm --expect-real-data
```

Smoke 需要真实 DashScope、PostgreSQL、Neo4j、Redis 全部可用。若任一服务失败，P0 strict 应返回明确错误，不应静默退回 hash、memory、seed、rule 或 keyword 实现。
