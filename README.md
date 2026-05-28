# GustoBot-v2

GustoBot-v2 是一个基于 FastAPI 和 LangGraph 的菜谱领域多路由智能问答系统。项目在 `GustoBot-v2` 中从头搭建，不改动旧的 `GustoBot-develop`。

当前仓库状态：后端已接入 FastAPI + LangGraph 多路由工作流，前端为 Vue 3 + Vite 本地聊天应用；Docker Compose 可同时启动 API、PostgreSQL + pgvector、Neo4j、Redis 和 Web；数据侧已包含 USDA + FoodOn 清洗后的 processed 食品数据、评估集、Router/Answer SFT 数据和多种 Router 训练导出格式。默认本地模式保留内存库、hash embedding、关键词 reranker、SQLite 等 fallback，便于无外部依赖联调；生产/P0 路径已支持真实 embedding、HTTP reranker、PostgreSQL、Neo4j、Redis 和 strict 外部依赖校验。

## 当前阶段状态

### 第一阶段：主骨架，已完成

- FastAPI 项目骨架
- LangGraph 主流程
- Global Guardrails
- Input Preprocess 占位
- Router
- General 节点
- Clarify 节点
- Evidence Normalizer
- Answer Generator / Answer Guardrails
- 基础测试

### 第二阶段：KB RAG，已完成可运行版本

- 文档切块
- 默认 hash embedding fallback，便于本地测试和流程联调
- OpenAI-compatible embedding provider，可接 bge、text-embedding 或兼容 `/v1/embeddings` 的服务
- 内存知识库存储，便于本地测试
- PostgreSQL + pgvector 存储适配器
- pgvector 相似度检索 SQL
- 关键词 reranker fallback
- HTTP reranker 适配器，可接 bge-reranker、DashScope qwen3-rerank 或兼容 rerank 服务
- KB Evidence 输出
- `/api/v1/kb/documents` 文档入库接口
- `kb_rag_node` 已接入 LangGraph 主流程
- `GUSTOBOT_ENV=prod` 或 `GUSTOBOT_STRICT_EXTERNAL_STORES=true` 时会阻止静默退回 hash embedding、内存库或关键词 reranker

说明：当前默认本地模式使用内存存储、hash embedding 和关键词 reranker，目标是让开发环境开箱可跑；配置 `GUSTOBOT_POSTGRES_DSN`、`GUSTOBOT_KB_EMBEDDING_*` 和 `GUSTOBOT_KB_RERANK_*` 后，会切换到 PostgreSQL + pgvector、真实 embedding 和外部 reranker。

### 第三阶段：GraphRAG + Text2SQL，已完成最小可运行版本

- GraphRAG 内存图谱
- Neo4j 图谱适配器
- 实体识别和实体链接
- 有限跳数子图提取
- 图谱 Evidence 输出
- PostgreSQL Schema Catalog
- 规则 SQL 生成器
- SQL 安全校验
- PostgreSQL 只读执行器，未配置数据库时使用 SQLite fallback
- SQL Evidence 输出
- `graphrag_node` 和 `text2sql_node` 已接入 LangGraph 主流程

说明：GraphRAG 默认使用内存种子图谱；配置 `GUSTOBOT_NEO4J_URI` 后会切换到 Neo4j。Text2SQL 在配置 `GUSTOBOT_TEXT2SQL_POSTGRES_DSN` 或 `GUSTOBOT_POSTGRES_DSN` 后使用 PostgreSQL 业务表和 `schema_catalog`；未配置数据库时才使用 SQLite 内存示例表作为本地 fallback。

### 第四阶段：工程化，已完成最小可运行版本

- 图片理解服务：支持 OpenAI-compatible Vision LLM、通用 OCR HTTP 服务和文件名/附件 text 降级
- 图片转结构化文本后重新进入 Router
- 文件附件解析和 KB 入库：支持文本、CSV、Excel、PDF、docx 段落与表格
- `/api/v1/files/ingest` 文件入库接口
- Redis 缓存适配器，默认内存缓存
- 请求级 JSONL 日志追踪
- 离线评估脚本 `scripts/evaluate.py`
- Dockerfile
- Docker Compose：FastAPI、PostgreSQL + pgvector、Neo4j、Redis

说明：图片理解会在配置 `GUSTOBOT_VISION_*` / `GUSTOBOT_OCR_*` 后调用真实服务，并把结构化结果重新送回 Router；未配置或调用失败时保留文件名和附件 `text` 降级。文件入库已支持 PDF、Excel、docx 等二进制附件解析，解析后的文本仍统一写入 KB。

## 主流程

```text
用户输入
-> Global Guardrails
-> Input Preprocess
-> LangGraph Router
-> General / KB RAG / GraphRAG / Text2SQL / Image / File / Clarify
-> Evidence Normalizer
-> Answer Generator
-> Answer Guardrails
```

Router 已保留后续路由类型：

- `general`
- `kb`
- `graphrag`
- `text2sql`
- `image`
- `file`
- `clarify`

当前阶段中，`kb`、`graphrag`、`text2sql`、`image`、`file` 都已接入执行链路。图片链路会先理解图片并重新进入 Router；文件链路会解析附件文本并写入 KB。

## 面试展示视角

### 架构亮点

- 多路由问答：Router 先判断问题类型，再分流到 KB RAG、GraphRAG、Text2SQL、图片理解或文件入库，避免把所有问题都塞进同一个 RAG。
- Evidence 优先：KB、GraphRAG、Text2SQL 都只产出答案草稿和 Evidence，统一答案层只消费 Evidence，不直接访问数据库或图谱。
- 真实数据链路清晰：dev/test 可以使用内存 fallback；真实部署推荐 KB=PostgreSQL + pgvector、GraphRAG=Neo4j、缓存=Redis。
- 关系型问题走图谱：菜谱、食材、步骤、菜系、工具等结构化关系由 GraphRAG 负责，统计分析由 Text2SQL 负责。

### 关键问题与解决方案

- 食材上位词泛化：旧逻辑只匹配具体 `Ingredient`，例如“猪肉”无法自动关联“猪排骨、猪肉里脊、五花肉”。当前已升级为 `IngredientCategory` 上位词图谱，通过 `BELONGS_TO_CATEGORY` 和 `IS_A` 查询“类别 -> 具体食材 -> 菜谱”。
- 路由意图冲突：“多少”既可能是统计，也可能是食材用量。当前 Router 在配置 LLM/SFT 时由模型做主体判断，规则只作为轻量特征和失败兜底；未配置模型时，规则兜底仍会优先把“菜谱 + 食材 + 用量/放多少/几克/几勺”识别为 GraphRAG，并保留“菜系有多少道菜谱”等统计问题走 Text2SQL。
- 生产环境可信度：`GUSTOBOT_ENV=prod` 时建议开启严格外部存储校验，避免服务看似启动成功但实际退回内存种子数据。

### 演示问题清单

- `介绍一下宫保鸡丁的历史和文化`：KB RAG + Evidence 引用。
- `宫保鸡丁需要哪些食材`：GraphRAG 菜谱到食材。
- `宫保鸡丁里鸡肉用量是多少`：GraphRAG 菜谱-食材用量模板。
- `猪肉可以做什么菜`：IngredientCategory 上位词泛化到具体食材。
- `统计一下每个菜系的菜谱数量`：Text2SQL 只读统计查询。

## 当前数据集与数据资产

当前仓库不仅包含菜谱演示数据，也提交了一版可复现实验用的食品数据底座。原始大文件位于 `data/raw/external/`，该目录已被 `.gitignore` 忽略；GitHub 仓库中保留的是清洗后的 processed 数据、评估样本和 SFT 样本。

数据来源：

- USDA Foundation Foods 04/2026：基础食品营养数据。
- USDA SR Legacy 04/2018：传统标准食品营养数据。
- USDA Branded Foods 04/2026：品牌商品食品数据。
- FoodOn 同义词表：食品类别、别名和上位词关系；OWL 原文件只记录在 manifest 中，不作为当前运行时直接解析输入。
- Open Food Facts 本轮未进入正式 processed 数据；第一版使用 USDA Branded Foods 补足商品食品数据。

当前 processed 数据规模：

| 数据资产 | 文件位置 | 当前规模 | 主要用途 |
| --- | --- | --- | --- |
| 食品商品主表 | `data/processed/food_products.csv`、`data/processed/products.csv` | 15522 条商品 | Text2SQL 统计、GraphRAG 产品节点、KB 文档生成 |
| 营养素长表 | `data/processed/food_nutrients.csv`、`data/processed/nutrients.csv` | 215321 条营养记录 | 营养查询、排序、统计分析 |
| 图谱节点 | `data/processed/graph_nodes.csv` | 15777 个节点 | Neo4j 商品、配料、过敏原、分类、营养素节点 |
| 图谱边 | `data/processed/graph_edges.csv` | 74355 条边 | Neo4j 产品-配料、产品-过敏原、产品-分类、产品-营养关系 |
| KB 文档 | `data/processed/kb_documents.jsonl` | 2406 条文档 | PostgreSQL + pgvector 或内存 KB 检索 |
| 食品评估集 | `data/eval/food_eval.jsonl` | 230 条样本 | Router、Evidence、引用覆盖率评估 |
| Router SFT | `data/sft/router_*.jsonl` | 800 条，train/dev/test = 640/80/80 | 训练结构化 Router |
| Answer SFT | `data/sft/answer_*.jsonl` | 300 条，train/dev/test = 240/30/30 | 训练基于 Evidence 的答案生成 |
| Router 训练导出 | `data/sft/router_training/` | OpenAI、ms-swift、DashScope、LLaMA-Factory 格式 | 云端或本地 Router SFT |

正式评估结果保存在 `reports/eval_after_food_data.json`，当前食品数据闭环的 Router Accuracy、Evidence Source Coverage、Citation Coverage 和危险输入拦截率均为 100%。第一轮未达标报告保留在 `reports/eval_after_food_data_first_run.json`，用于追踪规则修复前后的差异。

说明：`reports/eval_after_food_data.json` 是当前 GitHub 稳定基线。本机可额外生成 Router SFT 对比报告，例如 `reports/eval_router_local_sft.json` 或云端 Router 报告；这类文件用于实验对比，不代表当前默认稳定配置，未纳入本轮 README-only 同步范围。

数据重建和导入入口：

```powershell
python scripts\data\prepare_food_dataset.py --raw-root data\raw\external --sr-limit 8000 --branded-limit 8000 --graph-product-limit 12000 --max-kb-products 2000
docker compose exec -T api python scripts\data\import_food_dataset.py --products data\processed\food_products.csv --nutrients data\processed\food_nutrients.csv --kb-documents data\processed\kb_documents.jsonl --reset --ingest-kb
docker compose exec -T api python scripts\data\bootstrap_food_neo4j.py --nodes data\processed\graph_nodes.csv --edges data\processed\graph_edges.csv --reset
```

更完整的数据准备、评估和 SFT 生成流程见 `docs/DATASET_EXPANSION_PLAN.md` 与 `docs/SFT_DATASET_DESIGN.md`。

### LLM Router

Router 采用 `hard rules + LLM/SFT Router + rule fallback` 的混合模式：图片、文件、明确问候和低信息输入先由确定性规则处理；KB / GraphRAG / Text2SQL 等主体业务文本优先调用 OpenAI-compatible 或 DashScope LLM/SFT 生成结构化 `RouteDecision`。如果 LLM 未配置、超时、返回非法 JSON、未知 route 或低置信度，系统会自动回退到规则 Router。

```powershell
$env:GUSTOBOT_ROUTER_LLM_ENABLED="true"
$env:GUSTOBOT_ROUTER_LLM_BASE_URL="http://127.0.0.1:8001/v1"
$env:GUSTOBOT_ROUTER_LLM_API_KEY="your-api-key"
$env:GUSTOBOT_ROUTER_LLM_MODEL="your-chat-model"
$env:GUSTOBOT_ROUTER_LLM_TEMPERATURE="0"
```

未配置 `GUSTOBOT_ROUTER_LLM_*` 时会兼容读取 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`。路由结果的 `slots` 中会记录 `router_provider`、`router_model`、`fallback_used` 和 `fallback_reason`，便于 trace 排查。

Router SFT 当前定位为可插拔路由后端，而不是替代 KB、GraphRAG 或 Text2SQL。仓库已包含：

- `data/sft/router_*.jsonl`：Router SFT train/dev/test 数据。
- `data/sft/router_training/`：OpenAI、ms-swift、DashScope、LLaMA-Factory 导出格式。
- `scripts/data/dashscope_router_finetune.py`：DashScope 微调管理脚本，默认 dry-run，真实上传/训练/部署需显式 `--submit`。
- `scripts/sft/`：本地 QLoRA、LoRA 合并和 Router 服务部署脚本，适合在有 GPU 的机器上做本地实验。

Router SFT 接回主流程后仍建议通过 `scripts/evaluate.py --samples data/eval/food_eval.jsonl` 做基线对比，重点看 GraphRAG / Text2SQL / KB 的混淆样本，而不是只看单条演示问题。

## 应用外壳与前端

当前应用外壳提供会话、消息历史、上传、回答快照和本地前端壳，不改变 KB RAG、GraphRAG、Text2SQL 的核心链路。

- `/api/v1/chat` 会自动创建或复用会话，并返回 `session_id`、`message_id`。
- `/api/v1/chat/stream` 返回 NDJSON 增量事件，前端用它实现流式回答。
- `/api/v1/sessions` 提供会话列表、新建、详情、更新、软删除和消息历史查询。
- `/api/v1/sessions/{session_id}/snapshots` 提供每轮回答快照，便于回看 Evidence 和 trace。
- `/api/v1/upload/file`、`/api/v1/upload/image` 使用 multipart 上传，并返回可放入 chat/files ingest 的 `upload://{file_id}` 附件；上传文件也支持读取和删除。
- `upload://` 只解析服务端已登记文件，不允许业务链路读取任意本地路径。
- `web/` 是 Vue 3 + Vite + TypeScript 本地聊天应用，默认读取 `VITE_API_BASE_URL`，未设置时通过 Vite proxy 请求 `http://localhost:8000`。

本地前端开发：

```powershell
cd web
npm install
npm run dev
```

## KB RAG

默认本地模式：

- 使用内存存储
- 启动时写入少量种子菜谱知识
- 不依赖 PostgreSQL 服务
- 默认使用 hash embedding，仅用于本地测试和流程联调

真实 embedding 模式：

```powershell
$env:GUSTOBOT_KB_EMBEDDING_PROVIDER="openai-compatible"
$env:GUSTOBOT_KB_EMBEDDING_BASE_URL="http://127.0.0.1:8001/v1"
$env:GUSTOBOT_KB_EMBEDDING_API_KEY="your-embedding-api-key"
$env:GUSTOBOT_KB_EMBEDDING_MODEL="bge-m3"
$env:GUSTOBOT_KB_EMBEDDING_DIMENSION="1024"
```

说明：

- `GUSTOBOT_KB_EMBEDDING_BASE_URL` 需要兼容 `/v1/embeddings`
- `GUSTOBOT_KB_EMBEDDING_DIMENSION` 必须等于模型实际返回维度
- 切换 embedding 模型或维度后，已有 pgvector 表需要重建或重新建库，否则 vector 维度会不一致
- 也兼容旧项目变量名：`EMBEDDING_PROVIDER`、`EMBEDDING_MODEL`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_DIMENSION`

真实 reranker 模式：

```powershell
$env:GUSTOBOT_KB_RERANK_BASE_URL="http://127.0.0.1:8002"
$env:GUSTOBOT_KB_RERANK_ENDPOINT="/rerank"
$env:GUSTOBOT_KB_RERANK_MODEL="bge-reranker"
```

未配置 reranker 服务时，系统会使用关键词重排 fallback；配置后会调用外部 rerank 服务，服务不可用时自动退回向量召回顺序。

PostgreSQL + pgvector 模式：

```powershell
$env:GUSTOBOT_POSTGRES_DSN="postgresql://user:password@127.0.0.1:5432/gustobot"
uvicorn app.main:app --reload
```

首次使用时会尝试创建：

- `vector` extension
- `kb_documents`
- `kb_chunks`
- metadata GIN 索引
- embedding HNSW 索引

如果数据库账号没有 `CREATE EXTENSION` 权限，需要先由 DBA 或初始化脚本安装 pgvector。

兼容旧项目数据：

- 旧项目 pgvector 表名是 `searchable_documents`
- 如果使用旧项目的 `PGHOST`、`PGDATABASE`、`PGUSER`、`PGPASSWORD` 等变量，v2 会默认查询 `searchable_documents`
- 也可以显式设置：

```powershell
$env:GUSTOBOT_KB_PGVECTOR_TABLE="searchable_documents"
```

如果使用 v2 自己的新表结构，则默认表是 `kb_chunks`。

KB 状态检查：

```powershell
curl http://127.0.0.1:8000/api/v1/kb/status
```

返回当前存储类型、embedding provider、模型名、向量维度和 chunk 数，不会暴露 DSN 或 API Key。

文档入库接口：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/kb/documents `
  -H "Content-Type: application/json" `
  -d "{\"title\":\"宫保鸡丁资料\",\"content\":\"宫保鸡丁是一道经典川菜，与丁宝桢相关。\",\"source_id\":\"doc-gongbao\"}"
```

入库后，知识解释类问题会走 KB RAG：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"介绍一下宫保鸡丁的历史和文化\"}"
```

## GraphRAG

默认本地模式：

- 使用内存图谱
- 启动时包含少量菜谱、食材、步骤、菜系、口味节点
- 不依赖 Neo4j 服务

Neo4j 模式：

```powershell
$env:GUSTOBOT_NEO4J_URI="bolt://127.0.0.1:17687"
$env:GUSTOBOT_NEO4J_USERNAME="neo4j"
$env:GUSTOBOT_NEO4J_PASSWORD="password"
uvicorn app.main:app --reload
```

将内存种子图谱导入 Neo4j：

```powershell
python scripts/bootstrap_neo4j.py
```

关系类问题会走 GraphRAG：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"宫保鸡丁需要哪些食材\"}"
```

GraphRAG 的职责边界：

- 适合回答食材、菜谱、步骤、菜系、口味之间的关系问题
- 不承担所有知识解释类问题
- 历史、文化、典故类问题仍优先走 KB RAG

食材上位词泛化：

- 真实图谱会写入 `IngredientCategory` 节点，例如猪肉、瘦肉、绿叶菜等。
- 具体食材通过 `BELONGS_TO_CATEGORY` 连接到类别，类别之间通过 `IS_A` 表示上位层级。
- 修改 `app/graphrag/ingredient_taxonomy.py` 或重建真实菜谱数据后，需要重新执行 Neo4j 导入脚本；否则正在运行的 Neo4j 仍然是旧类别边。
- 使用真实 PostgreSQL 数据导入 Neo4j 时，可通过 `--report-uncategorized` 输出未归类食材，便于后续补充 taxonomy 规则。

```powershell
python scripts/bootstrap_neo4j_from_postgres.py --reset --report-uncategorized
```

## Text2SQL

默认本地模式：

- 未配置 PostgreSQL DSN 时使用 SQLite 内存示例表 `recipes`
- 使用 Schema Catalog 检索相关表结构
- 优先使用低温 Text2SQL LLM 生成候选 SQL；未配置或调用失败时回退到规则生成器
- 使用 SQLValidator 拦截非 SELECT、写操作、多语句、未知表
- 使用只读执行器执行通过校验的 SQL

PostgreSQL 模式：

```powershell
$env:GUSTOBOT_TEXT2SQL_POSTGRES_DSN="postgresql://gustobot:gustobot@127.0.0.1:5432/gustobot"
$env:GUSTOBOT_TEXT2SQL_SCHEMA_TABLE="schema_catalog"
$env:GUSTOBOT_TEXT2SQL_LLM_BASE_URL="http://127.0.0.1:8001/v1"
$env:GUSTOBOT_TEXT2SQL_LLM_API_KEY="your-api-key"
$env:GUSTOBOT_TEXT2SQL_LLM_MODEL="your-chat-model"
uvicorn app.main:app --reload
```

说明：

- Text2SQL 只读取 `schema_catalog` 中登记的表结构，不直接暴露数据库全部表。
- SQL 生成器只生成候选 SQL，最终是否执行由 `SQLValidator` 和只读执行器共同决定。
- 生产环境建议为 `GUSTOBOT_TEXT2SQL_POSTGRES_DSN` 配置数据库只读账号。

统计分析类问题会走 Text2SQL：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"统计一下每个菜系的菜谱数量\"}"
```

安全原则：

- LLM 或规则生成器只生成候选 SQL
- 最终是否执行由程序校验决定
- 当前只允许单条 SELECT 查询
- 执行器只使用只读模式

## 统一答案生成

默认模式：

- 使用各业务节点返回的确定性答案草稿
- 对 KB、GraphRAG、Text2SQL 等 Evidence 答案追加 `source_id`
- Evidence 不足时拒绝基于猜测回答

OpenAI-compatible LLM 模式：

```powershell
$env:GUSTOBOT_ANSWER_LLM_BASE_URL="http://127.0.0.1:8001/v1"
$env:GUSTOBOT_ANSWER_LLM_API_KEY="your-api-key"
$env:GUSTOBOT_ANSWER_LLM_MODEL="your-chat-model"
```

答案 LLM 只消费统一 Evidence，不直接访问数据库、图谱或工具。
当 `GUSTOBOT_ALLOW_GENERAL_RECIPE_FALLBACK=true` 时，如果“菜名 + 做法”问题没有命中本地 Evidence，
答案层会明确标注“本地库暂未检索到可引用菜谱”，再给出通用做法参考；不会把这类兜底内容伪装成数据库来源。

## 图片链路

图片链路支持 OpenAI-compatible Vision LLM 和通用 OCR HTTP 服务；未配置真实服务时会回退到附件 `text` 和文件名推断：

```powershell
$env:GUSTOBOT_VISION_BASE_URL="http://127.0.0.1:8001/v1"
$env:GUSTOBOT_VISION_API_KEY="your-api-key"
$env:GUSTOBOT_VISION_MODEL="qwen3-vl-plus"
$env:GUSTOBOT_OCR_BASE_URL="http://127.0.0.1:8003/ocr"
```

Vision 模型需要支持图片理解和 `image_url` 输入；`qwen-image-*` 是图片生成/编辑模型，不适合这里使用。

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"这张图里的菜需要哪些食材\",\"attachments\":[{\"type\":\"image\",\"filename\":\"gongbao.jpg\"}]}"
```

流程：

```text
image attachment
-> image_understanding_node
-> structured text / reroute_text
-> router
-> KB RAG / GraphRAG / Text2SQL
```

## 文件入库

聊天接口中的文件附件会进入文件入库节点：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"请把这个文件入库\",\"attachments\":[{\"type\":\"file\",\"filename\":\"闽菜资料.txt\",\"text\":\"佛跳墙属于闽菜，常用于宴席场景。\"}]}"
```

也可以直接调用文件入库接口：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/files/ingest `
  -H "Content-Type: application/json" `
  -d "{\"files\":[{\"type\":\"file\",\"filename\":\"闽菜资料.txt\",\"text\":\"佛跳墙属于闽菜，常用于宴席场景。\"}]}"
```

## 数据准备读取过程

新旧项目保持独立：v2 不在运行时直接读取 `GustoBot-develop` 路径。旧项目中的资料应先复制到 v2 的 `data/raw/kb`，再由 v2 自己完成读取、切块、embedding 和入库。

当前已经把旧项目 `data/kb` 下的示例资料复制到了：

- `data/raw/kb/data.txt`
- `data/raw/kb/历史菜谱源头.xlsx`

数据准备链路：

```text
data/raw/kb 原始资料
-> scripts/prepare_kb_data.py
-> app.files.local_loader.LocalKnowledgeFileLoader
-> PreparedDocument(title/content/source_id/metadata)
-> KnowledgeBaseService.ingest_document
-> split_text_to_chunks
-> embedding
-> PostgreSQL + pgvector 或内存存储
```

先做 dry-run，只检查读取结果，不入库：

```powershell
python scripts/prepare_kb_data.py --dry-run
```

正式入库：

```powershell
python scripts/prepare_kb_data.py
```

如果配置了 `GUSTOBOT_POSTGRES_DSN`，会写入 PostgreSQL + pgvector；否则会写入当前进程的内存知识库，适合本地演示但不会持久化。

脚本默认只允许读取 `GustoBot-v2` 项目目录内的数据目录和 `.env`，这是为了避免新项目在运行时隐式依赖旧项目。

如果数据库服务运行在 Docker 中，需要区分两种连接地址：

- 宿主机直接运行 `python scripts/prepare_kb_data.py`：使用 Docker 暴露到本机的端口，例如 `postgresql://gustobot:gustobot@127.0.0.1:5432/gustobot`
- API 运行在 compose 容器内：使用 Docker 服务名，例如 `postgresql://gustobot:gustobot@postgres:5432/gustobot`

v2 默认推荐重新入库到新表结构：

```env
GUSTOBOT_KB_PGVECTOR_TABLE=kb_chunks
```

## 缓存与日志

默认缓存：

- 未配置 Redis 时使用内存缓存
- 只缓存无附件、无需反问、无副作用的稳定回答
- 文件入库和图片请求不会被简单热点缓存短路

Redis 模式：

```powershell
$env:GUSTOBOT_REDIS_URL="redis://127.0.0.1:6379/0"
```

日志追踪：

- 默认写入 `logs/traces.jsonl`
- 每条事件包含 `trace_id`
- 记录请求开始、缓存命中、请求结束等事件

## 评估

运行内置评估样本：

```powershell
& 'D:\anaconda3\envs\qwen_agent_env\python.exe' scripts\evaluate.py
```

当前脚本输出：

- Router Accuracy
- Guardrails Block Accuracy
- Answer Has Evidence Rate
- Evidence Source Coverage
- Citation Coverage
- Failure Reason Counts
- Route Breakdown
- E2E Latency P50 / P95

当前稳定基线：

- `reports/eval_after_food_data.json`：食品数据闭环正式评估，Router Accuracy、Evidence Source Coverage、Citation Coverage 和危险输入拦截率均为 100%。
- `reports/eval_after_food_data_first_run.json`：规则修复前的第一轮报告，用于对比修复收益。
- 本地 Router SFT 或云端 Router SFT 评估报告属于实验结果，应与稳定基线并排比较；如果低于基线，不应直接替换默认 Router 配置。

## Docker Compose

```powershell
Copy-Item .env.example .env
docker compose up --build
```

`.env` 只放本机配置和真实 API Key，不要提交；`.env.example` 保留可公开的占位配置。

包含服务：

- FastAPI
- PostgreSQL + pgvector
- Neo4j
- Redis
- Vue 3 + Vite Web 前端

数据库服务都运行在 Docker 中。当前 compose 的默认连接方式：

| 服务 | 容器内地址 | 宿主机地址 | 用途 |
| --- | --- | --- | --- |
| PostgreSQL + pgvector | `postgres:5432` | `127.0.0.1:5432` | KB RAG、Text2SQL、Schema Catalog、评估日志 |
| Neo4j | `neo4j:7687` | `127.0.0.1:17687` | GraphRAG 图数据库 |
| Redis | `redis:6379` | `127.0.0.1:6379` | 缓存 |
| Web | `web:5173` | `127.0.0.1:5173` | 本地聊天界面 |

PostgreSQL 会在容器首次初始化时执行：

- `docker/postgres/init.sql`

该脚本会创建 pgvector extension、KB 表、示例 `recipes` 业务表、`schema_catalog`、评估日志表和 trace 元数据表。

如果你在宿主机执行入库脚本，`.env` 里应使用宿主机地址：

```env
GUSTOBOT_POSTGRES_DSN=postgresql://gustobot:gustobot@127.0.0.1:5432/gustobot
GUSTOBOT_KB_PGVECTOR_TABLE=kb_chunks
```

如果服务运行在 `api` 容器内，`docker-compose.yml` 会自动覆盖为容器内地址：

```env
GUSTOBOT_POSTGRES_DSN=postgresql://gustobot:gustobot@postgres:5432/gustobot
```

推荐的 Docker-first 启动顺序：

```powershell
docker compose up -d postgres neo4j redis
docker compose run --rm api python scripts/prepare_kb_data.py
docker compose run --rm api python scripts/bootstrap_neo4j.py
docker compose up -d api
docker compose exec -T api python scripts/docker_smoke.py
```

接入真实 embedding、reranker 或答案 LLM 后，可以打开更严格的 smoke 断言：

```powershell
docker compose exec -T api python scripts/docker_smoke.py `
  --expect-embedding-provider openai-compatible `
  --expect-reranker-success `
  --expect-answer-mode llm
```

如果 PostgreSQL 容器已经初始化过，后来才新增或修改初始化 SQL，需要重建对应 volume 后才会重新执行初始化脚本。

宿主机手动导入 Neo4j 时，可以显式使用本机端口：

```powershell
python scripts/bootstrap_neo4j.py `
  --uri bolt://127.0.0.1:17687 `
  --username neo4j `
  --password gustobotneo4j
```

Docker smoke 会验证：

- `/api/v1/health`
- KB 使用 `postgres_pgvector`
- Text2SQL Evidence 的 `executor_type=postgres_readonly`
- GraphRAG Evidence 的 `store_type=neo4j`
- PostgreSQL 初始化表和种子数据
- Neo4j 节点/关系数量
- Redis ping 和 set/get
- 响应 trace_id 写入 `logs/traces.jsonl`

如果只想验证 HTTP 链路而不直连数据库，可运行：

```powershell
python scripts/docker_smoke.py --skip-db-checks
```

## 快速开始

```powershell
cd E:\大模型\项目实战\GustoBot-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

健康检查：

```powershell
curl http://127.0.0.1:8000/api/v1/health
```

聊天接口：

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"你好\"}"
```

## 真实模型与真实数据

v2 正式链路不再依赖 MySQL/Milvus/LightRAG。MySQL SQL 文件只作为旧数据迁移输入，
运行时统一使用 PostgreSQL/pgvector、Neo4j 和 Redis。

当前项目会优先复用 `.env` 中已有的真实模型变量：

- embedding：`EMBEDDING_*` 或 `KB_EMBEDDING_*`
- reranker：`KB_RERANK_*` 或 `RERANK_*`
- Router / Text2SQL / Answer LLM：`LLM_*` 或对应的 `GUSTOBOT_*_LLM_*`
- Vision：`VISION_*` 或 `GUSTOBOT_VISION_*`

Docker 不再强制把 KB embedding 覆盖成 hash；如果 `.env` 已配置
`text-embedding-v4`、`qwen3-rerank`、`qwen3-max`，API 容器会走真实服务。

真实数据导入顺序：

```powershell
docker compose up -d postgres neo4j redis
docker compose run --rm api python scripts/import_real_recipe_data.py --reset --ingest-kb
docker compose run --rm api python scripts/bootstrap_neo4j_from_postgres.py --reset
docker compose up -d --build api
docker compose exec -T api python scripts/docker_smoke.py `
  --expect-embedding-provider openai-compatible `
  --expect-reranker-success `
  --expect-answer-mode llm `
  --expect-real-data
```

如需单独重建 raw KB 向量，可执行：

```powershell
docker compose run --rm api python scripts/prepare_kb_data.py --reset
```

`--reset` 会清空当前 KB pgvector 数据后重新入库，适合切换 embedding 模型或维度后使用。

## P0 / DashScope strict 验收

P0 验收路径用于确认生产配置不会静默退回 hash embedding、内存存储、seed 图谱、SQLite、规则 SQL 或关键词 rerank。

```powershell
docker compose -f docker-compose.yml -f docker-compose.p0.yml up -d --build api
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/check_p0_dashscope_readiness.py
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/p0_failure_injection.py
docker compose -f docker-compose.yml -f docker-compose.p0.yml exec -T api python scripts/docker_smoke.py --expect-embedding-provider openai-compatible --expect-reranker-success --expect-answer-mode llm --expect-real-data
```

P0 override 默认使用 DashScope OpenAI-compatible Chat、`text-embedding-v4`、`qwen3-rerank` 和 `qwen3-vl-plus`。图片理解必须使用支持 `image_url` 输入的视觉理解模型，例如 `qwen3-vl-plus`、`qwen-vl-plus` 或 `qwen-vl-max`；`qwen-image-*` 属于图片生成/编辑模型，不能作为 Vision 理解模型。

## 测试

```powershell
pytest
```

本项目在当前机器上可使用已有解释器验证：

```powershell
& 'D:\anaconda3\envs\qwen_agent_env\python.exe' -m pytest
```

## 后续扩展

- 扩展食品数据底座规模，补充 Open Food Facts 稳定下载链路
- 继续打磨 Vision/OCR 提示词、失败重试和图片评估样本
- 完成 Router SFT 本地/云端部署后的长期对比评估，沉淀低置信度和失败样本回流机制
- 补齐旧 `.doc`、复杂扫描 PDF 等边界文件解析能力
- 引入 sqlglot 等 AST 解析器进一步增强 SQL 校验
- 增加 prompt injection、PII、数据外泄和 RAG 文件投毒检测等更完整的 Guardrails 策略
- 将 `/chat/stream` 从“最终答案切块”升级为真正的工作流事件流或模型 token streaming
- 增加 route confusion matrix、低置信度样本集、多意图样本和更完整的跨领域评估样本集
