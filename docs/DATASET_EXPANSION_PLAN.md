# 食品数据底座扩展计划

## 目标

第一版只做小而完整的闭环，不追全量：`下载 -> 清洗 -> PostgreSQL -> Neo4j -> KB -> 评估 -> SFT 样本生成`。

建议规模：

- Open Food Facts：10000-20000 条商品
- USDA FoodData：5000-10000 条标准食物/品牌食物
- KB 文档：1000-3000 条食品说明文本
- 图谱节点：5000-20000
- 图谱边：20000-100000
- 磁盘：100MB-1GB，不含图片

## 数据源

- USDA Foundation Foods 04/2026：基础食品营养数据，第一版全量使用可用记录。
- USDA SR Legacy 04/2018：传统标准食品营养数据，第一版默认最多 8000 条。
- USDA Branded Foods 04/2026：品牌商品食品数据，从 zip 内流式抽样，默认 8000 条，不整体解压。
- FoodOn 同义词表：提供食品类别、别名和上位词关系；第一版不解析 OWL，只记录 OWL 到 manifest。

Open Food Facts 本轮未进入正式 processed 数据：官方 API/Hugging Face rows 接口不稳定，第一版用 USDA Branded Foods 补足商品食品数据。

## 清洗规则

保留规则：

- `name` 不为空
- `category` 不为空
- `ingredients_text` 或营养字段至少一个不为空
- 优先可解释的中文、英文文本

丢弃规则：

- 名称或分类严重缺失
- 营养值为负数或异常极大值
- 同名、同品牌、同分类的重复记录
- 过长的异常名称或分类

## 输出文件

```text
data/processed/food_products.csv
data/processed/food_nutrients.csv
data/processed/products.csv
data/processed/nutrients.csv
data/processed/kb_documents.jsonl
data/processed/graph_nodes.csv
data/processed/graph_edges.csv
data/eval/food_eval.jsonl
reports/eval_food_generation.json
reports/eval_after_food_data_first_run.json
reports/eval_after_food_data.json
reports/sft_food_generation.json
data/sft/router_train.jsonl
data/sft/router_dev.jsonl
data/sft/router_test.jsonl
data/sft/answer_train.jsonl
data/sft/answer_dev.jsonl
data/sft/answer_test.jsonl
```

`data/raw/external/` 已在 `.gitignore` 中忽略，避免把原始 zip/CSV/OWL/TSV 大文件提交进 Git。

## 执行顺序

无网络自检：

```powershell
python scripts\data\prepare_openfoodfacts.py --sample
python scripts\data\prepare_usda.py --sample
python scripts\data\build_graph_edges.py --sample
python scripts\data\build_kb_docs.py
python scripts\data\generate_food_eval_sft.py
```

真实数据：

```powershell
python scripts\data\prepare_food_dataset.py --raw-root data\raw\external --sr-limit 8000 --branded-limit 8000 --graph-product-limit 12000 --max-kb-products 2000
python scripts\data\generate_food_eval.py --products data\processed\food_products.csv --nodes data\processed\graph_nodes.csv --edges data\processed\graph_edges.csv --kb-documents data\processed\kb_documents.jsonl --output data\eval\food_eval.jsonl --report reports\eval_food_generation.json --seed 20260525
```

导入 PostgreSQL / KB / Neo4j：

```powershell
docker compose exec -T api python scripts\data\import_food_dataset.py --products data\processed\food_products.csv --nutrients data\processed\food_nutrients.csv --kb-documents data\processed\kb_documents.jsonl --reset --ingest-kb
docker compose exec -T api python scripts\data\bootstrap_food_neo4j.py --nodes data\processed\graph_nodes.csv --edges data\processed\graph_edges.csv --reset
docker compose exec -T -e GUSTOBOT_ROUTER_LLM_ENABLED=false -e GUSTOBOT_ANSWER_LLM_BASE_URL= -e GUSTOBOT_TEXT2SQL_LLM_BASE_URL= -e GUSTOBOT_GRAPH_PLANNER_LLM_ENABLED=false -e GUSTOBOT_KB_EMBEDDING_PROVIDER=hash -e GUSTOBOT_KB_RERANK_BASE_URL= -e GUSTOBOT_CACHE_ENABLED=false -e GUSTOBOT_TRACE_ENABLED=false api python scripts\evaluate.py --samples data\eval\food_eval.jsonl --output reports\eval_after_food_data.json --max-workers 4
Copy-Item reports\eval_after_food_data.json data\eval\eval_after_food_data.json -Force
docker compose exec -T -e GUSTOBOT_ROUTER_LLM_ENABLED=false -e GUSTOBOT_ANSWER_LLM_BASE_URL= -e GUSTOBOT_TEXT2SQL_LLM_BASE_URL= -e GUSTOBOT_GRAPH_PLANNER_LLM_ENABLED=false -e GUSTOBOT_KB_EMBEDDING_PROVIDER=hash -e GUSTOBOT_KB_RERANK_BASE_URL= -e GUSTOBOT_CACHE_ENABLED=false -e GUSTOBOT_TRACE_ENABLED=false api python scripts\data\generate_food_sft.py --eval-samples data\eval\food_eval.jsonl --eval-report data\eval\eval_after_food_data.json --report data\sft\sft_food_generation.json --router-count 800 --answer-count 300 --answer-candidate-limit 180 --max-workers 4 --seed 20260526
```

## 三条链路映射

PostgreSQL：

- `food_products`：商品、品牌、分类、配料、过敏原、营养标签
- `food_nutrients`：营养素长表
- `schema_catalog`：暴露给 Text2SQL 的白名单 schema

Neo4j：

- `(:Product)-[:HAS_INGREDIENT]->(:Ingredient)`
- `(:Product)-[:HAS_ALLERGEN]->(:Allergen)`
- `(:Product)-[:BELONGS_TO]->(:FoodCategory)`
- `(:Product)-[:HAS_NUTRIENT]->(:Nutrient)`
- `(:Ingredient)-[:IS_A]->(:IngredientCategory)`

KB：

- 食品分类说明
- 营养素解释
- 过敏原解释
- 商品说明

## 验证指标

- Router Accuracy >= 85%
- Evidence Source Coverage >= 80%
- Citation Coverage >= 90%
- 危险 SQL / 删除类问题拦截率 = 100%
- KB / GraphRAG / Text2SQL 三条链路都能返回 Evidence

## 当前正式结果

截至 2026-05-25，正式 processed 数据和评估闭环已经跑通：

- 食品产品：15522 条
- 营养记录：215321 条
- 图谱节点：15777 个
- 图谱边：74355 条
- KB 文档：2406 条
- 评估集：230 条

最终报告 `reports/eval_after_food_data.json`：

- Router Accuracy：100%
- Evidence Source Coverage：100%
- Citation Coverage：100%
- 危险输入拦截率：100%
- KB / GraphRAG / Text2SQL / Multi / Clarify / General 均返回预期 Evidence

第一轮未达标报告已保留为 `reports/eval_after_food_data_first_run.json`，用于追踪路由规则修复前后的差异。

SFT 已完成正式生成：

- Router SFT：800 条，train/dev/test = 640/80/80
- Answer SFT：300 条，train/dev/test = 240/30/30
- Answer 样本全部来自当前 workflow 返回的可引用 Evidence
- SFT 生成报告：`reports/sft_food_generation.json`

Router SFT 训练接入：

- 已增加 DashScope 专用导出格式：`data/sft/router_training/router_dashscope_train.jsonl` / `dev` / `test`
- 已增加 DashScope 微调管理脚本：`scripts/data/dashscope_router_finetune.py`
- 默认只 dry-run，真实上传、创建训练和部署必须显式传 `--submit`
- 训练部署后可通过 `GUSTOBOT_ROUTER_LLM_PROVIDER=dashscope` 接回 Router，并用 `reports/eval_router_dashscope_sft.json` 对比食品数据基线
