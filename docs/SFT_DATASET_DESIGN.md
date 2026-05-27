# SFT 数据集设计

## 目标

食品数据底座的 SFT 只服务两个可控任务：

- Router SFT：把用户问题路由到 `general / kb / graphrag / text2sql / clarify / multi`
- Answer SFT：给定 Evidence 后生成带来源、不编造的答案

当前状态：SFT 已基于正式 USDA + FoodOn 数据底座生成。脚本先校验 `reports/eval_after_food_data.json` 的指标，再读取 `data/eval/food_eval.jsonl` 扩展 Router SFT，并通过当前 workflow 重新采集真实 Evidence 生成 Answer SFT。旧的 `generate_food_eval_sft.py` 只能视为样例，不作为本轮正式训练数据。

第一版规模：

- Router SFT：500-1000 条，默认生成 800 条
- Answer SFT：200-500 条，默认生成 300 条
- 划分比例：train/dev/test = 80/10/10

## Router SFT 格式

每行 JSONL：

```json
{
  "messages": [
    {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
    {"role": "user", "content": "{\"question\":\"统计糖分最高的前 10 个产品\",\"input_features\":{}}"}
  ],
  "output": {
    "route_type": "text2sql",
    "confidence": 0.86,
    "reason": "食品数据底座 Router SFT 样本。",
    "slots": {"analysis_intent": true},
    "need_clarification": false
  }
}
```

覆盖问题类型：

- `general`：问候、能力介绍
- `kb`：营养素作用、过敏原概念、食品分类解释
- `graphrag`：产品-配料、产品-过敏原、产品-分类关系
- `text2sql`：品牌数量、糖分排名、蛋白质排名、分类统计
- `clarify`：低信息问题和危险 SQL
- `multi`：统计 + 解释、图谱关系 + 概念解释

## Answer SFT 格式

每行 JSONL：

```json
{
  "instruction": "根据给定 Evidence 回答，必须引用 source_id，不要编造 Evidence 之外的信息。",
  "input": {
    "question": "Peanut Protein Bar 有哪些过敏原？",
    "evidence": [
      {
        "source_type": "kb",
        "source_id": "food_product:sample:peanut_bar",
        "content": "Peanut Protein Bar；品牌：Demo Foods；分类：Protein bars；过敏原：peanuts, soy。"
      }
    ]
  },
  "output": "Peanut Protein Bar 标注的过敏原是：peanuts, soy。来源：food_product:sample:peanut_bar"
}
```

约束：

- 只能使用 Evidence 中出现的信息
- 必须保留 `source_id`
- 不根据常识补充食品功效、健康建议或医学判断
- 过敏原回答必须明确来源于标签/图谱证据

## 生成脚本

正式脚本：

Docker 默认只挂载 `data/` 和 `logs/`，因此如果使用容器内脚本，可先把最终 eval 报告同步一份到 `data/eval/eval_after_food_data.json`。

```powershell
Copy-Item reports\eval_after_food_data.json data\eval\eval_after_food_data.json -Force

docker compose exec -T `
  -e GUSTOBOT_ROUTER_LLM_ENABLED=false `
  -e GUSTOBOT_ANSWER_LLM_BASE_URL= `
  -e GUSTOBOT_TEXT2SQL_LLM_BASE_URL= `
  -e GUSTOBOT_GRAPH_PLANNER_LLM_ENABLED=false `
  -e GUSTOBOT_KB_EMBEDDING_PROVIDER=hash `
  -e GUSTOBOT_KB_RERANK_BASE_URL= `
  -e GUSTOBOT_CACHE_ENABLED=false `
  -e GUSTOBOT_TRACE_ENABLED=false `
  api python scripts/data/generate_food_sft.py `
  --eval-samples data/eval/food_eval.jsonl `
  --eval-report data/eval/eval_after_food_data.json `
  --report data/sft/sft_food_generation.json `
  --router-count 800 `
  --answer-count 300 `
  --answer-candidate-limit 180 `
  --max-workers 4 `
  --seed 20260526
```

正式执行前应先确认：

- `reports/eval_after_food_data.json` 已达标
- 失败样本已经归因并修复
- Router SFT 的标签分布来自正式 eval
- Answer SFT 的 Evidence 来自当前 PostgreSQL / Neo4j / KB 检索结果

输出：

```text
data/sft/router_train.jsonl
data/sft/router_dev.jsonl
data/sft/router_test.jsonl
data/sft/answer_train.jsonl
data/sft/answer_dev.jsonl
data/sft/answer_test.jsonl
```

本轮实际结果：

- Router SFT：800 条，train/dev/test = 640/80/80
- Answer SFT：300 条，train/dev/test = 240/30/30
- Answer Evidence 来源：graph / kb / sql
- 生成报告：`reports/sft_food_generation.json`

## 质量检查

生成后至少检查：

- Router 标签分布是否覆盖 6 类路由
- `clarify` 中危险 SQL 样本是否保留
- Answer 样本是否全部含 `source_id`
- Answer 输出是否没有 Evidence 外的营养/健康断言

## Router 训练格式导出

Router SFT 训练前先导出为训练框架常见格式：

```powershell
python scripts\data\export_router_sft_training.py `
  --router-train data\sft\router_train.jsonl `
  --router-dev data\sft\router_dev.jsonl `
  --router-test data\sft\router_test.jsonl `
  --output-dir data\sft\router_training
```

输出目录：

```text
data/sft/router_training/
```

包含：

- `router_openai_train.jsonl` / `dev` / `test`：OpenAI messages 格式
- `router_swift_train.jsonl` / `dev` / `test`：ms-swift messages 格式
- `router_dashscope_train.jsonl` / `dev` / `test`：阿里云百炼 DashScope fine-tune 格式，每行只保留 `messages`
- `router_llamafactory_train.json` / `dev` / `test`：LLaMA-Factory Alpaca 格式
- `dataset_info.json`：LLaMA-Factory 数据集配置
- `manifest.json`：导出统计、标签分布、各格式文件大小

## DashScope 云端 Router SFT

当前正式训练路径优先使用阿里云百炼 API 云端 SFT，只训练 Router，不训练 Answer。

本地先做格式校验和 dry-run，不产生费用：

```powershell
python scripts\data\dashscope_router_finetune.py validate
python scripts\data\dashscope_router_finetune.py upload --dry-run
python scripts\data\dashscope_router_finetune.py create --dry-run
```

确认账号、余额、地域和模型配额后，再显式提交：

```powershell
$env:DASHSCOPE_API_KEY="your-dashscope-api-key"
python scripts\data\dashscope_router_finetune.py upload --submit
python scripts\data\dashscope_router_finetune.py create --submit
python scripts\data\dashscope_router_finetune.py status --submit
python scripts\data\dashscope_router_finetune.py logs --submit
python scripts\data\dashscope_router_finetune.py deploy --submit
```

状态和报告文件：

```text
data/sft/router_training/dashscope_finetune_state.json
reports/dashscope_router_finetune.json
```

这两个文件只记录 `file_id / job_id / finetuned_output / deployment_id / status` 等非敏感信息，不保存 API Key。

部署成功后，Router 使用 DashScope 原生 endpoint：

```env
GUSTOBOT_ROUTER_LLM_PROVIDER=dashscope
GUSTOBOT_ROUTER_LLM_BASE_URL=https://dashscope.aliyuncs.com/api/v1
GUSTOBOT_ROUTER_LLM_API_KEY=your-dashscope-api-key
GUSTOBOT_ROUTER_LLM_MODEL=your-deployment-model-id
```

然后对比正式食品评估：

```powershell
docker compose exec -T api python scripts\evaluate.py `
  --samples data\eval\food_eval.jsonl `
  --output reports\eval_router_dashscope_sft.json `
  --max-workers 4
```

对比基线：

```text
reports/eval_after_food_data.json
reports/eval_router_dashscope_sft.json
```

本机训练条件检查：

- PyTorch：已安装，CUDA 可用
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，显存 8GB
- transformers：已安装
- LLaMA-Factory：未安装
- ms-swift：未安装
- peft：未安装
- 本地 HuggingFace 缓存中暂未发现 Qwen 训练模型

因此当前已经完成“可训练数据准备”，真正开 LoRA 训练前还需要安装训练框架并准备基础模型。

推荐第一版 Router LoRA：

```powershell
# 方案 A：LLaMA-Factory
llamafactory-cli train `
  --stage sft `
  --do_train `
  --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct `
  --dataset gustobot_router_train `
  --eval_dataset gustobot_router_dev `
  --dataset_dir data\sft\router_training `
  --template qwen `
  --finetuning_type lora `
  --output_dir outputs\router-qwen2.5-1.5b-lora `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 2e-4 `
  --num_train_epochs 3 `
  --fp16
```

```powershell
# 方案 B：ms-swift
swift sft `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --train_type lora `
  --dataset data\sft\router_training\router_swift_train.jsonl `
  --val_dataset data\sft\router_training\router_swift_dev.jsonl `
  --output_dir outputs\router-qwen2.5-1.5b-lora `
  --num_train_epochs 3 `
  --per_device_train_batch_size 1 `
  --gradient_accumulation_steps 8 `
  --learning_rate 2e-4
```
