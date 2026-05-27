# GustoBot Router 本地微调指南

在本地 GPU 服务器上用 LLaMA-Factory + QLoRA 微调 Qwen3-8B 作为 GustoBot Router。

## 环境要求

| 项目 | 要求 |
|------|------|
| GPU | RTX 3090 x1 (24GB) 及以上 |
| Python | 3.10+ |
| CUDA | 12.x |
| 磁盘 | ~20GB（模型 + checkpoint） |

## 快速开始

### 1. 安装环境

```bash
bash scripts/sft/setup.sh
```

这会：
- 创建虚拟环境 `.venv-sft`
- 安装 PyTorch + LLaMA-Factory + bitsandbytes
- 下载 Qwen3-8B 模型到 `models/Qwen3-8B/`

> 如果模型下载慢，用镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 2. 训练

```bash
bash scripts/sft/train_qlora.sh
```

- QLoRA 4-bit 量化，单卡约 14GB 显存
- 800 条样本，约 10-30 分钟
- 输出到 `output/gustobot_router_qlora/`

### 3. 合并 LoRA 权重

```bash
bash scripts/sft/merge_lora.sh
```

将 LoRA 适配器合并进基座模型，输出到 `models/gustobot-router-merged/`。

### 4. 部署

```bash
bash scripts/sft/deploy_vllm.sh --merged
```

启动 vLLM 服务，兼容 OpenAI API：
- 地址：`http://localhost:8100/v1`
- 模型名：`gustobot-router`

### 5. 接入 GustoBot

修改 `.env`：

```env
GUSTOBOT_ROUTER_LLM_ENABLED=true
GUSTOBOT_ROUTER_LLM_PROVIDER=openai-compatible
GUSTOBOT_ROUTER_LLM_BASE_URL=http://localhost:8100/v1
GUSTOBOT_ROUTER_LLM_API_KEY=not-needed
GUSTOBOT_ROUTER_LLM_MODEL=gustobot-router
```

### 6. 评估

```bash
docker compose exec -T api python scripts/evaluate.py \
    --samples data/eval/food_eval.jsonl \
    --output reports/eval_local_sft.json
```

## 文件说明

| 文件 | 用途 |
|------|------|
| `setup.sh` | 环境安装 |
| `train_qlora.sh` | QLoRA 训练 |
| `train_qlora.yaml` | LLaMA-Factory 训练配置 |
| `merge_lora.sh` | LoRA 权重合并 |
| `deploy_vllm.sh` | vLLM 部署 |

## 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-8B | 8B 参数 |
| 微调方法 | QLoRA | 4-bit 量化 + LoRA |
| LoRA rank | 16 | 适配器维度 |
| LoRA alpha | 32 | 缩放系数 |
| 学习率 | 2e-4 | |
| Batch size | 4 x 4 = 16 | 有效 batch |
| Epochs | 3 | |
| 最大长度 | 512 | 足够覆盖 Router 输入 |

## 故障排查

**CUDA OOM：** 减小 `per_device_train_batch_size`（改 2 或 1）

**训练 loss 不下降：** 检查数据格式，确认 `dataset_info.json` 路径正确

**vLLM 启动失败：** 确认端口未被占用，检查 GPU 显存

**模型下载失败：** 设置 `HF_ENDPOINT=https://hf-mirror.com`
