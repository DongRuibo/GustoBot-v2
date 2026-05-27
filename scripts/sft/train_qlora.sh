#!/usr/bin/env bash
# ============================================================
# GustoBot Router QLoRA 训练脚本
# 用法: bash scripts/sft/train_qlora.sh
# 前提: 已运行 setup.sh 安装环境
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo " GustoBot Router QLoRA 训练"
echo "=========================================="

# ---------- 激活虚拟环境 ----------
VENV_DIR="$PROJECT_ROOT/.venv-sft"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "已激活虚拟环境: $VENV_DIR"
else
    echo "警告: 虚拟环境不存在，请先运行 setup.sh"
fi

cd "$PROJECT_ROOT"

# ---------- 检查 GPU ----------
echo ""
echo "GPU 状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""

# ---------- 检查训练数据 ----------
TRAIN_DATA="data/sft/router_training/router_llamafactory_train.json"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误: 训练数据不存在: $TRAIN_DATA"
    echo "请先运行数据导出: python scripts/data/export_router_sft_training.py"
    exit 1
fi
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$TRAIN_DATA'))))")
echo "训练样本数: $SAMPLE_COUNT"
echo ""

# ---------- 检查模型 ----------
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-8B"
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo "使用本地模型: $MODEL_PATH"
    # 临时修改 yaml 中的模型路径
    CONFIG_FILE="$SCRIPT_DIR/train_qlora.yaml"
    sed "s|model_name_or_path: Qwen/Qwen3-8B|model_name_or_path: $MODEL_PATH|" "$CONFIG_FILE" > /tmp/train_qlora_local.yaml
    CONFIG_FILE="/tmp/train_qlora_local.yaml"
else
    echo "使用 HuggingFace 模型: Qwen/Qwen3-8B"
    CONFIG_FILE="$SCRIPT_DIR/train_qlora.yaml"
fi

# ---------- 开始训练 ----------
echo "=========================================="
echo " 开始 QLoRA 训练"
echo "  模型: Qwen3-8B"
echo "  方法: LoRA (rank=16, alpha=32)"
echo "  量化: 4-bit BNB"
echo "  预计显存: ~14GB/卡"
echo "  预计时间: 10-30 分钟 (视数据量)"
echo "=========================================="
echo ""

START_TIME=$(date +%s)

llamafactory-cli train "$CONFIG_FILE"

END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "=========================================="
echo " 训练完成！耗时: ${DURATION} 分钟"
echo "=========================================="
echo ""
echo "输出目录: output/gustobot_router_qlora/"
echo ""
echo "下一步:"
echo "  1. 合并 LoRA 权重:  bash scripts/sft/merge_lora.sh"
echo "  2. 部署模型:        bash scripts/sft/deploy_vllm.sh"
