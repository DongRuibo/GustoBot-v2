#!/usr/bin/env bash
# ============================================================
# 合并 LoRA 权重到基座模型
# 用法: bash scripts/sft/merge_lora.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo " 合并 LoRA 权重"
echo "=========================================="

# 激活虚拟环境
VENV_DIR="$PROJECT_ROOT/.venv-sft"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_ROOT"

LORA_DIR="$PROJECT_ROOT/output/gustobot_router_qlora"
MERGED_DIR="$PROJECT_ROOT/models/gustobot-router-merged"

# 检查 LoRA 输出
if [ ! -d "$LORA_DIR" ]; then
    echo "错误: LoRA 输出目录不存在: $LORA_DIR"
    echo "请先完成训练: bash scripts/sft/train_qlora.sh"
    exit 1
fi

# 找到最新的 checkpoint
LATEST_CKPT=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "错误: 未找到 checkpoint 目录"
    exit 1
fi
echo "使用 checkpoint: $LATEST_CKPT"

# 检查基座模型
MODEL_PATH="$PROJECT_ROOT/models/Qwen3-8B"
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "错误: 基座模型不存在: $MODEL_PATH"
    exit 1
fi
echo "基座模型: $MODEL_PATH"
echo "输出路径: $MERGED_DIR"
echo ""

# 合并
mkdir -p "$MERGED_DIR"

python3 "$SCRIPT_DIR/merge_lora.py"

echo ""
echo "=========================================="
echo " 合并完成！"
echo "=========================================="
echo "合并后模型: $MERGED_DIR"
echo ""
echo "下一步: bash scripts/sft/deploy_vllm.sh"
