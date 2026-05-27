#!/usr/bin/env bash
# ============================================================
# vLLM 部署脚本 - 启动 OpenAI 兼容 API 服务
# 用法: bash scripts/sft/deploy_vllm.sh [--merged]
#   默认: 加载 LoRA 适配器 (基座 + LoRA)
#   --merged: 加载已合并的完整模型
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo " vLLM 部署 GustoBot Router"
echo "=========================================="

# 激活虚拟环境
VENV_DIR="$PROJECT_ROOT/.venv-sft"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_ROOT"

# ---------- 参数解析 ----------
USE_MERGED=false
for arg in "$@"; do
    case $arg in
        --merged) USE_MERGED=true ;;
    esac
done

# ---------- 配置 ----------
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8100}"
TENSOR_PARALLEL_SIZE="${VLLM_TP:-1}"
MAX_MODEL_LEN="${VLLM_MAX_LEN:-512}"
GPU_MEM_UTIL="${VLLM_GPU_MEM:-0.90}"

if [ "$USE_MERGED" = true ]; then
    # 使用已合并的模型
    MODEL_PATH="$PROJECT_ROOT/models/gustobot-router-merged"
    if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "错误: 合并模型不存在: $MODEL_PATH"
        echo "请先运行: bash scripts/sft/merge_lora.sh"
        exit 1
    fi
    echo "模式: 合并模型"
    echo "模型: $MODEL_PATH"
    EXTRA_ARGS=""
else
    # 使用基座 + LoRA 适配器
    BASE_MODEL="$PROJECT_ROOT/models/Qwen3-8B"
    LORA_DIR="$PROJECT_ROOT/output/gustobot_router_qlora"
    LATEST_CKPT=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)

    if [ ! -d "$BASE_MODEL" ] || [ ! -f "$BASE_MODEL/config.json" ]; then
        echo "错误: 基座模型不存在: $BASE_MODEL"
        exit 1
    fi
    if [ -z "$LATEST_CKPT" ]; then
        echo "错误: 未找到 LoRA checkpoint: $LORA_DIR/checkpoint-*"
        exit 1
    fi

    MODEL_PATH="$BASE_MODEL"
    echo "模式: 基座 + LoRA"
    echo "基座: $MODEL_PATH"
    echo "LoRA: $LATEST_CKPT"
    EXTRA_ARGS="--enable-lora --lora-modules gustobot-router=$LATEST_CKPT --max-lora-rank 16"
fi

echo ""
echo "配置:"
echo "  地址: $HOST:$PORT"
echo "  张量并行: $TENSOR_PARALLEL_SIZE"
echo "  最大序列长度: $MAX_MODEL_LEN"
echo "  GPU 显存利用: $GPU_MEM_UTIL"
echo ""
echo "=========================================="
echo " 启动 vLLM 服务..."
echo "=========================================="

# ---------- 启动 vLLM ----------
# 如果使用 LoRA，模型名用 gustobot-router（LoRA 别名）
# 如果使用合并模型，模型名用 gustobot-router
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name gustobot-router \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype bfloat16 \
    --trust-remote-code \
    $EXTRA_ARGS
