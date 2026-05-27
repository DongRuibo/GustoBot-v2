#!/usr/bin/env bash
# ============================================================
# 部署 GustoBot Router API 服务
# 用法: bash scripts/sft/deploy.sh [--merged]
#   默认: 加载 LoRA 适配器 (基座 + LoRA)
#   --merged: 加载已合并的完整模型
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo " 部署 GustoBot Router"
echo "=========================================="

# 激活虚拟环境
VENV_DIR="$PROJECT_ROOT/.venv-sft"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_ROOT"

# ---------- 参数解析 ----------
MERGED_FLAG=""
for arg in "$@"; do
    case $arg in
        --merged) MERGED_FLAG="--merged" ;;
    esac
done

HOST="${GUSTOBOT_ROUTER_HOST:-0.0.0.0}"
PORT="${GUSTOBOT_ROUTER_PORT:-8100}"

echo ""
echo "配置:"
echo "  地址: $HOST:$PORT"
echo "  模式: $([ -n "$MERGED_FLAG" ] && echo '合并模型' || echo '基座+LoRA')"
echo ""
echo "=========================================="
echo " 启动 API 服务..."
echo "=========================================="

python3 "$SCRIPT_DIR/deploy_fastapi.py" --host "$HOST" --port "$PORT" $MERGED_FLAG
