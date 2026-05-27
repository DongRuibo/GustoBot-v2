#!/usr/bin/env bash
# ============================================================
# GustoBot Router 本地微调环境安装脚本
# 用法: bash scripts/sft/setup.sh
# 前提: 已安装 Python 3.10+ 和 CUDA 12.x
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo " GustoBot Router 本地微调环境安装"
echo "=========================================="
echo "项目目录: $PROJECT_ROOT"
echo ""

# ---------- 1. 检查 CUDA ----------
echo "[1/5] 检查 CUDA 环境..."
if command -v nvidia-smi &>/dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "警告: 未找到 nvidia-smi，请确认 CUDA 驱动已安装"
fi

# ---------- 2. 创建虚拟环境 ----------
VENV_DIR="$PROJECT_ROOT/.venv-sft"
echo "[2/5] 创建虚拟环境: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "  -> 虚拟环境已创建"
else
    echo "  -> 虚拟环境已存在，跳过"
fi
source "$VENV_DIR/bin/activate"
echo "  -> Python: $(python3 --version)"
echo "  -> Pip: $(pip --version)"
echo ""

# ---------- 3. 安装 PyTorch ----------
echo "[3/5] 安装 PyTorch (CUDA 12.4)..."
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' || echo "12.4")
echo "  检测到 CUDA 版本: $CUDA_VERSION"

if [[ "$CUDA_VERSION" == 12.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
elif [[ "$CUDA_VERSION" == 11.* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" 2>&1 | tail -1
python3 -c "import torch; print(f'  -> PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU 数量: {torch.cuda.device_count()}')"
echo ""

# ---------- 4. 安装 LLaMA-Factory ----------
echo "[4/5] 安装 LLaMA-Factory..."
pip install "llamafactory[torch]>=0.9.0" 2>&1 | tail -1
python3 -c "import llamafactory; print(f'  -> LLaMA-Factory 版本检查通过')" 2>/dev/null || echo "  -> llamafactory 已安装（版本检查跳过）"

# 安装 bitsandbytes (QLoRA 量化)
pip install "bitsandbytes>=0.45.0" 2>&1 | tail -1
echo ""

# ---------- 5. 下载模型 ----------
MODEL_DIR="$PROJECT_ROOT/models"
MODEL_NAME="Qwen3-8B"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

echo "[5/5] 检查模型: $MODEL_NAME"
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo "  -> 模型已存在: $MODEL_PATH"
else
    echo "  -> 下载模型 $MODEL_NAME ..."
    mkdir -p "$MODEL_DIR"
    # 优先用 huggingface-cli
    if command -v huggingface-cli &>/dev/null; then
        huggingface-cli download "Qwen/$MODEL_NAME" --local-dir "$MODEL_PATH"
    else
        pip install huggingface_hub
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/$MODEL_NAME', local_dir='$MODEL_PATH')
"
    fi
    echo "  -> 模型已下载到: $MODEL_PATH"
fi
echo ""

# ---------- 完成 ----------
echo "=========================================="
echo " 安装完成！"
echo "=========================================="
echo ""
echo "后续步骤:"
echo "  1. 激活虚拟环境: source $VENV_DIR/bin/activate"
echo "  2. 开始训练:     bash scripts/sft/train_qlora.sh"
echo "  3. 部署模型:     bash scripts/sft/deploy_vllm.sh"
echo ""
echo "如果模型下载慢，可以用镜像:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo "  然后重新运行本脚本"
