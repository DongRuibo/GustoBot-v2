#!/usr/bin/env python3
"""
合并 LoRA 权重到基座模型（纯 Python 实现，绕过 llamafactory-cli 的参数解析问题）
用法: python scripts/sft/merge_lora.py
"""
import sys
import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LORA_DIR = os.path.join(PROJECT_ROOT, "output", "gustobot_router_qlora")
MERGED_DIR = os.path.join(PROJECT_ROOT, "models", "gustobot-router-merged")

# 找最新 checkpoint
checkpoints = sorted(
    [d for d in os.listdir(LORA_DIR) if d.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[1])
)
if not checkpoints:
    print("错误: 未找到 checkpoint 目录")
    sys.exit(1)

LATEST_CKPT = os.path.join(LORA_DIR, checkpoints[-1])
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen3-8B")

print("=" * 50)
print(" 合并 LoRA 权重")
print("=" * 50)
print(f"基座模型: {MODEL_PATH}")
print(f"LoRA:     {LATEST_CKPT}")
print(f"输出:     {MERGED_DIR}")
print()

# 加载基座模型
print("加载基座模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)

# 加载 LoRA
print("加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, LATEST_CKPT)

# 合并
print("合并权重...")
model = model.merge_and_unload()

# 保存
print("保存合并模型...")
os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)

# 保存 tokenizer
print("保存 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_DIR)

print()
print("=" * 50)
print(" 合并完成！")
print("=" * 50)
print(f"合并后模型: {MERGED_DIR}")
print()
print("下一步: bash scripts/sft/deploy_vllm.sh --merged")
