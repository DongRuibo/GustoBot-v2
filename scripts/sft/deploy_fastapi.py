#!/usr/bin/env python3
"""轻量级 OpenAI-compatible Router 模型服务。

用 transformers + FastAPI 直接推理，不依赖 vLLM。这个脚本主要用于把本地
Router LoRA/合并模型部署成 `/v1/chat/completions`，供 GustoBot-v2 远程调用。
"""
import argparse
import time
import os
import torch
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- 参数解析 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--merged", action="store_true", help="使用已合并的模型")
parser.add_argument("--model-dir", default=None, help="显式指定模型目录，优先级高于 --merged")
parser.add_argument("--model-name", default="gustobot-router")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8100)
parser.add_argument("--max-len", type=int, default=512)
args = parser.parse_args()

# ---------- 加载模型 ----------
from transformers import AutoModelForCausalLM, AutoTokenizer

if args.model_dir:
    model_path = args.model_dir
    print("模式: 显式模型目录")
elif args.merged:
    model_path = os.path.join(PROJECT_ROOT, "models", "gustobot-router-merged")
    print("模式: 合并模型")
else:
    model_path = os.path.join(PROJECT_ROOT, "models", "Qwen3-8B")
    print("模式: 基座模型（无 LoRA，仅供测试）")

print(f"模型: {model_path}")
print(f"加载中...")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"模型加载完成，设备: {model.device}")

# ---------- API 定义 ----------
app = FastAPI(title="GustoBot Router API")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gustobot-router"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

@app.get("/health")
async def health():
    return {"status": "ok", "model": args.model_name, "model_path": model_path}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": args.model_name, "object": "model", "owned_by": "local"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Router 是结构化分类任务，Qwen3 推理时关闭 thinking，减少无关解释文本。
    messages = [m.model_dump() for m in request.messages]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    temperature = float(request.temperature or 0.0)
    max_tokens = min(int(request.max_tokens or args.max_len), args.max_len)

    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = 0.95
        outputs = model.generate(**inputs, **generation_kwargs)

    # 只取生成部分
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    prompt_tokens = inputs["input_ids"].shape[-1]
    completion_tokens = len(generated_ids)

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=request.model or args.model_name,
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=response_text),
            finish_reason="stop",
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

if __name__ == "__main__":
    print(f"\n启动 API 服务: http://{args.host}:{args.port}")
    print(f"模型名: {args.model_name}")
    print(f"健康检查: http://{args.host}:{args.port}/health")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print()
    uvicorn.run(app, host=args.host, port=args.port)
