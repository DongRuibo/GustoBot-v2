"""查询 DashScope 部署状态。"""
import json
import os
import sys

import httpx


def main():
    deployment_id = sys.argv[1] if len(sys.argv) > 1 else "qwen3-8b-59d22cc35f09"
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("GUSTOBOT_ROUTER_LLM_API_KEY")
    if not api_key:
        print("ERROR: 请设置 DASHSCOPE_API_KEY 或 GUSTOBOT_ROUTER_LLM_API_KEY", file=sys.stderr)
        sys.exit(1)

    url = f"https://dashscope.aliyuncs.com/api/v1/deployments/{deployment_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = httpx.get(url, headers=headers, timeout=60)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
