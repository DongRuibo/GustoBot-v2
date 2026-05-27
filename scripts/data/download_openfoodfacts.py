"""下载 Open Food Facts 原始 JSONL dump。

默认 URL 来自 Open Food Facts 官方复用数据说明；下载文件较大，建议只在需要刷新原始数据时运行。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import RAW_DIR, ensure_parent, resolve_project_path  # noqa: E402


DEFAULT_URL = "https://static.openfoodfacts.org/data/openfoodfacts-products.jsonl.gz"
DEFAULT_OUTPUT = RAW_DIR / "openfoodfacts" / "openfoodfacts-products.jsonl.gz"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = resolve_project_path(args.output, label="output")
    if output.exists() and not args.force:
        raise SystemExit(f"目标文件已存在：{output}。如需覆盖请加 --force。")

    ensure_parent(output)
    request = urllib.request.Request(args.url, headers={"User-Agent": "GustoBot-v2 food data builder"})
    bytes_written = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        with output.open("wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                bytes_written += len(chunk)

    print(
        json.dumps(
            {
                "url": args.url,
                "output": str(output),
                "size_bytes": bytes_written,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Open Food Facts JSONL dump.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

