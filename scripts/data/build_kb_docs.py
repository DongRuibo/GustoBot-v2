"""根据食品商品 CSV 生成 KB 文档 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    PROCESSED_DIR,
    SAMPLE_PRODUCTS,
    build_kb_documents,
    read_products_csv,
    resolve_project_path,
    write_jsonl,
)


DEFAULT_PRODUCTS = PROCESSED_DIR / "products.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "kb_documents.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    products_path = resolve_project_path(args.products, label="products")
    output = resolve_project_path(args.output, label="output")
    if products_path.exists():
        products = read_products_csv(products_path)
    elif args.sample:
        products = SAMPLE_PRODUCTS
    else:
        raise SystemExit(f"找不到产品 CSV：{products_path}。请先运行 build_graph_edges.py，或加 --sample。")

    documents = build_kb_documents(products, max_product_docs=args.max_product_docs)
    count = write_jsonl(output, documents)
    print(
        json.dumps(
            {
                "products": str(products_path),
                "output": str(output),
                "document_count": count,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build food KB documents JSONL.")
    parser.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-product-docs", type=int, default=3000)
    parser.add_argument("--sample", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

