"""清洗 Open Food Facts 数据，输出统一食品商品 CSV。"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLE_PRODUCTS,
    dedupe_products,
    product_from_openfoodfacts,
    resolve_project_path,
    write_nutrients_csv,
    write_products_csv,
)


DEFAULT_INPUT = RAW_DIR / "openfoodfacts" / "openfoodfacts-products.jsonl.gz"
DEFAULT_PRODUCTS_OUTPUT = PROCESSED_DIR / "products_openfoodfacts.csv"
DEFAULT_NUTRIENTS_OUTPUT = PROCESSED_DIR / "nutrients_openfoodfacts.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = resolve_project_path(args.output, label="output")
    nutrients_output = resolve_project_path(args.nutrients_output, label="nutrients-output")

    if args.sample:
        products = [product for product in SAMPLE_PRODUCTS if product.source == "sample"]
    else:
        input_path = resolve_project_path(args.input, label="input")
        if not input_path.exists():
            raise SystemExit(f"找不到 Open Food Facts 原始文件：{input_path}。可先运行 download_openfoodfacts.py，或加 --sample 做链路自检。")
        products = _load_openfoodfacts_products(input_path, limit=args.limit)

    products = dedupe_products(products)
    product_count = write_products_csv(output, products)
    nutrient_count = write_nutrients_csv(nutrients_output, products)
    print(
        json.dumps(
            {
                "products_output": str(output),
                "nutrients_output": str(nutrients_output),
                "product_count": product_count,
                "nutrient_row_count": nutrient_count,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Open Food Facts subset for GustoBot-v2.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_PRODUCTS_OUTPUT))
    parser.add_argument("--nutrients-output", default=str(DEFAULT_NUTRIENTS_OUTPUT))
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--sample", action="store_true", help="使用内置小样本生成输出，便于无网络自检。")
    return parser.parse_args(argv)


def _load_openfoodfacts_products(path: Path, *, limit: int) -> list:
    products = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if limit > 0 and len(products) >= limit:
                break
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            product = product_from_openfoodfacts(raw)
            if product is not None:
                products.append(product)
    return products


if __name__ == "__main__":
    raise SystemExit(main())

