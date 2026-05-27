"""合并食品商品 CSV，并生成图谱节点边。"""

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
    build_graph,
    dedupe_products,
    read_products_csv,
    resolve_project_path,
    write_graph_csv,
    write_nutrients_csv,
    write_products_csv,
)


DEFAULT_INPUTS = [
    PROCESSED_DIR / "products_openfoodfacts.csv",
    PROCESSED_DIR / "products_usda.csv",
]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    product_paths = [resolve_project_path(path, label="products") for path in args.products]
    products = []
    for path in product_paths:
        if path.exists():
            products.extend(read_products_csv(path))
    if not products and args.sample:
        products = SAMPLE_PRODUCTS
    if not products:
        raise SystemExit("没有可用的产品 CSV。请先运行 prepare_openfoodfacts.py / prepare_usda.py，或加 --sample。")

    products = dedupe_products(products)
    products_output = resolve_project_path(args.products_output, label="products-output")
    nutrients_output = resolve_project_path(args.nutrients_output, label="nutrients-output")
    graph_nodes_output = resolve_project_path(args.graph_nodes_output, label="graph-nodes-output")
    graph_edges_output = resolve_project_path(args.graph_edges_output, label="graph-edges-output")
    stats_output = resolve_project_path(args.stats_output, label="stats-output") if args.stats_output else None

    product_count = write_products_csv(products_output, products)
    nutrient_count = write_nutrients_csv(nutrients_output, products)
    nodes, edges = build_graph(products)
    write_graph_csv(graph_nodes_output, graph_edges_output, nodes, edges)

    payload = {
        "product_count": product_count,
        "nutrient_row_count": nutrient_count,
        "graph_node_count": len(nodes),
        "graph_edge_count": len(edges),
        "products_output": str(products_output),
        "nutrients_output": str(nutrients_output),
        "graph_nodes_output": str(graph_nodes_output),
        "graph_edges_output": str(graph_edges_output),
        "status": "ok",
    }
    if stats_output:
        stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build food product graph CSV files.")
    parser.add_argument("--products", nargs="*", default=[str(path) for path in DEFAULT_INPUTS])
    parser.add_argument("--products-output", default=str(PROCESSED_DIR / "products.csv"))
    parser.add_argument("--nutrients-output", default=str(PROCESSED_DIR / "nutrients.csv"))
    parser.add_argument("--graph-nodes-output", default=str(PROCESSED_DIR / "graph_nodes.csv"))
    parser.add_argument("--graph-edges-output", default=str(PROCESSED_DIR / "graph_edges.csv"))
    parser.add_argument("--stats-output", default=str(PROCESSED_DIR / "food_dataset_stats.json"))
    parser.add_argument("--sample", action="store_true", help="没有输入 CSV 时使用内置小样本。")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

