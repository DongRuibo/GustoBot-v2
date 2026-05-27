"""正式食品评估集生成测试。"""

import json
from pathlib import Path

from scripts.data.food_dataset import FoodProduct, write_graph_csv, write_jsonl, write_products_csv
from scripts.data.generate_food_eval import DEFAULT_TOTALS, load_graph, main as generate_food_eval_main


def test_generate_food_eval_writes_fixed_distribution_without_sft(tmp_path: Path) -> None:
    products_path = tmp_path / "food_products.csv"
    nodes_path = tmp_path / "graph_nodes.csv"
    edges_path = tmp_path / "graph_edges.csv"
    kb_path = tmp_path / "kb_documents.jsonl"
    output_path = tmp_path / "eval" / "food_eval.jsonl"
    report_path = tmp_path / "reports" / "eval_food_generation.json"

    products = [
        FoodProduct(
            product_id="usda:1",
            name="Sample Milk",
            brand="Demo Dairy",
            category="Dairy and Egg Products",
            country="United States",
            ingredients_text="milk, vitamin d",
            energy_kcal=60,
            protein=3.2,
            fat=1.0,
            carbohydrates=5.0,
            sugars=5.0,
            salt=0.1,
            source="usda_foundation",
        ),
        FoodProduct(
            product_id="usda:2",
            name="Sample Beans",
            brand="Demo Foods",
            category="Legumes and Legume Products",
            country="United States",
            ingredients_text="beans, salt",
            energy_kcal=120,
            protein=7.0,
            fat=0.5,
            carbohydrates=20.0,
            sugars=1.0,
            salt=0.2,
            source="usda_sr_legacy",
        ),
    ]
    write_products_csv(products_path, products)
    write_graph_csv(nodes_path, edges_path, _graph_nodes(), _graph_edges())
    write_jsonl(
        kb_path,
        [
            {
                "source_id": "food:nutrient_protein",
                "title": "蛋白质作用说明",
                "content": "蛋白质是重要营养素。",
                "metadata": {"source": "food_dataset", "doc_type": "nutrient"},
            },
            {
                "source_id": "food:foodon_overview",
                "title": "FoodOn 类别和同义词说明",
                "content": "FoodOn 提供食品类别和同义词。",
                "metadata": {"source": "food_dataset", "doc_type": "foodon"},
            },
        ],
    )

    generate_food_eval_main(
        [
            "--products",
            str(products_path),
            "--nodes",
            str(nodes_path),
            "--edges",
            str(edges_path),
            "--kb-documents",
            str(kb_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
            "--seed",
            "20260525",
        ]
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    route_counts = {}
    for row in rows:
        route_counts[row["expected_route"]] = route_counts.get(row["expected_route"], 0) + 1

    assert len(rows) == 230
    assert route_counts == DEFAULT_TOTALS
    assert report["sample_count"] == 230
    assert report["sft_generated"] is False
    assert report["guardrail_sample_count"] == 10
    assert all("expected_evidence_sources" in row for row in rows)
    assert any(row["should_block"] for row in rows)
    assert not (tmp_path / "sft").exists()


def test_generate_food_eval_is_deterministic_for_same_seed(tmp_path: Path) -> None:
    products_path = tmp_path / "food_products.csv"
    nodes_path = tmp_path / "graph_nodes.csv"
    edges_path = tmp_path / "graph_edges.csv"
    kb_path = tmp_path / "kb_documents.jsonl"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    write_products_csv(
        products_path,
        [
            FoodProduct(
                product_id="usda:1",
                name="Sample Milk",
                brand="Demo Dairy",
                category="Dairy and Egg Products",
                ingredients_text="milk",
                energy_kcal=60,
                protein=3.2,
                sugars=5.0,
                source="usda_foundation",
            )
        ],
    )
    write_graph_csv(nodes_path, edges_path, _graph_nodes(), _graph_edges())
    write_jsonl(
        kb_path,
        [
            {
                "source_id": "food:nutrient_protein",
                "title": "蛋白质作用说明",
                "content": "蛋白质是重要营养素。",
                "metadata": {"source": "food_dataset"},
            }
        ],
    )

    base_args = [
        "--products",
        str(products_path),
        "--nodes",
        str(nodes_path),
        "--edges",
        str(edges_path),
        "--kb-documents",
        str(kb_path),
        "--seed",
        "7",
    ]
    generate_food_eval_main([*base_args, "--output", str(first_output), "--report", str(tmp_path / "first.json")])
    generate_food_eval_main([*base_args, "--output", str(second_output), "--report", str(tmp_path / "second.json")])

    assert first_output.read_text(encoding="utf-8") == second_output.read_text(encoding="utf-8")


def test_load_graph_groups_food_relations(tmp_path: Path) -> None:
    nodes_path = tmp_path / "graph_nodes.csv"
    edges_path = tmp_path / "graph_edges.csv"
    write_graph_csv(nodes_path, edges_path, _graph_nodes(), _graph_edges())

    graph = load_graph(nodes_path, edges_path)

    assert len(graph.product_nodes) == 2
    assert len(graph.products_with_category) == 2
    assert len(graph.products_with_nutrients) == 2
    assert len(graph.products_with_ingredients) == 2


def _graph_nodes() -> list[dict[str, str]]:
    return [
        _node("product:usda_1", "Product", "Sample Milk", ["Sample Milk"], {"product_id": "usda:1"}),
        _node("product:usda_2", "Product", "Sample Beans", ["Sample Beans"], {"product_id": "usda:2"}),
        _node("food_category:dairy", "FoodCategory", "Dairy and Egg Products", ["Dairy"]),
        _node("food_category:legumes", "FoodCategory", "Legumes and Legume Products", ["Legumes"]),
        _node("ingredient:milk", "Ingredient", "milk", ["milk"]),
        _node("ingredient:beans", "Ingredient", "beans", ["beans"]),
        _node("nutrient:protein", "Nutrient", "protein", ["protein"]),
    ]


def _graph_edges() -> list[dict[str, str]]:
    return [
        _edge("edge:1:category", "product:usda_1", "food_category:dairy", "BELONGS_TO"),
        _edge("edge:2:category", "product:usda_2", "food_category:legumes", "BELONGS_TO"),
        _edge("edge:1:ingredient", "product:usda_1", "ingredient:milk", "HAS_INGREDIENT"),
        _edge("edge:2:ingredient", "product:usda_2", "ingredient:beans", "HAS_INGREDIENT"),
        _edge("edge:1:protein", "product:usda_1", "nutrient:protein", "HAS_NUTRIENT", {"value": 3.2, "unit": "g"}),
        _edge("edge:2:protein", "product:usda_2", "nutrient:protein", "HAS_NUTRIENT", {"value": 7.0, "unit": "g"}),
    ]


def _node(node_id: str, label: str, name: str, aliases: list[str], properties: dict | None = None) -> dict[str, str]:
    return {
        "node_id": node_id,
        "label": label,
        "name": name,
        "aliases": json.dumps(aliases, ensure_ascii=False),
        "properties": json.dumps(properties or {}, ensure_ascii=False),
    }


def _edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    properties: dict | None = None,
) -> dict[str, str]:
    return {
        "edge_id": edge_id,
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "properties": json.dumps(properties or {}, ensure_ascii=False),
    }
