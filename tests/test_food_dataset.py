"""食品数据底座单元测试。"""

import csv
import json
import zipfile
from types import SimpleNamespace

from app.core import router as router_module
from app.core.router import route_question
from app.graphrag.models import GraphEdge, GraphNode
from app.graphrag.service import GraphRAGService
from app.graphrag.store import InMemoryGraphStore
from app.text2sql.generator import RuleBasedSQLGenerator
from app.text2sql.schema import ColumnSchema, SchemaMatch, TableSchema
from scripts.data.food_dataset import SAMPLE_PRODUCTS, build_graph, build_kb_documents, product_from_openfoodfacts
from scripts.data.prepare_food_dataset import main as prepare_food_dataset_main


def test_openfoodfacts_product_cleaning_keeps_required_fields() -> None:
    raw = {
        "code": "123",
        "product_name": "Peanut Protein Bar",
        "brands": "Demo Foods",
        "categories": "Snacks, Protein bars",
        "ingredients_text": "peanuts, soy protein, sugar",
        "allergens_tags": ["en:peanuts", "en:soybeans"],
        "countries": "United States",
        "nutriments": {"energy-kcal_100g": 420, "proteins_100g": 28, "sugars_100g": 18},
    }

    product = product_from_openfoodfacts(raw)

    assert product is not None
    assert product.product_id == "off:123"
    assert product.name == "Peanut Protein Bar"
    assert product.protein == 28
    assert "peanuts" in product.allergens


def test_food_graph_and_kb_documents_are_generated() -> None:
    nodes, edges = build_graph(SAMPLE_PRODUCTS)
    documents = build_kb_documents(SAMPLE_PRODUCTS, max_product_docs=2)

    assert any(node["label"] == "Product" for node in nodes)
    assert any(edge["relation"] == "HAS_ALLERGEN" for edge in edges)
    assert any(document["metadata"]["doc_type"] == "allergen" for document in documents)
    assert any("Peanut Protein Bar" in document["content"] for document in documents)


def test_graphrag_allergen_to_products_food_template() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("product:bar", "Product", "Peanut Protein Bar", ("Peanut Protein Bar",)))
    store.add_node(GraphNode("allergen:peanut", "Allergen", "花生", ("peanuts", "花生")))
    store.add_edge(GraphEdge("edge:bar:peanut", "product:bar", "allergen:peanut", "HAS_ALLERGEN"))

    result = GraphRAGService(store=store, max_depth=2).query("哪些产品含有花生过敏原？")

    assert "Peanut Protein Bar" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "allergen_to_products"
    assert metadata["template_id"] == "allergen_to_products_v1"


def test_graphrag_product_category_template() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("product:bar", "Product", "Peanut Protein Bar", ("Peanut Protein Bar",)))
    store.add_node(GraphNode("food_category:bars", "FoodCategory", "Protein bars", ("Protein bars",)))
    store.add_edge(GraphEdge("edge:bar:category", "product:bar", "food_category:bars", "BELONGS_TO"))

    result = GraphRAGService(store=store, max_depth=2).query("Peanut Protein Bar 属于什么类别？")

    assert "Protein bars" in result.answer
    assert result.raw_evidence[0]["metadata"]["graph_intent"] == "product_category"


def test_graphrag_product_detail_template_handles_category_and_nutrition_query() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("product:bar", "Product", "Peanut Protein Bar", ("Peanut Protein Bar",)))
    store.add_node(GraphNode("food_category:bars", "FoodCategory", "Protein bars", ("Protein bars",)))
    store.add_node(GraphNode("nutrient:protein", "Nutrient", "protein", ("protein",)))
    store.add_edge(GraphEdge("edge:bar:category", "product:bar", "food_category:bars", "BELONGS_TO"))
    store.add_edge(
        GraphEdge(
            "edge:bar:protein",
            "product:bar",
            "nutrient:protein",
            "HAS_NUTRIENT",
            {"value": 28, "unit": "g"},
        )
    )

    result = GraphRAGService(store=store, max_depth=2).query("Peanut Protein Bar 的分类和营养素是什么？")

    assert "Protein bars" in result.answer
    assert "protein=28g" in result.answer
    assert result.raw_evidence[0]["metadata"]["graph_intent"] == "product_detail"


def test_rule_sql_generator_food_product_templates() -> None:
    table = TableSchema(
        name="food_products",
        comment="食品商品主表",
        business_meaning="用于食品商品统计和营养排序。",
        module="food_analytics",
        updated_at="2026-05-25",
        columns=[
            ColumnSchema("name", "text", "商品名称"),
            ColumnSchema("brand", "text", "品牌"),
            ColumnSchema("sugars", "numeric", "糖分"),
            ColumnSchema("protein", "numeric", "蛋白质"),
        ],
    )
    matches = [SchemaMatch(table=table, score=1.0)]
    generator = RuleBasedSQLGenerator()

    sugar_sql = generator.generate("统计糖分最高的前 10 个产品", matches)
    brand_sql = generator.generate("Demo Foods 品牌有多少种产品？", matches)

    assert "FROM food_products" in sugar_sql.sql
    assert "ORDER BY sugars DESC" in sugar_sql.sql
    assert "COUNT(*) AS product_count" in brand_sql.sql
    assert "ILIKE '%Demo Foods%'" in brand_sql.sql


def test_router_food_questions_choose_expected_routes(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "settings",
        SimpleNamespace(
            router_llm_enabled=False,
            route_confidence_threshold=0.6,
        ),
    )

    graph_decision = route_question("哪些产品含有花生过敏原？", {})
    sql_decision = route_question("统计糖分最高的前 10 个产品", {})
    kb_decision = route_question("蛋白质有什么作用？", {})

    assert graph_decision.route_type.value == "graphrag"
    assert sql_decision.route_type.value == "text2sql"
    assert kb_decision.route_type.value == "kb"


def test_prepare_food_dataset_reads_usda_foodon_and_writes_manifest(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "external"
    output_dir = tmp_path / "processed"
    foundation_dir = raw_root / "usda" / "foundation_food_csv_2026-04-30" / "FoodData_Central_foundation"
    sr_dir = raw_root / "usda" / "sr_legacy_food_csv_2018-04" / "FoodData_Central_sr"
    foodon_dir = raw_root / "foodon"
    foundation_dir.mkdir(parents=True)
    sr_dir.mkdir(parents=True)
    foodon_dir.mkdir(parents=True)

    _write_usda_dir(
        foundation_dir,
        id_table="foundation_food.csv",
        fdc_id="1001",
        data_type="foundation_food",
        name="Foundation Milk",
        category_id="1",
    )
    _write_usda_dir(
        sr_dir,
        id_table="sr_legacy_food.csv",
        fdc_id="2001",
        data_type="sr_legacy_food",
        name="Legacy Beans",
        category_id="2",
    )
    _write_branded_zip(raw_root / "usda" / "FoodData_Central_branded_food_csv_2026-04-30.zip")
    (foodon_dir / "foodon.owl").write_text("<rdf:RDF />\n", encoding="utf-8")
    (foodon_dir / "foodon-synonyms.tsv").write_text(
        "\t".join(["?class", "?parent", "?type", "?label"])
        + "\n"
        + "<http://purl.obolibrary.org/obo/FOODON_0000001>\t<http://purl.obolibrary.org/obo/FOODON_0000000>\t\t\n"
        + '<http://purl.obolibrary.org/obo/FOODON_0000001>\t\t"label"\t"milk product"@en\n'
        + '<http://purl.obolibrary.org/obo/FOODON_0000001>\t\t"synonym (exact)"\t"dairy product"@en\n'
        + '<http://purl.obolibrary.org/obo/FOODON_0000002>\t<http://purl.obolibrary.org/obo/FOODON_0000001>\t\t\n'
        + '<http://purl.obolibrary.org/obo/FOODON_0000002>\t\t"label"\t"cheddar cheese"@en\n',
        encoding="utf-8",
    )

    prepare_food_dataset_main(
        [
            "--raw-root",
            str(raw_root),
            "--output-dir",
            str(output_dir),
            "--sr-limit",
            "1",
            "--branded-limit",
            "1",
            "--graph-product-limit",
            "10",
            "--max-kb-products",
            "10",
            "--max-foodon-terms",
            "10",
            "--max-foodon-docs",
            "10",
        ]
    )

    manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    with (output_dir / "food_products.csv").open("r", encoding="utf-8", newline="") as file:
        products = list(csv.DictReader(file))
    with (output_dir / "food_nutrients.csv").open("r", encoding="utf-8", newline="") as file:
        nutrients = list(csv.DictReader(file))
    with (output_dir / "graph_nodes.csv").open("r", encoding="utf-8", newline="") as file:
        nodes = list(csv.DictReader(file))

    assert manifest["product_count"] == 3
    assert manifest["source_counts"] == {"usda_branded": 1, "usda_foundation": 1, "usda_sr_legacy": 1}
    assert manifest["foodon"]["term_count"] == 2
    assert len(products) == 3
    assert any(row["nutrient_name"] == "sodium" for row in nutrients)
    assert any(row["node_id"].startswith("foodon_category:") for row in nodes)
    assert (output_dir / "products.csv").exists()
    assert (output_dir / "nutrients.csv").exists()


def _write_usda_dir(path, *, id_table: str, fdc_id: str, data_type: str, name: str, category_id: str) -> None:
    path.joinpath(id_table).write_text(f'"fdc_id","NDB_number","footnote"\n"{fdc_id}","1",""\n', encoding="utf-8")
    path.joinpath("food.csv").write_text(
        '"fdc_id","data_type","description","food_category_id","publication_date"\n'
        f'"{fdc_id}","{data_type}","{name}","{category_id}","2026-04-30"\n',
        encoding="utf-8",
    )
    path.joinpath("food_category.csv").write_text(
        '"id","code","description"\n'
        f'"{category_id}","0100","Sample Category {category_id}"\n',
        encoding="utf-8",
    )
    path.joinpath("nutrient.csv").write_text(_nutrient_csv(), encoding="utf-8")
    path.joinpath("food_nutrient.csv").write_text(_food_nutrient_csv(fdc_id), encoding="utf-8")


def _write_branded_zip(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        prefix = "FoodData_Central_branded_food_csv_2026-04-30/"
        archive.writestr(
            prefix + "branded_food.csv",
            '"fdc_id","brand_owner","brand_name","ingredients","serving_size","serving_size_unit","branded_food_category","market_country"\n'
            '"3001","Demo Brand","","beans, salt","30","g","Snack bars","United States"\n',
        )
        archive.writestr(
            prefix + "food.csv",
            '"fdc_id","data_type","description","food_category_id","publication_date","market_country","trade_channel","microbe_data"\n'
            '"3001","branded_food","Demo Bean Bar","","2026-04-30","United States","","[]"\n',
        )
        archive.writestr(prefix + "nutrient.csv", _nutrient_csv())
        archive.writestr(prefix + "food_nutrient.csv", _food_nutrient_csv("3001"))


def _nutrient_csv() -> str:
    return (
        '"id","name","unit_name","nutrient_nbr","rank"\n'
        '"1008","Energy","KCAL","208","300"\n'
        '"1003","Protein","G","203","600"\n'
        '"1004","Total lipid (fat)","G","204","800"\n'
        '"1005","Carbohydrate, by difference","G","205","1100"\n'
        '"2000","Sugars, total including NLEA","G","269","1500"\n'
        '"1093","Sodium, Na","MG","307","5800"\n'
    )


def _food_nutrient_csv(fdc_id: str) -> str:
    return (
        '"id","fdc_id","nutrient_id","amount","data_points","derivation_id","min","max","median","footnote","min_year_acquired"\n'
        f'"1","{fdc_id}","1008","120","","","","","","",""\n'
        f'"2","{fdc_id}","1003","5","","","","","","",""\n'
        f'"3","{fdc_id}","1004","3","","","","","","",""\n'
        f'"4","{fdc_id}","1005","20","","","","","","",""\n'
        f'"5","{fdc_id}","2000","4","","","","","","",""\n'
        f'"6","{fdc_id}","1093","200","","","","","","",""\n'
    )
