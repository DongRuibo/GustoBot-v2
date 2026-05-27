"""数据准备读取测试模块。

这个文件验证 v2 自己的本地数据读取过程是否可靠：资料先进入项目内目录，
再被解析成 PreparedDocument，后续才交给 KBService 做切块、embedding 和入库。
"""

from copy import deepcopy
from pathlib import Path

from app.files.local_loader import LocalKnowledgeFileLoader
from scripts.bootstrap_neo4j_from_postgres import _taxonomy_stats
from scripts.import_real_recipe_data import _parse_insert


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "kb_loader"


def test_local_loader_reads_text_and_csv() -> None:
    # README 是目录说明，不应该作为业务知识入库；txt 和 csv 行才应该变成 PreparedDocument。
    data_dir = FIXTURE_DIR / "mixed"
    documents = LocalKnowledgeFileLoader(data_dir).load()

    assert len(documents) == 3
    assert {document.metadata["parser_type"] for document in documents} == {"plain_text", "csv_row"}
    assert any(document.source_id == "local:闽菜资料.txt" for document in documents)
    assert any("菜名: 宫保鸡丁" in document.content for document in documents)


def test_local_loader_reads_json_documents() -> None:
    # JSON 可以直接携带 content，也可以携带任意字段；任意字段会被扁平化为可 embedding 的文本。
    data_dir = FIXTURE_DIR / "json"
    documents = LocalKnowledgeFileLoader(data_dir).load()

    assert len(documents) == 2
    assert documents[0].content == "辣椒在明清之后逐渐进入中国饮食。"
    assert "菜名: 西湖醋鱼" in documents[1].content
    assert documents[1].metadata["item_index"] == 2


def test_real_recipe_sql_parser_handles_variables_and_json_array() -> None:
    # 旧 MySQL SQL 中包含 @变量，导入脚本需要在不运行 MySQL 的情况下解析它们。
    statement = """
    INSERT INTO recipes (
        name, description, image_url, video_url, total_time, servings, difficulty,
        cuisine_id, total_calories, total_protein, total_carbs, total_fat
    ) VALUES (
        '宫保鸡丁', '经典川菜', 'image.jpg', 'video.mp4',
        35, 2, 'medium', @cuisine_sichuan,
        520.00, 48.00, 26.00, 28.00
    )
    """

    table, rows = _parse_insert(statement, {"cuisine_sichuan": 7})

    assert table == "recipes"
    assert rows[0]["name"] == "宫保鸡丁"
    assert rows[0]["cuisine_id"] == 7
    assert rows[0]["total_calories"] == 520.00


def test_real_recipe_sql_parser_handles_step_tools_json_array() -> None:
    # 旧 MySQL SQL 还包含 JSON_ARRAY，导入脚本应转成 Python list 后写入 PostgreSQL jsonb。
    statement = """
    INSERT INTO recipe_steps (
        recipe_id, step_number, action, instruction, duration, temperature, tools_used, tips
    ) VALUES (
        @recipe_gongbao, 1, '切配', '鸡肉切丁',
        10, '常温', JSON_ARRAY('砧板', '菜刀'), NULL
    )
    """

    table, rows = _parse_insert(statement, {"recipe_gongbao": 3})

    assert table == "recipe_steps"
    assert rows[0]["recipe_id"] == 3
    assert rows[0]["tools_used"] == ["砧板", "菜刀"]
    assert rows[0]["tips"] is None


def test_neo4j_taxonomy_stats_reports_uncategorized_without_mutating_graph() -> None:
    graph = {
        "nodes": [
            {"node_id": "ingredient:ribs", "label": "Ingredient", "name": "猪排骨"},
            {"node_id": "ingredient:salt", "label": "Ingredient", "name": "盐"},
            {"node_id": "ingredient_category:pork", "label": "IngredientCategory", "name": "猪肉"},
        ],
        "edges": [
            {
                "edge_id": "edge:ingredient:ribs:category:pork",
                "source_id": "ingredient:ribs",
                "target_id": "ingredient_category:pork",
                "relation": "BELONGS_TO_CATEGORY",
                "properties": {},
            }
        ],
    }
    original = deepcopy(graph)

    stats = _taxonomy_stats(graph, report_uncategorized=True)

    assert stats["ingredient_count"] == 2
    assert stats["categorized_ingredient_count"] == 1
    assert stats["uncategorized_ingredient_count"] == 1
    assert stats["ingredient_category_edge_count"] == 1
    assert stats["category_coverage"] == {"猪肉": 1}
    assert stats["uncategorized_ingredients"] == ["盐"]
    assert stats["uncategorized_ingredients_top"] == ["盐"]
    assert graph == original
