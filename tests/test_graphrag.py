"""GraphRAG 单元测试模块。

这个文件验证第三阶段图谱问答能力本身是否可用，包括实体链接、关系子图抽取、
菜谱到食材查询、食材反查菜谱，以及 Graph Evidence 输出。
"""

from app.graphrag.models import GraphEdge, GraphNode
from app.graphrag.ingredient_taxonomy import (
    BELONGS_TO_CATEGORY,
    categories_for_ingredient,
    category_hierarchy_edges,
    category_node_data,
)
from app.graphrag.service import GraphRAGService, get_graphrag_service
from app.graphrag.store import InMemoryGraphStore


def _add_ingredient_taxonomy(store: InMemoryGraphStore) -> None:
    for category in category_node_data():
        node_id = str(category["node_id"])
        if node_id not in store.nodes:
            store.add_node(
                GraphNode(
                    node_id,
                    str(category["label"]),
                    str(category["name"]),
                    tuple(str(alias) for alias in category["aliases"]),
                    dict(category["properties"]),
                )
            )
    for edge in category_hierarchy_edges():
        store.add_edge(
            GraphEdge(
                str(edge["edge_id"]),
                str(edge["source_id"]),
                str(edge["target_id"]),
                str(edge["relation"]),
                dict(edge["properties"]),
            )
        )
    for node in list(store.nodes.values()):
        if node.label != "Ingredient":
            continue
        for category in categories_for_ingredient(node.name, node.properties.get("category")):
            store.add_edge(
                GraphEdge(
                    f"edge:{node.node_id}:category:{category.slug}",
                    node.node_id,
                    category.node_id,
                    BELONGS_TO_CATEGORY,
                    {"source": "taxonomy"},
                )
            )


def test_graphrag_recipe_ingredients() -> None:
    # 菜谱问食材是 GraphRAG 的典型正向关系查询：Recipe -> USES_INGREDIENT -> Ingredient。
    result = get_graphrag_service().query("宫保鸡丁需要哪些食材")

    assert "鸡肉" in result.answer
    assert "花生" in result.answer
    assert result.raw_evidence
    assert result.raw_evidence[0]["source_type"].value == "graph"
    assert result.raw_evidence[0]["metadata"]["edge_count"] > 0


def test_graphrag_ingredient_to_recipes() -> None:
    # 食材问能做什么菜是反向关系查询：Recipe -> USES_INGREDIENT -> Ingredient，再从 Ingredient 找回 Recipe。
    result = get_graphrag_service().query("白菜能做什么菜")

    assert "白菜炖豆腐" in result.answer
    assert result.linked_entities[0].node.name == "白菜"


def test_graphrag_general_pork_links_to_specific_ingredients() -> None:
    # “猪肉”是上位食材词，图谱里常保存的是猪排骨、猪里脊、五花肉等具体食材。
    store = InMemoryGraphStore()
    store.add_node(GraphNode("recipe:ribs", "Recipe", "红烧排骨", ("红烧排骨",)))
    store.add_node(GraphNode("recipe:loin", "Recipe", "酱香猪里脊", ("酱香猪里脊",)))
    store.add_node(GraphNode("recipe:pork_belly", "Recipe", "红烧肉", ("红烧肉",)))
    store.add_node(GraphNode("ingredient:ribs", "Ingredient", "猪排骨", ("排骨",)))
    store.add_node(GraphNode("ingredient:loin", "Ingredient", "猪肉里脊", ("猪里脊",)))
    store.add_node(GraphNode("ingredient:pork_belly", "Ingredient", "五花肉", ()))
    store.add_edge(GraphEdge("edge:ribs:pork", "recipe:ribs", "ingredient:ribs", "USES_INGREDIENT"))
    store.add_edge(GraphEdge("edge:loin:pork", "recipe:loin", "ingredient:loin", "USES_INGREDIENT"))
    store.add_edge(GraphEdge("edge:pork_belly:pork", "recipe:pork_belly", "ingredient:pork_belly", "USES_INGREDIENT"))
    _add_ingredient_taxonomy(store)

    result = GraphRAGService(store=store, max_depth=2).query("猪肉可以做什么菜")

    assert "泛化到了这些具体食材：猪排骨、猪肉里脊、五花肉" in result.answer
    assert "红烧排骨" in result.answer
    assert "酱香猪里脊" in result.answer
    assert "红烧肉" in result.answer
    assert result.linked_entities[0].node.label == "IngredientCategory"
    assert result.linked_entities[0].matched_text == "猪肉"
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "ingredient_to_recipes"
    assert metadata["template_id"] == "ingredient_to_recipes_v1"


def test_graphrag_unknown_howto_does_not_treat_single_char_meat_as_category() -> None:
    store = InMemoryGraphStore()
    _add_ingredient_taxonomy(store)

    result = GraphRAGService(store=store, max_depth=2).query("红烧肉怎么做")

    assert "没有在图谱中链接到明确的菜谱或食材实体" in result.answer
    assert result.linked_entities == []
    assert result.raw_evidence == []


def test_graphrag_lean_meat_category_links_to_specific_ingredients() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("recipe:loin", "Recipe", "青椒肉丝", ("青椒肉丝",)))
    store.add_node(GraphNode("recipe:tenderloin", "Recipe", "糖醋里脊", ("糖醋里脊",)))
    store.add_node(GraphNode("ingredient:loin", "Ingredient", "猪肉里脊", ("猪里脊",)))
    store.add_node(GraphNode("ingredient:tenderloin", "Ingredient", "通脊", ()))
    store.add_edge(GraphEdge("edge:loin:pork", "recipe:loin", "ingredient:loin", "USES_INGREDIENT"))
    store.add_edge(GraphEdge("edge:tenderloin:pork", "recipe:tenderloin", "ingredient:tenderloin", "USES_INGREDIENT"))
    _add_ingredient_taxonomy(store)

    result = GraphRAGService(store=store, max_depth=2).query("瘦肉可以做什么菜")

    assert "泛化到了这些具体食材：猪肉里脊、通脊" in result.answer
    assert "青椒肉丝" in result.answer
    assert "糖醋里脊" in result.answer
    assert result.linked_entities[0].node.label == "IngredientCategory"


def test_graphrag_leafy_vegetable_alias_links_to_specific_ingredients() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("recipe:cabbage", "Recipe", "白菜炖豆腐", ("白菜炖豆腐",)))
    store.add_node(GraphNode("recipe:spinach", "Recipe", "菠菜炒蛋", ("菠菜炒蛋",)))
    store.add_node(GraphNode("ingredient:cabbage", "Ingredient", "白菜", ("大白菜",)))
    store.add_node(GraphNode("ingredient:spinach", "Ingredient", "菠菜", ()))
    store.add_edge(GraphEdge("edge:cabbage:ingredient", "recipe:cabbage", "ingredient:cabbage", "USES_INGREDIENT"))
    store.add_edge(GraphEdge("edge:spinach:ingredient", "recipe:spinach", "ingredient:spinach", "USES_INGREDIENT"))
    _add_ingredient_taxonomy(store)

    result = GraphRAGService(store=store, max_depth=2).query("青菜可以做什么菜")

    assert "将“青菜”泛化到了这些具体食材：白菜、菠菜" in result.answer
    assert "白菜炖豆腐" in result.answer
    assert "菠菜炒蛋" in result.answer


def test_graphrag_specific_ingredient_still_prefers_exact_match() -> None:
    store = InMemoryGraphStore()
    store.add_node(GraphNode("recipe:cabbage", "Recipe", "白菜炖豆腐", ("白菜炖豆腐",)))
    store.add_node(GraphNode("ingredient:cabbage", "Ingredient", "白菜", ("大白菜",)))
    store.add_edge(GraphEdge("edge:cabbage:ingredient", "recipe:cabbage", "ingredient:cabbage", "USES_INGREDIENT"))
    _add_ingredient_taxonomy(store)

    result = GraphRAGService(store=store, max_depth=2).query("白菜能做什么菜")

    assert "白菜 在当前图谱中可以关联到这些菜谱：白菜炖豆腐" in result.answer
    assert "泛化到了这些具体食材" not in result.answer
    assert result.linked_entities[0].node.label == "Ingredient"


def test_graphrag_recipe_howto_includes_step_instruction_and_quantities() -> None:
    # 真实菜谱图谱里，步骤 instruction 和食材用量必须进入 Evidence，答案层才能生成详细做法。
    store = InMemoryGraphStore()
    store.add_node(
        GraphNode(
            "recipe:ribs",
            "Recipe",
            "红烧排骨",
            ("红烧排骨",),
            {"description": "家常风味红烧排骨", "total_time": 50, "servings": 2, "difficulty": "medium"},
        )
    )
    store.add_node(GraphNode("ingredient:ribs", "Ingredient", "猪排骨", ("排骨",)))
    store.add_node(GraphNode("ingredient:ginger", "Ingredient", "姜", ("姜片",)))
    store.add_node(
        GraphNode(
            "step:ribs:1",
            "Step",
            "第1步 焯水",
            properties={
                "order": 1,
                "action": "焯水",
                "instruction": "排骨冷水下锅，加入姜片和料酒，煮沸后捞出。",
                "duration": 10,
                "temperature": "大火",
            },
        )
    )
    store.add_edge(
        GraphEdge(
            "edge:ribs:ingredient:ribs",
            "recipe:ribs",
            "ingredient:ribs",
            "USES_INGREDIENT",
            {"quantity": "500", "unit": "克", "prep_method": "切段", "is_main": True},
        )
    )
    store.add_edge(
        GraphEdge(
            "edge:ribs:ingredient:ginger",
            "recipe:ribs",
            "ingredient:ginger",
            "USES_INGREDIENT",
            {"quantity": "3", "unit": "片"},
        )
    )
    store.add_edge(GraphEdge("edge:ribs:step:1", "recipe:ribs", "step:ribs:1", "HAS_STEP", {"order": 1}))

    result = GraphRAGService(store=store, max_depth=2).query("红烧排骨怎么做")

    assert "猪排骨 500克" in result.answer
    assert "排骨冷水下锅" in result.answer
    evidence_content = result.raw_evidence[0]["content"]
    assert "食材清单" in evidence_content
    assert "制作步骤" in evidence_content
    assert "约10分钟" in evidence_content


def test_graphrag_compare_common_ingredients_uses_intent_template() -> None:
    # 两个菜谱的共同食材属于定向多跳查询：Recipe -> Ingredient <- Recipe。
    result = get_graphrag_service().query("宫保鸡丁和麻婆豆腐有什么共同食材")

    assert "辣椒" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "recipe_compare"
    assert metadata["template_id"] == "common_ingredients_v1"


def test_graphrag_steps_question_uses_steps_template() -> None:
    # 明确只问步骤时，GraphRAG 应使用步骤模板，而不是抽取所有直接关系。
    result = get_graphrag_service().query("宫保鸡丁的步骤是什么")

    assert "炒香干辣椒" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "recipe_steps"
    assert metadata["template_id"] == "recipe_steps_v1"


def test_graphrag_recipe_ingredient_amount_template() -> None:
    result = get_graphrag_service().query("宫保鸡丁里鸡肉用量是多少")

    assert "250克" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "recipe_ingredient_amount"
    assert metadata["template_id"] == "recipe_ingredient_amount_v1"
    assert metadata["planner_provider"] == "rule"
    assert metadata["planner_confidence"] >= 0.8


def test_graphrag_recipe_tools_template() -> None:
    result = get_graphrag_service().query("宫保鸡丁需要什么工具")

    assert "炒锅" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "recipe_tools"
    assert metadata["template_id"] == "recipe_tools_v1"


def test_graphrag_cuisine_to_recipes_template() -> None:
    result = get_graphrag_service().query("川菜有哪些菜")

    assert "宫保鸡丁" in result.answer
    assert "麻婆豆腐" in result.answer
    metadata = result.raw_evidence[0]["metadata"]
    assert metadata["graph_intent"] == "cuisine_to_recipes"
    assert metadata["template_id"] == "cuisine_to_recipes_v1"


def test_graphrag_fuzzy_recipe_name_match() -> None:
    result = get_graphrag_service().query("宫保鸡叮需要哪些食材")

    assert "鸡肉" in result.answer
    assert result.linked_entities[0].node.name == "宫保鸡丁"
