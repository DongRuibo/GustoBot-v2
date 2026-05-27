"""GraphRAG 意图规划测试。

这里重点验证自然问题会被映射到白名单路径模板，而不是退回无目标的全量扩散。
"""

import json

import httpx

from app.graphrag.models import GraphIntent, GraphNode, LinkedEntity
from app.graphrag import planner as planner_module
from app.graphrag.models import GraphQueryPlan
from app.graphrag.planner import GraphIntentPlanner, GraphPlannerLLMConfig, LLMGraphIntentPlanner
from app.graphrag.templates import (
    COMMON_INGREDIENTS_TEMPLATE,
    CUISINE_TO_RECIPES_TEMPLATE,
    DIRECT_NEIGHBORS_TEMPLATE,
    INGREDIENT_TO_RECIPES_TEMPLATE,
    RECIPE_DETAIL_TEMPLATE,
    RECIPE_INGREDIENT_AMOUNT_TEMPLATE,
    RECIPE_INGREDIENTS_TEMPLATE,
    RECIPE_STEPS_TEMPLATE,
    RECIPE_TOOLS_TEMPLATE,
)


def _linked(node_id: str, label: str, name: str) -> LinkedEntity:
    return LinkedEntity(node=GraphNode(node_id, label, name), matched_text=name, score=1.0)


def test_planner_recipe_howto_uses_detail_template() -> None:
    plan = GraphIntentPlanner().plan("红烧排骨怎么做", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.graph_intent == GraphIntent.RECIPE_DETAIL
    assert plan.template_id == RECIPE_DETAIL_TEMPLATE
    assert plan.params["recipe_node_id"] == "recipe:ribs"


def test_planner_recipe_steps_uses_steps_template() -> None:
    plan = GraphIntentPlanner().plan("红烧排骨的步骤是什么", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.graph_intent == GraphIntent.RECIPE_STEPS
    assert plan.template_id == RECIPE_STEPS_TEMPLATE


def test_planner_recipe_ingredients_uses_ingredients_template() -> None:
    plan = GraphIntentPlanner().plan("红烧排骨需要哪些食材", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.graph_intent == GraphIntent.RECIPE_INGREDIENTS
    assert plan.template_id == RECIPE_INGREDIENTS_TEMPLATE


def test_planner_recipe_ingredient_amount_uses_amount_template() -> None:
    linked_entities = [
        _linked("recipe:gongbao", "Recipe", "宫保鸡丁"),
        _linked("ingredient:chicken", "Ingredient", "鸡肉"),
    ]

    plan = GraphIntentPlanner().plan("宫保鸡丁里鸡肉用量是多少", linked_entities)

    assert plan.graph_intent == GraphIntent.RECIPE_INGREDIENT_AMOUNT
    assert plan.template_id == RECIPE_INGREDIENT_AMOUNT_TEMPLATE
    assert plan.params == {
        "recipe_node_id": "recipe:gongbao",
        "ingredient_node_id": "ingredient:chicken",
    }


def test_planner_recipe_tools_uses_tools_template() -> None:
    plan = GraphIntentPlanner().plan("宫保鸡丁需要什么工具", [_linked("recipe:gongbao", "Recipe", "宫保鸡丁")])

    assert plan.graph_intent == GraphIntent.RECIPE_TOOLS
    assert plan.template_id == RECIPE_TOOLS_TEMPLATE


def test_planner_ingredient_to_recipe_uses_reverse_template() -> None:
    plan = GraphIntentPlanner().plan("白菜能做什么菜", [_linked("ingredient:cabbage", "Ingredient", "白菜")])

    assert plan.graph_intent == GraphIntent.INGREDIENT_TO_RECIPES
    assert plan.template_id == INGREDIENT_TO_RECIPES_TEMPLATE
    assert plan.params["ingredient_node_id"] == "ingredient:cabbage"
    assert plan.params["ingredient_node_ids"] == ["ingredient:cabbage"]
    assert plan.params["category_node_ids"] == []


def test_planner_ingredient_category_to_recipe_uses_reverse_template() -> None:
    plan = GraphIntentPlanner().plan(
        "猪肉可以做什么菜",
        [_linked("ingredient_category:pork", "IngredientCategory", "猪肉")],
    )

    assert plan.graph_intent == GraphIntent.INGREDIENT_TO_RECIPES
    assert plan.template_id == INGREDIENT_TO_RECIPES_TEMPLATE
    assert plan.start_node_ids == ["ingredient_category:pork"]
    assert "ingredient_node_id" not in plan.params
    assert plan.params["ingredient_node_ids"] == []
    assert plan.params["category_node_ids"] == ["ingredient_category:pork"]


def test_planner_keeps_specific_category_over_parent_category() -> None:
    plan = GraphIntentPlanner().plan(
        "猪肉可以做什么菜",
        [
            _linked("ingredient_category:pork", "IngredientCategory", "猪肉"),
            _linked("ingredient_category:meat", "IngredientCategory", "肉类"),
        ],
    )

    assert plan.graph_intent == GraphIntent.INGREDIENT_TO_RECIPES
    assert plan.template_id == INGREDIENT_TO_RECIPES_TEMPLATE
    assert plan.start_node_ids == ["ingredient_category:pork"]
    assert plan.params["category_node_ids"] == ["ingredient_category:pork"]


def test_planner_two_recipes_compare_uses_common_ingredients_template() -> None:
    linked_entities = [
        _linked("recipe:gongbao", "Recipe", "宫保鸡丁"),
        _linked("recipe:mapo", "Recipe", "麻婆豆腐"),
    ]

    plan = GraphIntentPlanner().plan("宫保鸡丁和麻婆豆腐有什么共同食材", linked_entities)

    assert plan.graph_intent == GraphIntent.RECIPE_COMPARE
    assert plan.template_id == COMMON_INGREDIENTS_TEMPLATE
    assert plan.params == {"recipe_a_id": "recipe:gongbao", "recipe_b_id": "recipe:mapo"}


def test_planner_cuisine_to_recipes_uses_cuisine_template() -> None:
    plan = GraphIntentPlanner().plan("川菜有哪些菜", [_linked("cuisine:sichuan", "Cuisine", "川菜")])

    assert plan.graph_intent == GraphIntent.CUISINE_TO_RECIPES
    assert plan.template_id == CUISINE_TO_RECIPES_TEMPLATE
    assert plan.params["cuisine_node_id"] == "cuisine:sichuan"


def test_hybrid_planner_rule_high_confidence_does_not_call_llm(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        raise AssertionError("规则高置信命中时不应调用 LLM Planner")

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = LLMGraphIntentPlanner(config=_planner_config())

    plan = planner.plan("红烧排骨怎么做", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.template_id == RECIPE_DETAIL_TEMPLATE
    assert plan.planner_provider == "rule"


def test_hybrid_planner_low_confidence_accepts_valid_llm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return _planner_response(
            {
                "template_id": RECIPE_STEPS_TEMPLATE,
                "params": {"recipe_node_id": "recipe:ribs"},
                "confidence": 0.86,
                "reason": "用户询问步骤。",
            }
        )

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = LLMGraphIntentPlanner(rule_planner=_LowConfidenceRulePlanner(), config=_planner_config())

    plan = planner.plan("红烧排骨流程", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert captured["url"] == "http://planner.local/v1/chat/completions"
    assert plan.template_id == RECIPE_STEPS_TEMPLATE
    assert plan.params["recipe_node_id"] == "recipe:ribs"
    assert plan.planner_provider == "llm"
    assert plan.planner_model == "planner-test"


def test_hybrid_planner_unknown_template_falls_back(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        return _planner_response(
            {
                "template_id": "free_cypher_v1",
                "params": {},
                "confidence": 0.9,
                "reason": "非法模板。",
            }
        )

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = LLMGraphIntentPlanner(rule_planner=_LowConfidenceRulePlanner(), config=_planner_config())

    plan = planner.plan("红烧排骨流程", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.template_id == DIRECT_NEIGHBORS_TEMPLATE
    assert plan.planner_provider == "rule_fallback"
    assert "llm_unknown_template" in str(plan.fallback_reason)


def test_hybrid_planner_illegal_param_falls_back(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        return _planner_response(
            {
                "template_id": RECIPE_STEPS_TEMPLATE,
                "params": {"recipe_node_id": "recipe:invented"},
                "confidence": 0.9,
                "reason": "凭空创造节点。",
            }
        )

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = LLMGraphIntentPlanner(rule_planner=_LowConfidenceRulePlanner(), config=_planner_config())

    plan = planner.plan("红烧排骨流程", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.template_id == DIRECT_NEIGHBORS_TEMPLATE
    assert plan.planner_provider == "rule_fallback"
    assert "llm_param_not_linked" in str(plan.fallback_reason)


def test_hybrid_planner_low_confidence_falls_back(monkeypatch) -> None:
    def fake_post(url, **kwargs):
        return _planner_response(
            {
                "template_id": RECIPE_STEPS_TEMPLATE,
                "params": {"recipe_node_id": "recipe:ribs"},
                "confidence": 0.2,
                "reason": "置信度不足。",
            }
        )

    monkeypatch.setattr(planner_module.httpx, "post", fake_post)
    planner = LLMGraphIntentPlanner(rule_planner=_LowConfidenceRulePlanner(), config=_planner_config())

    plan = planner.plan("红烧排骨流程", [_linked("recipe:ribs", "Recipe", "红烧排骨")])

    assert plan.template_id == DIRECT_NEIGHBORS_TEMPLATE
    assert plan.planner_provider == "rule_fallback"
    assert "llm_low_confidence" in str(plan.fallback_reason)


class _LowConfidenceRulePlanner:
    def plan(self, question: str, linked_entities: list[LinkedEntity]) -> GraphQueryPlan:
        return GraphQueryPlan(
            graph_intent=GraphIntent.UNKNOWN,
            template_id=DIRECT_NEIGHBORS_TEMPLATE,
            start_node_ids=[entity.node.node_id for entity in linked_entities],
            reason="测试用低置信规则结果。",
            confidence=0.2,
        )


def _planner_config() -> GraphPlannerLLMConfig:
    return GraphPlannerLLMConfig(
        base_url="http://planner.local/v1",
        model="planner-test",
        api_key="planner-key",
        timeout_seconds=5,
        temperature=0,
        max_retries=0,
        confidence_threshold=0.6,
    )


def _planner_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://planner.local/v1/chat/completions"),
        json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
    )
