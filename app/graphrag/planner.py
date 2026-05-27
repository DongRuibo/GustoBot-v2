"""GraphRAG 意图规划器。

Planner 负责把已经链接到的图谱实体转换成白名单路径模板。
规则 planner 保持确定性；LLM planner 只在规则低置信或未知时补充选择 template_id。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any

import httpx

from app.core.config import settings
from app.graphrag.ingredient_taxonomy import category_rules
from app.graphrag.models import GraphIntent, GraphQueryPlan, LinkedEntity
from app.graphrag.templates import (
    ALLERGEN_TO_PRODUCTS_TEMPLATE,
    COMMON_INGREDIENTS_TEMPLATE,
    CUISINE_TO_RECIPES_TEMPLATE,
    DIRECT_NEIGHBORS_TEMPLATE,
    INGREDIENT_TO_RECIPES_TEMPLATE,
    PRODUCT_ALLERGENS_TEMPLATE,
    PRODUCT_CATEGORY_TEMPLATE,
    PRODUCT_DETAIL_TEMPLATE,
    PRODUCT_INGREDIENTS_TEMPLATE,
    RECIPE_DETAIL_TEMPLATE,
    RECIPE_INGREDIENT_AMOUNT_TEMPLATE,
    RECIPE_INGREDIENTS_TEMPLATE,
    RECIPE_STEPS_TEMPLATE,
    RECIPE_TOOLS_TEMPLATE,
    TEMPLATE_INTENTS,
    is_known_template,
)


STEP_KEYWORDS = ("步骤", "第几步", "烹饪步骤", "制作步骤")
HOWTO_KEYWORDS = ("怎么做", "如何做", "做法", "制作方法", "烹饪方法", "怎么烧", "怎么煮", "怎么炒", "怎么炖", "怎么蒸")
INGREDIENT_KEYWORDS = ("需要哪些食材", "有哪些食材", "食材", "主料", "辅料", "调料")
AMOUNT_KEYWORDS = ("用量", "多少", "几克", "几两", "几勺", "放多少", "要多少", "比例")
INGREDIENT_TO_RECIPE_KEYWORDS = ("能做什么菜", "做什么菜", "哪些菜", "可以做")
TOOL_KEYWORDS = ("工具", "器具", "厨具", "锅", "用什么锅", "用什么工具", "需要什么工具")
CUISINE_TO_RECIPE_KEYWORDS = ("有哪些菜", "哪些菜", "有什么菜", "菜谱", "能做什么", "推荐")
COMPARE_KEYWORDS = ("共同", "相同", "区别", "对比", "比较", "都用", "都需要")
PRODUCT_INGREDIENT_KEYWORDS = ("配料", "成分", "ingredients", "ingredient")
PRODUCT_ALLERGEN_KEYWORDS = ("过敏原", "过敏", "allergen", "含有")
PRODUCT_CATEGORY_KEYWORDS = ("类别", "分类", "属于", "category")
PRODUCT_DETAIL_KEYWORDS = ("商品", "产品", "食品", "营养标签", "营养")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RULE_HIGH_CONFIDENCE = 0.8


class GraphPlannerLLMError(RuntimeError):
    """LLM Graph Planner 调用或输出校验失败。"""


@dataclass(slots=True)
class GraphPlannerLLMConfig:
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    temperature: float
    max_retries: int
    confidence_threshold: float


class GraphIntentPlanner:
    """规则版 GraphRAG 查询规划器。"""

    def plan(self, question: str, linked_entities: list[LinkedEntity]) -> GraphQueryPlan:
        if not linked_entities:
            return GraphQueryPlan(
                graph_intent=GraphIntent.UNKNOWN,
                template_id=DIRECT_NEIGHBORS_TEMPLATE,
                start_node_ids=[],
                reason="没有可用的已链接实体，退回直接关系模板。",
                confidence=0.0,
            )

        recipes = [entity for entity in linked_entities if entity.node.label == "Recipe"]
        ingredients = [entity for entity in linked_entities if entity.node.label == "Ingredient"]
        categories = _prefer_specific_categories(
            [entity for entity in linked_entities if entity.node.label == "IngredientCategory"]
        )
        cuisines = [entity for entity in linked_entities if entity.node.label == "Cuisine"]
        products = [entity for entity in linked_entities if entity.node.label == "Product"]
        allergens = [entity for entity in linked_entities if entity.node.label == "Allergen"]
        primary = linked_entities[0]

        if primary.node.label == "Allergen" or (allergens and _has_any(question, PRODUCT_ALLERGEN_KEYWORDS)):
            allergen = primary if primary.node.label == "Allergen" else allergens[0]
            return GraphQueryPlan(
                graph_intent=GraphIntent.ALLERGEN_TO_PRODUCTS,
                template_id=ALLERGEN_TO_PRODUCTS_TEMPLATE,
                start_node_ids=[allergen.node.node_id],
                params={"allergen_node_id": allergen.node.node_id},
                reason="用户从过敏原反查食品商品，使用过敏原到产品模板。",
                confidence=0.9,
            )

        if primary.node.label == "Product" or products:
            product = primary if primary.node.label == "Product" else products[0]
            if _has_any(question, PRODUCT_ALLERGEN_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.PRODUCT_ALLERGENS,
                    template_id=PRODUCT_ALLERGENS_TEMPLATE,
                    start_node_ids=[product.node.node_id],
                    params={"product_node_id": product.node.node_id},
                    reason="用户询问食品商品过敏原关系，读取 Product -> Allergen。",
                    confidence=0.9,
                )
            if _has_any(question, PRODUCT_INGREDIENT_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.PRODUCT_INGREDIENTS,
                    template_id=PRODUCT_INGREDIENTS_TEMPLATE,
                    start_node_ids=[product.node.node_id],
                    params={"product_node_id": product.node.node_id},
                    reason="用户询问食品商品配料关系，读取 Product -> Ingredient。",
                    confidence=0.9,
                )
            asks_product_category = _has_any(question, PRODUCT_CATEGORY_KEYWORDS)
            asks_product_nutrition = _has_any(question, ("营养标签", "营养"))
            if asks_product_nutrition or (_has_any(question, PRODUCT_DETAIL_KEYWORDS) and not asks_product_category):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.PRODUCT_DETAIL,
                    template_id=PRODUCT_DETAIL_TEMPLATE,
                    start_node_ids=[product.node.node_id],
                    params={"product_node_id": product.node.node_id},
                    reason="用户询问食品商品关系概览，读取商品直接关系。",
                    confidence=0.84,
                )
            if asks_product_category:
                return GraphQueryPlan(
                    graph_intent=GraphIntent.PRODUCT_CATEGORY,
                    template_id=PRODUCT_CATEGORY_TEMPLATE,
                    start_node_ids=[product.node.node_id],
                    params={"product_node_id": product.node.node_id},
                    reason="用户询问食品商品分类关系，读取 Product -> FoodCategory。",
                    confidence=0.9,
                )

        if len(recipes) >= 2 and _has_any(question, COMPARE_KEYWORDS):
            return GraphQueryPlan(
                graph_intent=GraphIntent.RECIPE_COMPARE,
                template_id=COMMON_INGREDIENTS_TEMPLATE,
                start_node_ids=[recipes[0].node.node_id, recipes[1].node.node_id],
                params={
                    "recipe_a_id": recipes[0].node.node_id,
                    "recipe_b_id": recipes[1].node.node_id,
                },
                reason="问题包含两个菜谱和对比/共同关系意图，使用共同食材模板。",
                confidence=0.92,
            )

        if recipes and ingredients and _has_any(question, AMOUNT_KEYWORDS):
            return GraphQueryPlan(
                graph_intent=GraphIntent.RECIPE_INGREDIENT_AMOUNT,
                template_id=RECIPE_INGREDIENT_AMOUNT_TEMPLATE,
                start_node_ids=[recipes[0].node.node_id, ingredients[0].node.node_id],
                params={
                    "recipe_node_id": recipes[0].node.node_id,
                    "ingredient_node_id": ingredients[0].node.node_id,
                },
                reason="问题同时命中菜谱和食材，并询问用量，使用菜谱-食材用量模板。",
                confidence=0.94,
            )

        if primary.node.label == "Recipe":
            if _has_any(question, STEP_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.RECIPE_STEPS,
                    template_id=RECIPE_STEPS_TEMPLATE,
                    start_node_ids=[primary.node.node_id],
                    params={"recipe_node_id": primary.node.node_id},
                    reason="用户明确询问菜谱步骤，只读取步骤和步骤工具关系。",
                    confidence=0.9,
                )
            if _has_any(question, HOWTO_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.RECIPE_DETAIL,
                    template_id=RECIPE_DETAIL_TEMPLATE,
                    start_node_ids=[primary.node.node_id],
                    params={"recipe_node_id": primary.node.node_id},
                    reason="用户询问菜谱做法，读取当前菜谱的食材、步骤、工具和基础信息。",
                    confidence=0.9,
                )
            if _has_any(question, TOOL_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.RECIPE_TOOLS,
                    template_id=RECIPE_TOOLS_TEMPLATE,
                    start_node_ids=[primary.node.node_id],
                    params={"recipe_node_id": primary.node.node_id},
                    reason="用户询问菜谱需要的工具，读取步骤到工具的白名单关系。",
                    confidence=0.9,
                )
            if _has_any(question, INGREDIENT_KEYWORDS):
                return GraphQueryPlan(
                    graph_intent=GraphIntent.RECIPE_INGREDIENTS,
                    template_id=RECIPE_INGREDIENTS_TEMPLATE,
                    start_node_ids=[primary.node.node_id],
                    params={"recipe_node_id": primary.node.node_id},
                    reason="用户询问菜谱食材，读取当前菜谱的食材关系。",
                    confidence=0.9,
                )
            return GraphQueryPlan(
                graph_intent=GraphIntent.RECIPE_DETAIL,
                template_id=RECIPE_DETAIL_TEMPLATE,
                start_node_ids=[primary.node.node_id],
                params={"recipe_node_id": primary.node.node_id},
                reason="已命中菜谱实体，默认读取菜谱详情关系。",
                confidence=0.82,
            )

        if primary.node.label in {"Ingredient", "IngredientCategory"}:
            if _has_any(question, INGREDIENT_TO_RECIPE_KEYWORDS) or not recipes:
                ingredient_node_ids = [entity.node.node_id for entity in ingredients]
                # 只有具体食材是首要命中时才忽略类别；类别首命中说明用户更像是在问泛化词。
                category_node_ids = [] if ingredients and primary.node.label == "Ingredient" else [entity.node.node_id for entity in categories]
                start_node_ids = [*ingredient_node_ids, *category_node_ids]
                params: dict[str, object] = {
                    "ingredient_node_ids": ingredient_node_ids,
                    "category_node_ids": category_node_ids,
                }
                if ingredients:
                    params["ingredient_node_id"] = ingredients[0].node.node_id
                return GraphQueryPlan(
                    graph_intent=GraphIntent.INGREDIENT_TO_RECIPES,
                    template_id=INGREDIENT_TO_RECIPES_TEMPLATE,
                    start_node_ids=start_node_ids,
                    params=params,
                    reason="用户从食材或食材类别反查菜谱，使用食材/类别到菜谱模板。",
                    confidence=0.88,
                )

        if primary.node.label == "Cuisine" or cuisines:
            cuisine = primary if primary.node.label == "Cuisine" else cuisines[0]
            if _has_any(question, CUISINE_TO_RECIPE_KEYWORDS) or primary.node.label == "Cuisine":
                return GraphQueryPlan(
                    graph_intent=GraphIntent.CUISINE_TO_RECIPES,
                    template_id=CUISINE_TO_RECIPES_TEMPLATE,
                    start_node_ids=[cuisine.node.node_id],
                    params={"cuisine_node_id": cuisine.node.node_id},
                    reason="用户从菜系反查菜谱，使用菜系到菜谱模板。",
                    confidence=0.88,
                )

        return GraphQueryPlan(
            graph_intent=GraphIntent.UNKNOWN,
            template_id=DIRECT_NEIGHBORS_TEMPLATE,
            start_node_ids=[entity.node.node_id for entity in linked_entities],
            reason="未命中更具体的图谱意图，退回直接关系模板。",
            confidence=0.35,
        )


class LLMGraphIntentPlanner:
    """规则优先的 GraphRAG Planner；LLM 只补充选择白名单模板。"""

    def __init__(
        self,
        rule_planner: GraphIntentPlanner | None = None,
        config: GraphPlannerLLMConfig | None = None,
    ) -> None:
        self.rule_planner = rule_planner or GraphIntentPlanner()
        self.config = config

    def plan(self, question: str, linked_entities: list[LinkedEntity]) -> GraphQueryPlan:
        rule_plan = self.rule_planner.plan(question, linked_entities)
        if _is_high_confidence_rule_plan(rule_plan):
            return rule_plan

        llm_config = self.config or _graph_planner_llm_config()
        if llm_config is None:
            return rule_plan

        try:
            payload = self._call_llm(question, linked_entities, llm_config)
            llm_plan = _plan_from_llm_payload(payload, linked_entities, llm_config)
        except Exception as exc:
            return replace(
                rule_plan,
                planner_provider="rule_fallback",
                planner_model=llm_config.model,
                fallback_reason=str(exc)[:200],
            )
        return llm_plan

    def _call_llm(
        self,
        question: str,
        linked_entities: list[LinkedEntity],
        config: GraphPlannerLLMConfig,
    ) -> dict[str, Any]:
        response = _post_planner_llm(
            config,
            {
                "model": config.model,
                "temperature": config.temperature,
                "messages": [
                    {"role": "system", "content": _planner_system_prompt()},
                    {"role": "user", "content": _planner_user_prompt(question, linked_entities)},
                ],
            },
        )
        content = _chat_content(response.json())
        return _parse_json_object(content)


def _graph_planner_llm_config() -> GraphPlannerLLMConfig | None:
    if not settings.graph_planner_llm_enabled:
        return None
    if not settings.graph_planner_llm_base_url or not settings.graph_planner_llm_model:
        return None
    return GraphPlannerLLMConfig(
        base_url=settings.graph_planner_llm_base_url,
        model=settings.graph_planner_llm_model,
        api_key=settings.graph_planner_llm_api_key,
        timeout_seconds=settings.graph_planner_llm_timeout_seconds,
        temperature=settings.graph_planner_llm_temperature,
        max_retries=settings.graph_planner_llm_max_retries,
        confidence_threshold=settings.graph_planner_llm_confidence_threshold,
    )


def _plan_from_llm_payload(
    payload: dict[str, Any],
    linked_entities: list[LinkedEntity],
    config: GraphPlannerLLMConfig,
) -> GraphQueryPlan:
    allowed_fields = {"template_id", "params", "confidence", "reason"}
    extra_fields = set(payload) - allowed_fields
    if extra_fields:
        raise GraphPlannerLLMError(f"llm_extra_fields:{sorted(extra_fields)}")

    missing_fields = allowed_fields - set(payload)
    if missing_fields:
        raise GraphPlannerLLMError(f"llm_missing_fields:{sorted(missing_fields)}")

    template_id = str(payload["template_id"])
    if not is_known_template(template_id):
        raise GraphPlannerLLMError(f"llm_unknown_template:{template_id}")
    if template_id == DIRECT_NEIGHBORS_TEMPLATE:
        raise GraphPlannerLLMError("llm_direct_neighbors_not_needed")

    params = payload["params"]
    if not isinstance(params, dict):
        raise GraphPlannerLLMError("llm_params_not_object")

    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise GraphPlannerLLMError("llm_invalid_confidence") from exc
    if confidence < config.confidence_threshold:
        raise GraphPlannerLLMError(f"llm_low_confidence:{confidence}")

    start_node_ids, normalized_params = _validate_llm_params(template_id, params, linked_entities)
    return GraphQueryPlan(
        graph_intent=TEMPLATE_INTENTS[template_id],
        template_id=template_id,
        start_node_ids=start_node_ids,
        params=normalized_params,
        reason=str(payload["reason"]).strip() or "LLM Graph Planner 选择白名单模板。",
        confidence=confidence,
        planner_provider="llm",
        planner_model=config.model,
    )


def _validate_llm_params(
    template_id: str,
    params: dict[str, Any],
    linked_entities: list[LinkedEntity],
) -> tuple[list[str], dict[str, Any]]:
    linked_by_id = {entity.node.node_id: entity for entity in linked_entities}

    if template_id in {RECIPE_DETAIL_TEMPLATE, RECIPE_INGREDIENTS_TEMPLATE, RECIPE_STEPS_TEMPLATE, RECIPE_TOOLS_TEMPLATE}:
        _ensure_only_params(params, {"recipe_node_id"})
        recipe_node_id = _node_id_param(params, "recipe_node_id", "Recipe", linked_by_id)
        return [recipe_node_id], {"recipe_node_id": recipe_node_id}

    if template_id == RECIPE_INGREDIENT_AMOUNT_TEMPLATE:
        _ensure_only_params(params, {"recipe_node_id", "ingredient_node_id"})
        recipe_node_id = _node_id_param(params, "recipe_node_id", "Recipe", linked_by_id)
        ingredient_node_id = _node_id_param(params, "ingredient_node_id", "Ingredient", linked_by_id)
        return [recipe_node_id, ingredient_node_id], {
            "recipe_node_id": recipe_node_id,
            "ingredient_node_id": ingredient_node_id,
        }

    if template_id == INGREDIENT_TO_RECIPES_TEMPLATE:
        _ensure_only_params(params, {"ingredient_node_id", "ingredient_node_ids", "category_node_ids"})
        ingredient_node_ids = _node_id_list_param(params, "ingredient_node_ids", "Ingredient", linked_by_id)
        if not ingredient_node_ids and params.get("ingredient_node_id"):
            ingredient_node_ids = [_node_id_param(params, "ingredient_node_id", "Ingredient", linked_by_id)]
        category_node_ids = _node_id_list_param(params, "category_node_ids", "IngredientCategory", linked_by_id)
        start_node_ids = [*ingredient_node_ids, *category_node_ids]
        if not start_node_ids:
            raise GraphPlannerLLMError("llm_missing_ingredient_or_category_param")
        normalized_params: dict[str, Any] = {
            "ingredient_node_ids": ingredient_node_ids,
            "category_node_ids": category_node_ids,
        }
        if ingredient_node_ids:
            normalized_params["ingredient_node_id"] = ingredient_node_ids[0]
        return start_node_ids, normalized_params

    if template_id == CUISINE_TO_RECIPES_TEMPLATE:
        _ensure_only_params(params, {"cuisine_node_id"})
        cuisine_node_id = _node_id_param(params, "cuisine_node_id", "Cuisine", linked_by_id)
        return [cuisine_node_id], {"cuisine_node_id": cuisine_node_id}

    if template_id == COMMON_INGREDIENTS_TEMPLATE:
        _ensure_only_params(params, {"recipe_a_id", "recipe_b_id"})
        recipe_a_id = _node_id_param(params, "recipe_a_id", "Recipe", linked_by_id)
        recipe_b_id = _node_id_param(params, "recipe_b_id", "Recipe", linked_by_id)
        return [recipe_a_id, recipe_b_id], {"recipe_a_id": recipe_a_id, "recipe_b_id": recipe_b_id}

    if template_id in {PRODUCT_DETAIL_TEMPLATE, PRODUCT_INGREDIENTS_TEMPLATE, PRODUCT_ALLERGENS_TEMPLATE, PRODUCT_CATEGORY_TEMPLATE}:
        _ensure_only_params(params, {"product_node_id"})
        product_node_id = _node_id_param(params, "product_node_id", "Product", linked_by_id)
        return [product_node_id], {"product_node_id": product_node_id}

    if template_id == ALLERGEN_TO_PRODUCTS_TEMPLATE:
        _ensure_only_params(params, {"allergen_node_id"})
        allergen_node_id = _node_id_param(params, "allergen_node_id", "Allergen", linked_by_id)
        return [allergen_node_id], {"allergen_node_id": allergen_node_id}

    raise GraphPlannerLLMError(f"llm_template_without_validator:{template_id}")


def _ensure_only_params(params: dict[str, Any], allowed: set[str]) -> None:
    extra = set(params) - allowed
    if extra:
        raise GraphPlannerLLMError(f"llm_extra_params:{sorted(extra)}")


def _node_id_param(
    params: dict[str, Any],
    key: str,
    expected_label: str,
    linked_by_id: dict[str, LinkedEntity],
) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value:
        raise GraphPlannerLLMError(f"llm_missing_param:{key}")
    entity = linked_by_id.get(value)
    if entity is None:
        raise GraphPlannerLLMError(f"llm_param_not_linked:{key}")
    if entity.node.label != expected_label:
        raise GraphPlannerLLMError(f"llm_param_label_mismatch:{key}:{entity.node.label}")
    return value


def _node_id_list_param(
    params: dict[str, Any],
    key: str,
    expected_label: str,
    linked_by_id: dict[str, LinkedEntity],
) -> list[str]:
    value = params.get(key)
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise GraphPlannerLLMError(f"llm_param_list_expected:{key}")

    normalized: list[str] = []
    for item in values:
        if not isinstance(item, str) or not item:
            raise GraphPlannerLLMError(f"llm_invalid_param_item:{key}")
        entity = linked_by_id.get(item)
        if entity is None:
            raise GraphPlannerLLMError(f"llm_param_not_linked:{key}")
        if entity.node.label != expected_label:
            raise GraphPlannerLLMError(f"llm_param_label_mismatch:{key}:{entity.node.label}")
        normalized.append(item)
    return list(dict.fromkeys(normalized))


def _planner_system_prompt() -> str:
    return (
        "你是 GustoBot-v2 的 GraphRAG Planner，只能选择白名单图谱模板并填充参数。"
        "禁止生成 Cypher，禁止创造节点 ID，禁止输出 Markdown。"
        "只返回 JSON，字段只能是 template_id、params、confidence、reason。"
    )


def _planner_user_prompt(question: str, linked_entities: list[LinkedEntity]) -> str:
    entities = [
        {
            "node_id": entity.node.node_id,
            "label": entity.node.label,
            "name": entity.node.name,
            "matched_text": entity.matched_text,
            "score": entity.score,
        }
        for entity in linked_entities
    ]
    template_specs = [
        {"template_id": RECIPE_DETAIL_TEMPLATE, "params": {"recipe_node_id": "Recipe"}},
        {"template_id": RECIPE_INGREDIENTS_TEMPLATE, "params": {"recipe_node_id": "Recipe"}},
        {"template_id": RECIPE_INGREDIENT_AMOUNT_TEMPLATE, "params": {"recipe_node_id": "Recipe", "ingredient_node_id": "Ingredient"}},
        {"template_id": RECIPE_STEPS_TEMPLATE, "params": {"recipe_node_id": "Recipe"}},
        {"template_id": RECIPE_TOOLS_TEMPLATE, "params": {"recipe_node_id": "Recipe"}},
        {"template_id": INGREDIENT_TO_RECIPES_TEMPLATE, "params": {"ingredient_node_ids": ["Ingredient"], "category_node_ids": ["IngredientCategory"]}},
        {"template_id": CUISINE_TO_RECIPES_TEMPLATE, "params": {"cuisine_node_id": "Cuisine"}},
        {"template_id": COMMON_INGREDIENTS_TEMPLATE, "params": {"recipe_a_id": "Recipe", "recipe_b_id": "Recipe"}},
        {"template_id": PRODUCT_DETAIL_TEMPLATE, "params": {"product_node_id": "Product"}},
        {"template_id": PRODUCT_INGREDIENTS_TEMPLATE, "params": {"product_node_id": "Product"}},
        {"template_id": PRODUCT_ALLERGENS_TEMPLATE, "params": {"product_node_id": "Product"}},
        {"template_id": PRODUCT_CATEGORY_TEMPLATE, "params": {"product_node_id": "Product"}},
        {"template_id": ALLERGEN_TO_PRODUCTS_TEMPLATE, "params": {"allergen_node_id": "Allergen"}},
    ]
    return (
        f"用户问题：{question}\n"
        f"已链接实体：{json.dumps(entities, ensure_ascii=False)}\n"
        f"可选模板：{json.dumps(template_specs, ensure_ascii=False)}\n"
        '请返回 JSON，例如：{"template_id":"recipe_detail_v1","params":{"recipe_node_id":"recipe:1"},"confidence":0.8,"reason":"..."}'
    )


def _post_planner_llm(config: GraphPlannerLLMConfig, payload: dict[str, Any]) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    last_exc: Exception | None = None
    for attempt in range(max(0, config.max_retries) + 1):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=config.timeout_seconds)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < config.max_retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= config.max_retries:
                raise GraphPlannerLLMError(str(exc)) from exc
            time.sleep(0.2 * (attempt + 1))
    raise GraphPlannerLLMError(str(last_exc) if last_exc else "llm_request_failed")


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        if match is None:
            raise GraphPlannerLLMError("llm_json_parse_failed")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise GraphPlannerLLMError("llm_json_not_object")
    return data


def _chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        raise GraphPlannerLLMError("llm_empty_choices")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def _is_high_confidence_rule_plan(plan: GraphQueryPlan) -> bool:
    return plan.template_id != DIRECT_NEIGHBORS_TEMPLATE and plan.confidence >= RULE_HIGH_CONFIDENCE


def _prefer_specific_categories(categories: list[LinkedEntity]) -> list[LinkedEntity]:
    if len(categories) <= 1:
        return categories
    selected_slugs = {
        slug
        for entity in categories
        if (slug := _category_slug_from_node_id(entity.node.node_id))
    }
    if not selected_slugs:
        return categories
    rules_by_slug = {rule.slug: rule for rule in category_rules()}
    ancestor_slugs: set[str] = set()
    for slug in selected_slugs:
        ancestor_slugs.update(_category_ancestors(slug, rules_by_slug))
    return [
        entity
        for entity in categories
        if _category_slug_from_node_id(entity.node.node_id) not in ancestor_slugs
    ]


def _category_ancestors(slug: str, rules_by_slug: dict[str, Any]) -> set[str]:
    ancestors: set[str] = set()
    pending = list(getattr(rules_by_slug.get(slug), "parent_slugs", ()))
    while pending:
        current = pending.pop()
        if current in ancestors:
            continue
        ancestors.add(current)
        pending.extend(getattr(rules_by_slug.get(current), "parent_slugs", ()))
    return ancestors


def _category_slug_from_node_id(node_id: str) -> str | None:
    prefix = "ingredient_category:"
    if not node_id.startswith(prefix):
        return None
    return node_id[len(prefix) :]


def _has_any(question: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in question for keyword in keywords)
