"""Router 单元测试模块。"""

import json
from types import SimpleNamespace

import httpx

from app.core import router as router_module
from app.core.route_plan import plan_parallel_routes
from app.core.router import route_question
from app.models import RouteDecision, RouteType


def test_router_general() -> None:
    # 明确问候语应进入 general，而不是被短文本规则误判为 clarify。
    for question in ("你好", "帮助: 我可以怎么提问？"):
        decision = route_question(question, {"has_image": False, "has_file": False})
        assert decision.route_type == RouteType.GENERAL
        assert decision.need_clarification is False
        assert decision.slots["router_provider"] == "rule"


def test_router_text2sql() -> None:
    # 统计/排名类关键词应优先识别为 Text2SQL。
    decision = route_question("统计一下川菜菜谱数量排名", {"has_image": False, "has_file": False})
    assert decision.route_type == RouteType.TEXT2SQL
    assert decision.confidence >= 0.6
    assert decision.slots["router_provider"] == "rule"


def test_router_recipe_ingredient_amount_prefers_graphrag() -> None:
    # “多少”既可能是统计，也可能是食材用量；菜谱内食材用量应优先走 GraphRAG。
    for question in ("宫保鸡丁里鸡肉用量是多少", "红烧排骨要放多少姜", "麻婆豆腐豆瓣酱放几勺"):
        decision = route_question(question, {"has_image": False, "has_file": False})
        assert decision.route_type == RouteType.GRAPHRAG
        assert decision.slots["ingredient_amount_intent"] is True


def test_router_ingredient_reverse_recipe_questions_use_graphrag() -> None:
    for question in ("猪肉可以做什么菜", "猪肉可以做哪些菜", "豆腐能做哪些菜"):
        decision = route_question(question, {"has_image": False, "has_file": False})
        assert decision.route_type == RouteType.GRAPHRAG


def test_router_text2sql_count_questions_still_use_text2sql() -> None:
    for question in ("川菜有多少道菜谱", "统计一下每个菜系的菜谱数量", "数据库里有多少道菜"):
        decision = route_question(question, {"has_image": False, "has_file": False})
        assert decision.route_type == RouteType.TEXT2SQL


def test_router_short_hi_does_not_match_inside_food_names() -> None:
    for question in (
        "ORIGINAL APPLE CHIPS 有哪些配料？",
        "Turkey Hill Dairy Inc. 品牌有多少种食品产品？",
    ):
        decision = route_question(question, {"has_image": False, "has_file": False})
        assert decision.route_type != RouteType.GENERAL


def test_router_food_nutrition_relation_and_explanation_split() -> None:
    relation = route_question("WESSON Vegetable Oil 的营养标签和主要营养素有哪些？", {})
    explanation = route_question("解释一下 Dairy and Egg Products 这类食品的营养标签含义", {})
    field = route_question("USDA FoodData Central 的 fdc_id 是什么？", {})

    assert relation.route_type == RouteType.GRAPHRAG
    assert explanation.route_type == RouteType.KB
    assert field.route_type == RouteType.KB


def test_router_pronoun_relation_without_context_clarifies() -> None:
    decision = route_question("它属于什么？", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.CLARIFY


def test_route_plan_does_not_split_single_sql_category_stat() -> None:
    plan = plan_parallel_routes(
        "食品产品按分类统计数量并排序",
        {},
        RouteDecision(
            route_type=RouteType.TEXT2SQL,
            confidence=0.82,
            reason="统计问题。",
            slots={},
            need_clarification=False,
        ),
    )

    assert plan.is_multi is False


def test_router_recipe_howto_with_dish_name(monkeypatch) -> None:
    # 无 LLM 配置时，规则兜底也能识别“菜名 + 怎么做”。
    _patch_router_settings_disabled(monkeypatch)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})
    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.need_clarification is False
    assert decision.slots["router_provider"] == "rule"


def test_router_recipe_howto_without_subject_still_clarifies() -> None:
    # 没有菜名或附件上下文时，“这个怎么做”仍然信息不足，需要反问。
    decision = route_question("这个怎么做", {"has_image": False, "has_file": False})
    assert decision.route_type == RouteType.CLARIFY
    assert decision.need_clarification is True


def test_router_image_first() -> None:
    # 只要存在图片附件，就先进入图片理解链路。
    decision = route_question("这是什么菜", {"has_image": True, "has_file": False})
    assert decision.route_type == RouteType.IMAGE
    assert decision.slots["router_provider"] == "rule"


def test_llm_router_routes_recipe_howto_to_graphrag(monkeypatch) -> None:
    # 非确定性文本会交给 LLM Router，LLM 输出通过校验后直接成为 RouteDecision。
    _patch_router_settings(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return _router_response(
            {
                "route_type": "graphrag",
                "confidence": 0.88,
                "reason": "用户询问红烧肉做法，属于菜谱步骤问题。",
                "slots": {"dish_name": "红烧肉", "intent": "recipe_steps"},
                "need_clarification": False,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert captured["url"] == "http://router.local/v1/chat/completions"
    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.need_clarification is False
    assert decision.slots["router_provider"] == "llm"
    assert decision.slots["router_model"] == "router-test"
    assert decision.slots["dish_name"] == "红烧肉"


def test_dashscope_router_uses_native_generation_endpoint(monkeypatch) -> None:
    _patch_router_settings_dashscope(monkeypatch)
    captured: dict[str, object] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        payload = {
            "route_type": "kb",
            "confidence": 0.86,
            "reason": "用户询问食品概念解释。",
            "slots": {"knowledge_intent": True},
            "need_clarification": False,
        }
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"output": {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]}},
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉在饮食里有什么背景", {"has_image": False, "has_file": False})

    assert captured["url"] == "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    request_payload = captured["payload"]
    assert request_payload["model"] == "router-ft"
    assert request_payload["input"]["messages"][0]["role"] == "system"
    assert request_payload["parameters"]["result_format"] == "message"
    assert decision.route_type == RouteType.KB
    assert decision.slots["router_provider"] == "dashscope"
    assert decision.slots["router_model"] == "router-ft"


def test_dashscope_router_invalid_json_falls_back_to_rule(monkeypatch) -> None:
    _patch_router_settings_dashscope(monkeypatch)

    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"output": {"choices": [{"message": {"content": "not json"}}]}},
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.slots["router_provider"] == "rule"
    assert decision.slots["fallback_used"] is True
    assert "llm_json_parse_failed" in decision.slots["fallback_reason"]


def test_llm_router_routes_knowledge_question_to_kb(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return _router_response(
            {
                "route_type": "kb",
                "confidence": 0.86,
                "reason": "用户询问红烧肉历史文化。",
                "slots": {"dish_name": "红烧肉", "intent": "history"},
                "need_clarification": False,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉在饮食里有什么背景", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.KB
    assert decision.slots["router_provider"] == "llm"


def test_llm_router_routes_analysis_question_to_text2sql(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return _router_response(
            {
                "route_type": "text2sql",
                "confidence": 0.84,
                "reason": "用户想分析菜谱受欢迎程度。",
                "slots": {"metric": "popularity"},
                "need_clarification": False,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("哪些菜更受欢迎", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.TEXT2SQL
    assert decision.slots["router_provider"] == "llm"


def test_llm_router_invalid_json_falls_back_to_rule(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "not json"}}]},
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.slots["router_provider"] == "rule"
    assert decision.slots["fallback_used"] is True
    assert "llm_json_parse_failed" in decision.slots["fallback_reason"]


def test_llm_router_unknown_route_falls_back_to_rule(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return _router_response(
            {
                "route_type": "recipe_agent",
                "confidence": 0.91,
                "reason": "非法路由。",
                "slots": {},
                "need_clarification": False,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.slots["router_provider"] == "rule"
    assert decision.slots["fallback_used"] is True


def test_llm_router_low_confidence_falls_back_to_rule(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return _router_response(
            {
                "route_type": "kb",
                "confidence": 0.2,
                "reason": "置信度过低。",
                "slots": {},
                "need_clarification": False,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.slots["router_provider"] == "rule"
    assert "llm_low_confidence" in decision.slots["fallback_reason"]


def test_llm_router_clarify_for_recipe_howto_is_overridden(monkeypatch) -> None:
    # LLM 如果把明确“菜名 + 怎么做”错判成 clarify，要用规则结果纠偏。
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        return _router_response(
            {
                "route_type": "clarify",
                "confidence": 0.95,
                "reason": "误判为信息不足。",
                "slots": {},
                "need_clarification": True,
            }
        )

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧排骨怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.need_clarification is False
    assert decision.slots["router_provider"] == "rule"
    assert decision.slots["fallback_used"] is True
    assert decision.slots["fallback_reason"] == "llm_route_overridden:clarify"


def test_llm_router_http_failure_falls_back_to_rule(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        raise httpx.ConnectError("router unavailable")

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("红烧肉怎么做", {"has_image": False, "has_file": False})

    assert decision.route_type == RouteType.GRAPHRAG
    assert decision.slots["router_provider"] == "rule"
    assert decision.slots["fallback_used"] is True


def test_deterministic_file_route_does_not_call_llm(monkeypatch) -> None:
    _patch_router_settings(monkeypatch)

    def fake_post(url, **kwargs):
        raise AssertionError("file route should not call LLM Router")

    monkeypatch.setattr(router_module.httpx, "post", fake_post)
    decision = route_question("请解析这个文件", {"has_image": False, "has_file": True})

    assert decision.route_type == RouteType.FILE
    assert decision.slots["router_provider"] == "rule"


def _patch_router_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "settings",
        SimpleNamespace(
            route_confidence_threshold=0.6,
            router_llm_enabled=True,
            router_llm_provider="openai-compatible",
            router_llm_base_url="http://router.local/v1",
            router_llm_api_key="router-key",
            router_llm_model="router-test",
            router_llm_timeout_seconds=5,
            router_llm_temperature=0,
            router_llm_max_retries=0,
        ),
    )


def _patch_router_settings_dashscope(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "settings",
        SimpleNamespace(
            route_confidence_threshold=0.6,
            router_llm_enabled=True,
            router_llm_provider="dashscope",
            router_llm_base_url="https://dashscope.aliyuncs.com/api/v1",
            router_llm_api_key="router-key",
            router_llm_model="router-ft",
            router_llm_timeout_seconds=5,
            router_llm_temperature=0,
            router_llm_max_retries=0,
        ),
    )


def _patch_router_settings_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        router_module,
        "settings",
        SimpleNamespace(
            route_confidence_threshold=0.6,
            router_llm_enabled=False,
            router_llm_base_url=None,
            router_llm_api_key=None,
            router_llm_model=None,
            router_llm_timeout_seconds=5,
            router_llm_temperature=0,
            router_llm_max_retries=0,
        ),
    )


def _router_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://router.local/v1/chat/completions"),
        json={"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
    )
