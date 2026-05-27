"""缓存单元测试模块。

这个文件验证第四阶段缓存抽象的基础行为，包括 JSON 写入、读取和 TTL 过期。
测试直接使用内存缓存实现，不依赖外部 Redis 服务。
"""

import time

from app.cache.store import InMemoryCacheStore, reset_cache_store_for_tests
from app.graph.workflow import (
    _graphrag_semantic_cache_key,
    _load_cached_response,
    _semantic_cache_disabled_reason_for_state,
    _store_cached_response,
    run_chat,
)
from app.graphrag.models import GraphIntent, GraphQueryPlan
from app.graphrag.service import _semantic_cache_disabled_reason
from app.graphrag.templates import COMMON_INGREDIENTS_TEMPLATE, DIRECT_NEIGHBORS_TEMPLATE, RECIPE_DETAIL_TEMPLATE
from app.models import Attachment, ChatRequest, ChatResponse, Evidence, EvidenceSource, RouteDecision, RouteType
from app.text2sql.executor import ReadOnlySQLiteExecutor
from app.text2sql.generator import RuleBasedSQLGenerator
from app.text2sql.schema import build_default_schema_catalog
from app.text2sql.service import Text2SQLService, reset_text2sql_service_for_tests
from app.text2sql.validator import SQLValidator


class CountingSQLiteExecutor(ReadOnlySQLiteExecutor):
    def __init__(self) -> None:
        super().__init__()
        self.execute_count = 0

    def execute(self, sql: str):
        self.execute_count += 1
        return super().execute(sql)


def test_in_memory_cache_store_json_roundtrip() -> None:
    # 缓存层存取的是 JSON 风格 dict，方便 Redis 和内存缓存使用同一套调用方式。
    store = InMemoryCacheStore()
    store.set_json("chat:test", {"answer": "你好"}, ttl_seconds=60)

    assert store.get_json("chat:test") == {"answer": "你好"}


def test_in_memory_cache_store_expire() -> None:
    # TTL 过期后应返回 None，避免热点缓存长时间返回旧答案。
    store = InMemoryCacheStore()
    store.set_json("chat:test", {"answer": "你好"}, ttl_seconds=0)
    time.sleep(0.01)

    assert store.get_json("chat:test") is None


def test_graphrag_semantic_cache_key_is_stable_for_same_plan() -> None:
    # 语义缓存键只看路由、模板和规范化参数，不受问题原文或 dict 插入顺序影响。
    plan_a = GraphQueryPlan(
        graph_intent=GraphIntent.RECIPE_DETAIL,
        template_id=RECIPE_DETAIL_TEMPLATE,
        start_node_ids=["recipe:hongshaorou"],
        params={"recipe_node_id": "recipe:hongshaorou"},
    )
    plan_b = GraphQueryPlan(
        graph_intent=GraphIntent.RECIPE_DETAIL,
        template_id=RECIPE_DETAIL_TEMPLATE,
        start_node_ids=["recipe:hongshaorou"],
        params=dict([("recipe_node_id", "recipe:hongshaorou")]),
    )

    assert _graphrag_semantic_cache_key(plan_a) == _graphrag_semantic_cache_key(plan_b)


def test_graphrag_semantic_cache_keeps_ordered_params() -> None:
    # 对比类模板的 recipe_a_id/recipe_b_id 保持语义顺序，避免把有序参数误归一。
    plan_a = GraphQueryPlan(
        graph_intent=GraphIntent.RECIPE_COMPARE,
        template_id=COMMON_INGREDIENTS_TEMPLATE,
        start_node_ids=["recipe:a", "recipe:b"],
        params={"recipe_a_id": "recipe:a", "recipe_b_id": "recipe:b"},
    )
    plan_b = GraphQueryPlan(
        graph_intent=GraphIntent.RECIPE_COMPARE,
        template_id=COMMON_INGREDIENTS_TEMPLATE,
        start_node_ids=["recipe:b", "recipe:a"],
        params={"recipe_a_id": "recipe:b", "recipe_b_id": "recipe:a"},
    )

    assert _graphrag_semantic_cache_key(plan_a) != _graphrag_semantic_cache_key(plan_b)


def test_attachments_disable_semantic_cache_lookup() -> None:
    state = {
        "attachments": [Attachment(type="image", filename="dish.jpg").model_dump(mode="json")],
        "route_decision": RouteDecision(
            route_type=RouteType.GRAPHRAG,
            confidence=0.95,
            reason="test",
            slots={},
            need_clarification=False,
        ),
    }

    assert _semantic_cache_disabled_reason_for_state(state) == "attachments_present"


def test_conversation_history_disables_exact_and_semantic_cache() -> None:
    store = InMemoryCacheStore()
    reset_cache_store_for_tests(store)
    try:
        response = ChatResponse(
            trace_id="trace-test",
            answer="我不知道你的名字。",
            route_decision=RouteDecision(
                route_type=RouteType.GENERAL,
                confidence=0.9,
                reason="test",
                slots={},
                need_clarification=False,
            ),
            evidences=[
                Evidence(
                    source_type=EvidenceSource.GENERAL,
                    content="test",
                    score=1.0,
                    source_id="general_node",
                    trace_id="trace-test",
                )
            ],
            need_clarification=False,
        )
        request_without_history = ChatRequest(message="你还记得我叫什么吗？")
        request_with_history = ChatRequest(
            message="你还记得我叫什么吗？",
            conversation_history=[{"role": "user", "content": "我叫冻睿博。"}],
        )
        _store_cached_response(request_without_history, response)

        assert _load_cached_response(request_without_history, "trace-cache") is not None
        assert _load_cached_response(request_with_history, "trace-cache") is None
        assert (
            _semantic_cache_disabled_reason_for_state(
                {
                    "conversation_history": [{"role": "user", "content": "我叫冻睿博。"}],
                    "route_decision": response.route_decision,
                }
            )
            == "conversation_history_present"
        )
    finally:
        reset_cache_store_for_tests(None)


def test_direct_neighbors_plan_disables_semantic_cache() -> None:
    plan = GraphQueryPlan(
        graph_intent=GraphIntent.UNKNOWN,
        template_id=DIRECT_NEIGHBORS_TEMPLATE,
        start_node_ids=["recipe:unknown"],
        params={},
        confidence=0.35,
    )

    assert _semantic_cache_disabled_reason(plan) == "unstable_template"


def test_workflow_hits_graphrag_semantic_cache_for_equivalent_howto_questions() -> None:
    store = InMemoryCacheStore()
    reset_cache_store_for_tests(store)
    try:
        first = run_chat(ChatRequest(message="宫保鸡丁怎么做"))
        second = run_chat(ChatRequest(message="宫保鸡丁的做法"))
    finally:
        reset_cache_store_for_tests(None)

    assert first.route_decision.route_type == RouteType.GRAPHRAG
    assert second.route_decision.route_type == RouteType.GRAPHRAG
    assert second.answer == first.answer
    graph_evidence = next(evidence for evidence in second.evidences if evidence.source_type.value == "graph")
    assert graph_evidence.metadata["cache_hit"] is True
    assert graph_evidence.metadata["cache_key_type"] == "semantic"
    assert graph_evidence.metadata["semantic_cache_template_id"] == RECIPE_DETAIL_TEMPLATE


def test_workflow_hits_text2sql_semantic_cache_and_skips_executor() -> None:
    store = InMemoryCacheStore()
    executor = CountingSQLiteExecutor()
    catalog = build_default_schema_catalog()
    reset_cache_store_for_tests(store)
    reset_text2sql_service_for_tests(
        Text2SQLService(
            schema_catalog=catalog,
            sql_generator=RuleBasedSQLGenerator(),
            sql_validator=SQLValidator(allowed_tables={table.name for table in catalog.tables}, max_rows=50),
            executor=executor,
            schema_top_k=2,
        )
    )
    try:
        first = run_chat(ChatRequest(message="统计一下每个菜系的菜谱数量"))
        second = run_chat(ChatRequest(message="按菜系统计菜谱数量"))
    finally:
        reset_cache_store_for_tests(None)
        reset_text2sql_service_for_tests(None)

    assert first.route_decision.route_type == RouteType.TEXT2SQL
    assert second.route_decision.route_type == RouteType.TEXT2SQL
    assert first.answer == second.answer
    assert executor.execute_count == 1
    sql_evidence = next(evidence for evidence in second.evidences if evidence.source_type.value == "sql")
    assert sql_evidence.metadata["cache_hit"] is True
    assert sql_evidence.metadata["semantic_cache_route"] == "text2sql"


def test_workflow_hits_kb_semantic_cache_for_same_retrieval_signature() -> None:
    store = InMemoryCacheStore()
    reset_cache_store_for_tests(store)
    try:
        first = run_chat(ChatRequest(message="介绍一下宫保鸡丁的历史和文化"))
        second = run_chat(ChatRequest(message="讲讲宫保鸡丁的历史文化"))
    finally:
        reset_cache_store_for_tests(None)

    assert first.route_decision.route_type == RouteType.KB
    assert second.route_decision.route_type == RouteType.KB
    assert first.answer == second.answer
    kb_evidence = next(evidence for evidence in second.evidences if evidence.source_type.value == "kb")
    assert kb_evidence.metadata["cache_hit"] is True
    assert kb_evidence.metadata["semantic_cache_route"] == "kb"
