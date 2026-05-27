"""LangGraph 工作流测试模块。

这个文件从内部调用视角验证完整问答流程，包括 General、Clarify、KB RAG、
图片占位链路和 Global Guardrails 拦截，确保节点串联后仍能返回统一响应结构。
"""

from app.graph.workflow import run_chat
from app.kb.embeddings import HashEmbeddingProvider
from app.kb.reranker import KeywordReranker
from app.kb.service import KnowledgeBaseService, reset_kb_service_for_tests
from app.kb.store import InMemoryKnowledgeStore
from app.models import Attachment, ChatRequest, RouteType


def test_workflow_general() -> None:
    # 普通问候应走 General 节点并返回统一 Evidence。
    response = run_chat(ChatRequest(message="你好"))
    assert response.route_decision.route_type == RouteType.GENERAL
    assert response.evidences[0].trace_id == response.trace_id
    assert "GustoBot-v2" in response.answer


def test_workflow_clarify_low_info() -> None:
    # 信息不足时不猜测业务链路，直接进入 Clarify 反问。
    response = run_chat(ChatRequest(message="这个呢"))
    assert response.need_clarification is True
    assert response.route_decision.route_type == RouteType.CLARIFY


def test_workflow_text2sql() -> None:
    # 第三阶段 Text2SQL 已接入，统计类问题应返回 SQL Evidence，而不是 Clarify 占位。
    response = run_chat(ChatRequest(message="统计一下每个菜系的菜谱数量"))
    assert response.route_decision.route_type == RouteType.TEXT2SQL
    assert "SQL 查询结果" in response.answer
    assert response.evidences[0].source_type.value == "sql"


def test_workflow_kb_rag() -> None:
    # KB 知识类问题现在应进入 KB RAG，并返回统一 Evidence，而不是继续走 Clarify 占位。
    response = run_chat(ChatRequest(message="介绍一下宫保鸡丁的历史和文化"))
    assert response.route_decision.route_type == RouteType.KB
    assert response.evidences
    assert response.evidences[0].source_type.value == "kb"
    assert "宫保鸡丁" in response.answer


def test_workflow_kb_rag_exposes_hybrid_retrieval_metadata() -> None:
    response = run_chat(ChatRequest(message="宫保鸡丁有什么历史"))

    assert response.route_decision.route_type == RouteType.KB
    kb_metadata = response.evidences[0].metadata
    assert kb_metadata["retrieval_mode"] in {"hybrid", "vector", "lexical"}
    assert "retrieval_sources" in kb_metadata


def test_workflow_graphrag() -> None:
    # 第三阶段 GraphRAG 已接入，关系类问题应返回图谱 Evidence。
    response = run_chat(ChatRequest(message="宫保鸡丁需要哪些食材"))
    assert response.route_decision.route_type == RouteType.GRAPHRAG
    assert response.evidences[0].source_type.value == "graph"
    assert "鸡肉" in response.answer


def test_workflow_recipe_ingredient_amount_uses_graphrag_template() -> None:
    response = run_chat(ChatRequest(message="宫保鸡丁里鸡肉用量是多少"))

    assert response.route_decision.route_type == RouteType.GRAPHRAG
    assert response.evidences[0].source_type.value == "graph"
    assert response.evidences[0].metadata["graph_intent"] == "recipe_ingredient_amount"
    assert response.evidences[0].metadata["template_id"] == "recipe_ingredient_amount_v1"
    assert "250克" in response.answer


def test_workflow_recipe_howto_routes_to_graphrag() -> None:
    # “菜名 + 怎么做”属于步骤/做法问题，应进入 GraphRAG，而不是 clarify。
    response = run_chat(ChatRequest(message="宫保鸡丁怎么做"))
    assert response.route_decision.route_type == RouteType.GRAPHRAG
    assert response.need_clarification is False
    assert response.route_decision.slots["router_provider"] in {"rule", "llm"}
    assert response.evidences[0].source_type.value == "graph"


def test_workflow_multi_route_kb_and_graphrag() -> None:
    response = run_chat(ChatRequest(message="介绍宫保鸡丁的历史，并说明它需要哪些食材"))

    assert response.route_decision.route_type == RouteType.MULTI
    assert response.need_clarification is False
    assert response.route_decision.slots["executed_routes"] == ["kb", "graphrag"]
    evidence_sources = {evidence.source_type.value for evidence in response.evidences}
    assert {"kb", "graph"} <= evidence_sources
    assert "知识库信息" in response.answer
    assert "图谱信息" in response.answer


def test_workflow_multi_route_text2sql_and_kb() -> None:
    response = run_chat(ChatRequest(message="统计每个菜系的菜谱数量，并介绍宫保鸡丁的历史"))

    assert response.route_decision.route_type == RouteType.MULTI
    assert response.route_decision.slots["executed_routes"] == ["text2sql", "kb"]
    evidence_sources = {evidence.source_type.value for evidence in response.evidences}
    assert {"sql", "kb"} <= evidence_sources
    assert "统计结果" in response.answer
    assert "知识库信息" in response.answer


def test_workflow_image_reroutes_to_graphrag() -> None:
    # 第四阶段图片链路会先理解图片，再把结构化文本重新交给 Router 进入 GraphRAG。
    response = run_chat(
        ChatRequest(
            message="这张图里的菜需要哪些食材",
            attachments=[Attachment(type="image", filename="gongbao.jpg")],
        )
    )
    assert response.route_decision.route_type == RouteType.GRAPHRAG
    assert response.evidences[0].source_type.value == "image"
    assert response.evidences[-1].source_type.value == "graph"
    assert "鸡肉" in response.answer


def test_workflow_file_ingest() -> None:
    # 文件附件会被解析并写入 KB，节点返回文件 Evidence 和入库 chunk 数。
    response = run_chat(
        ChatRequest(
            message="请把这个文件入库",
            attachments=[
                Attachment(
                    type="file",
                    filename="佛跳墙资料.txt",
                    text="佛跳墙是闽菜代表菜，常见于宴席文化介绍。",
                )
            ],
        )
    )
    assert response.route_decision.route_type == RouteType.FILE
    assert response.evidences[0].source_type.value == "file"
    assert "已完成文件入库" in response.answer


def test_workflow_file_ingest_not_parallelized() -> None:
    # 文件入库是副作用链路，即使用户同时提出介绍诉求，也应优先完成入库而不触发 multi。
    response = run_chat(
        ChatRequest(
            message="请把这个文件入库，并介绍内容",
            attachments=[
                Attachment(
                    type="file",
                    filename="闽菜资料.txt",
                    text="佛跳墙属于闽菜，常用于宴席文化介绍。",
                )
            ],
        )
    )

    assert response.route_decision.route_type == RouteType.FILE
    assert response.evidences[0].source_type.value == "file"
    assert response.route_decision.slots.get("executed_routes") is None


def test_workflow_file_ingest_followup_falls_back_to_kb() -> None:
    # 上传文件只进入 KB；当后续食材问题被先路由到 GraphRAG 但图谱无命中时，应回退到 KB 检索文件内容。
    reset_kb_service_for_tests(_build_isolated_kb_service())
    try:
        ingest_response = run_chat(
            ChatRequest(
                message="请把这个文件入库",
                attachments=[
                    Attachment(
                        type="file",
                        filename="菜谱测试.txt",
                        text="砂锅鸡汤主要食材：鸡肉、姜片、枸杞、香菇。汤底使用清水和少量盐。",
                    )
                ],
            )
        )
        assert ingest_response.route_decision.route_type == RouteType.FILE

        response = run_chat(ChatRequest(message="砂锅鸡汤主要有哪些食材？"))
    finally:
        reset_kb_service_for_tests(None)

    assert response.route_decision.route_type == RouteType.KB
    assert response.evidences
    assert response.evidences[0].source_type.value == "kb"
    assert response.evidences[0].metadata["fallback_from"] == "graphrag"
    assert "鸡肉" in response.answer


def test_workflow_guardrails_block() -> None:
    # 危险 SQL 写操作应在 Router 前被 Global Guardrails 拦截。
    response = run_chat(ChatRequest(message="帮我 drop table recipes"))
    assert response.need_clarification is True
    assert response.evidences[0].source_type.value == "guardrail"
    assert "不能继续处理" in response.answer


def _build_isolated_kb_service() -> KnowledgeBaseService:
    return KnowledgeBaseService(
        store=InMemoryKnowledgeStore(),
        embedding_provider=HashEmbeddingProvider(dimension=64),
        reranker=KeywordReranker(),
        chunk_size=80,
        chunk_overlap=10,
        retrieve_top_k=5,
        rerank_top_k=2,
    )
