"""答案生成层测试。"""

from types import SimpleNamespace

import httpx
import pytest

from app.answer import service as answer_module
from app.answer.service import AnswerGenerationInput, AnswerGenerationService
from app.models import Evidence, EvidenceSource, RouteType


def test_llm_answer_keeps_source_citations(monkeypatch) -> None:
    # LLM 即使没有主动写 source_id，答案层也要补上来源，避免最终回答无引用。
    monkeypatch.setattr(
        answer_module,
        "settings",
        SimpleNamespace(
            answer_llm_base_url="http://llm.local/v1",
            answer_llm_api_key=None,
            answer_llm_model="test-chat",
            answer_llm_timeout_seconds=5,
            trace_enabled=False,
        ),
    )

    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://llm.local/v1/chat/completions"),
            json={"choices": [{"message": {"content": "宫保鸡丁主要使用鸡肉和花生。"}}]},
        )

    monkeypatch.setattr(answer_module.httpx, "post", fake_post)
    answer = AnswerGenerationService().generate(
        AnswerGenerationInput(
            question="宫保鸡丁需要哪些食材",
            route_type=RouteType.GRAPHRAG,
            raw_answer="草稿",
            evidences=[
                Evidence(
                    source_type=EvidenceSource.GRAPH,
                    content="宫保鸡丁 -[使用食材]-> 鸡肉",
                    score=0.9,
                    source_id="graphrag_subgraph",
                    trace_id="trace-test",
                )
            ],
        )
    )

    assert "宫保鸡丁" in answer
    assert "来源" in answer
    assert "graphrag_subgraph" in answer


def test_answer_prompt_includes_conversation_history() -> None:
    prompt = AnswerGenerationService()._build_user_prompt(
        AnswerGenerationInput(
            question="你还记得我叫什么吗？",
            route_type=RouteType.GENERAL,
            raw_answer="草稿",
            conversation_history=[{"role": "user", "content": "你好，我叫冻睿博。"}],
            evidences=[
                Evidence(
                    source_type=EvidenceSource.GENERAL,
                    content="通用闲聊",
                    score=1.0,
                    source_id="general_node",
                    trace_id="trace-test",
                )
            ],
        )
    )

    assert "当前会话历史" in prompt
    assert "[user] 你好，我叫冻睿博。" in prompt


def test_llm_failure_falls_back_to_template_with_citation(monkeypatch) -> None:
    monkeypatch.setattr(
        answer_module,
        "settings",
        SimpleNamespace(
            answer_llm_base_url="http://llm.local/v1",
            answer_llm_api_key=None,
            answer_llm_model="test-chat",
            answer_llm_timeout_seconds=5,
            trace_enabled=False,
        ),
    )

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("llm unavailable")

    monkeypatch.setattr(answer_module.httpx, "post", failing_post)
    answer = AnswerGenerationService().generate(
        AnswerGenerationInput(
            question="介绍一下宫保鸡丁",
            route_type=RouteType.KB,
            raw_answer="根据知识库资料，宫保鸡丁是经典川菜。",
            evidences=[
                Evidence(
                    source_type=EvidenceSource.KB,
                    content="宫保鸡丁是经典川菜。",
                    score=0.8,
                    source_id="seed:gongbao-history:0",
                    trace_id="trace-test",
                )
            ],
        )
    )

    assert "经典川菜" in answer
    assert "seed:gongbao-history:0" in answer


def test_llm_failure_raises_in_strict_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        answer_module,
        "settings",
        SimpleNamespace(
            answer_llm_base_url="http://llm.local/v1",
            answer_llm_api_key="test-key",
            answer_llm_model="test-chat",
            answer_llm_timeout_seconds=5,
            answer_llm_max_retries=0,
            strict_external_stores=True,
            trace_enabled=False,
        ),
    )

    def failing_post(*args, **kwargs):
        raise httpx.ConnectError("llm unavailable")

    monkeypatch.setattr(answer_module.httpx, "post", failing_post)

    with pytest.raises(RuntimeError, match="answer llm request failed"):
        AnswerGenerationService().generate(
            AnswerGenerationInput(
                question="介绍一下宫保鸡丁",
                route_type=RouteType.KB,
                raw_answer="根据知识库资料，宫保鸡丁是经典川菜。",
                evidences=[
                    Evidence(
                        source_type=EvidenceSource.KB,
                        content="宫保鸡丁是经典川菜。",
                        score=0.8,
                        source_id="kb:doc",
                        trace_id="trace-test",
                    )
                ],
            )
        )


def test_answer_normalizes_llm_source_block(monkeypatch) -> None:
    monkeypatch.setattr(
        answer_module,
        "settings",
        SimpleNamespace(
            answer_llm_base_url="http://llm.local/v1",
            answer_llm_api_key=None,
            answer_llm_model="test-chat",
            answer_llm_timeout_seconds=5,
            answer_llm_max_retries=0,
            trace_enabled=False,
        ),
    )

    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", "http://llm.local/v1/chat/completions"),
            json={"choices": [{"message": {"content": "宫保鸡丁通常需要鸡肉、花生和辣椒。\n\n来源：old-source"}}]},
        )

    monkeypatch.setattr(answer_module.httpx, "post", fake_post)
    answer = AnswerGenerationService().generate(
        AnswerGenerationInput(
            question="宫保鸡丁需要哪些食材？",
            route_type=RouteType.GRAPHRAG,
            raw_answer="宫保鸡丁需要鸡肉、花生和辣椒。",
            evidences=[
                Evidence(
                    source_type=EvidenceSource.GRAPH,
                    content="宫保鸡丁 -> 需要食材 -> 鸡肉",
                    score=0.9,
                    source_id="graphrag_subgraph",
                    trace_id="trace-test",
                )
            ],
        )
    )

    assert answer.count("来源：") == 1
    assert "old-source" not in answer
    assert "graphrag_subgraph" in answer
