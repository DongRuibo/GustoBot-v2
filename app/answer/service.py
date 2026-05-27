"""统一答案生成服务。

答案层只消费标准 Evidence，不直接访问 KB、Neo4j 或 SQL。
如果配置 OpenAI-compatible LLM，则基于证据生成自然语言回答；否则使用上游节点的确定性答案并补充来源。
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings
from app.models import Evidence, EvidenceSource, RouteType
from app.observability.tracing import record_trace_event


@dataclass(slots=True)
class AnswerGenerationInput:
    question: str
    route_type: RouteType | None
    raw_answer: str
    evidences: list[Evidence]
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    trace_id: str | None = None


class AnswerGenerationService:
    # 服务本身无状态，配置从 settings 读取，便于测试时通过环境变量切换。
    def generate(self, payload: AnswerGenerationInput) -> str:
        if payload.blocked:
            return "抱歉，这个请求包含危险或不适合执行的内容，我不能继续处理。"

        raw_answer = payload.raw_answer.strip()
        if self._requires_evidence(payload.route_type) and not payload.evidences:
            if self._can_use_general_recipe_fallback(payload):
                generated = self._try_generate_general_recipe_fallback(payload)
                if generated:
                    self._record_answer_event(
                        payload,
                        mode="llm_general_recipe_fallback",
                        answer=generated,
                    )
                    return generated
            return "当前没有检索到可引用证据，我不能基于猜测回答这个问题。请补充更明确的问题或先完成资料入库。"

        if self._llm_configured() and payload.evidences:
            generated = self._try_generate_with_llm(payload)
            if generated:
                answer = self._append_citations(generated, payload.evidences)
                self._record_answer_event(payload, mode="llm", answer=answer)
                return answer
        if getattr(settings, "strict_external_stores", False) and payload.evidences:
            raise RuntimeError("生产环境必须成功调用答案 LLM，不能回退模板答案。")

        if not raw_answer:
            raw_answer = "当前信息不足，我需要你再补充一点问题背景。"
        answer = self._append_citations(raw_answer, payload.evidences)
        self._record_answer_event(payload, mode="template", answer=answer)
        return answer

    def _try_generate_with_llm(self, payload: AnswerGenerationInput) -> str | None:
        started = time.perf_counter()
        user_prompt = self._build_user_prompt(payload)
        try:
            response = self._post_llm(
                {
                    "model": settings.answer_llm_model,
                    "temperature": 0.2,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是 GustoBot-v2 的答案生成层。只能依据给定 Evidence 和当前会话历史回答；"
                                "证据或历史不足时明确说明不足，不要编造。涉及 Evidence 时在回答末尾用“来源：”列出 source_id；"
                                "如果答案只来自当前会话历史，可以写“来源：当前会话”。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            generated = str(content).strip()
            self._record_trace(
                payload,
                "answer_llm_finished",
                {
                    "model": settings.answer_llm_model,
                    "latency_ms": _elapsed_ms(started),
                    "evidence_count": len(payload.evidences),
                    "prompt_summary": self._prompt_summary(payload),
                    "answer_preview": generated[:200],
                },
            )
            return generated
        except Exception as exc:
            if getattr(settings, "strict_external_stores", False):
                raise RuntimeError(f"answer llm request failed: {str(exc)[:160]}") from exc
            # LLM 失败不能影响主流程，回退到确定性答案和来源引用。
            self._record_trace(
                payload,
                "answer_llm_failed",
                {
                    "model": settings.answer_llm_model,
                    "latency_ms": _elapsed_ms(started),
                    "error": str(exc)[:300],
                    "evidence_count": len(payload.evidences),
                    "prompt_summary": self._prompt_summary(payload),
                },
            )
            return None

    def _try_generate_general_recipe_fallback(self, payload: AnswerGenerationInput) -> str | None:
        started = time.perf_counter()
        try:
            response = self._post_llm(
                {
                    "model": settings.answer_llm_model,
                    "temperature": 0.3,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "你是 GustoBot-v2 的菜谱兜底回答层。当前本地知识库和图谱没有检索到可引用证据。"
                                "你可以基于通用烹饪知识给出家庭做法参考，但必须第一句说明："
                                "“本地库暂未检索到可引用菜谱，以下是通用做法参考。”"
                                "不要声称答案来自数据库、图谱或知识库，不要编造来源。"
                            ),
                        },
                        {"role": "user", "content": f"用户问题：{payload.question}"},
                    ],
                },
            )
            response.raise_for_status()
            generated = str(response.json()["choices"][0]["message"]["content"]).strip()
            self._record_trace(
                payload,
                "answer_general_recipe_fallback_finished",
                {
                    "model": settings.answer_llm_model,
                    "latency_ms": _elapsed_ms(started),
                    "answer_preview": generated[:200],
                },
            )
            return generated
        except Exception as exc:
            self._record_trace(
                payload,
                "answer_general_recipe_fallback_failed",
                {
                    "model": settings.answer_llm_model,
                    "latency_ms": _elapsed_ms(started),
                    "error": str(exc)[:300],
                },
            )
            return None

    def _post_llm(self, payload: dict[str, Any]) -> httpx.Response:
        max_retries = max(0, int(getattr(settings, "answer_llm_max_retries", 1)))
        url = f"{settings.answer_llm_base_url.rstrip('/')}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                response = httpx.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=settings.answer_llm_timeout_seconds,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    raise
                time.sleep(0.2 * (attempt + 1))
        raise last_exc or RuntimeError("answer llm request failed")

    def _build_user_prompt(self, payload: AnswerGenerationInput) -> str:
        history_text = _format_conversation_history(payload.conversation_history)
        evidence_text = "\n".join(
            f"[{index}] source_id={evidence.source_id}; source_type={evidence.source_type.value}; "
            f"score={evidence.score:.3f}; content={evidence.content}"
            for index, evidence in enumerate(payload.evidences, start=1)
        )
        return (
            f"用户问题：{payload.question}\n"
            f"路由类型：{payload.route_type.value if payload.route_type else 'unknown'}\n"
            f"当前会话历史：\n{history_text}\n"
            f"上游答案草稿：{payload.raw_answer}\n"
            f"Evidence：\n{evidence_text}"
        )

    def _append_citations(self, answer: str, evidences: list[Evidence]) -> str:
        citeable = [
            evidence
            for evidence in evidences
            if evidence.source_type
            not in {EvidenceSource.GENERAL, EvidenceSource.CLARIFY, EvidenceSource.GUARDRAIL}
        ]
        if not citeable:
            return answer
        source_ids = []
        for evidence in citeable:
            if evidence.source_id not in source_ids:
                source_ids.append(evidence.source_id)
        normalized_answer = self._strip_existing_source_block(answer)
        if all(source_id in normalized_answer for source_id in source_ids[:5]):
            return normalized_answer
        return f"{normalized_answer}\n\n来源：{', '.join(source_ids[:5])}"

    def _strip_existing_source_block(self, answer: str) -> str:
        lines = answer.rstrip().splitlines()
        for index, line in enumerate(lines):
            if re.match(r"^\s*(来源|Sources?|References?)\s*[:：]", line, flags=re.I):
                return "\n".join(lines[:index]).rstrip()
        return answer.rstrip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.answer_llm_api_key:
            headers["Authorization"] = f"Bearer {settings.answer_llm_api_key}"
        return headers

    def _llm_configured(self) -> bool:
        return bool(settings.answer_llm_base_url and settings.answer_llm_model)

    def _requires_evidence(self, route_type: RouteType | None) -> bool:
        return route_type in {RouteType.KB, RouteType.GRAPHRAG, RouteType.TEXT2SQL, RouteType.MULTI}

    def _can_use_general_recipe_fallback(self, payload: AnswerGenerationInput) -> bool:
        return (
            payload.route_type == RouteType.GRAPHRAG
            and self._llm_configured()
            and not getattr(settings, "strict_external_stores", False)
            and getattr(settings, "answer_llm_allow_general_recipe_fallback", True)
            and _has_recipe_howto_intent(payload.question)
        )

    def _record_answer_event(self, payload: AnswerGenerationInput, *, mode: str, answer: str) -> None:
        self._record_trace(
            payload,
            "answer_generated",
            {
                "mode": mode,
                "route_type": payload.route_type.value if payload.route_type else None,
                "evidence_count": len(payload.evidences),
                "answer_preview": answer[:200],
            },
        )

    def _record_trace(self, payload: AnswerGenerationInput, event_type: str, event_payload: dict[str, Any]) -> None:
        if payload.trace_id:
            record_trace_event(payload.trace_id, event_type, event_payload)

    def _prompt_summary(self, payload: AnswerGenerationInput) -> dict[str, Any]:
        return {
            "question": payload.question[:200],
            "route_type": payload.route_type.value if payload.route_type else None,
            "history_count": len(payload.conversation_history),
            "raw_answer_preview": payload.raw_answer[:200],
            "evidences": [
                {
                    "source_type": evidence.source_type.value,
                    "source_id": evidence.source_id,
                    "score": evidence.score,
                    "content_preview": evidence.content[:160],
                }
                for evidence in payload.evidences[:5]
            ],
        }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _format_conversation_history(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in history[-12:]:
        if isinstance(item, dict):
            role = str(item.get("role") or "unknown")
            content = str(item.get("content") or "")
        else:
            role = str(getattr(item, "role", "unknown"))
            content = str(getattr(item, "content", ""))
        normalized_content = " ".join(content.split())
        if normalized_content:
            lines.append(f"[{role}] {normalized_content[:800]}")
    return "\n".join(lines) if lines else "无"


def _has_recipe_howto_intent(question: str) -> bool:
    return any(
        keyword in question
        for keyword in ("怎么做", "如何做", "做法", "制作方法", "烹饪方法", "怎么烧", "怎么煮", "怎么炒", "怎么炖", "怎么蒸")
    )


_answer_service = AnswerGenerationService()


def get_answer_generation_service() -> AnswerGenerationService:
    return _answer_service
