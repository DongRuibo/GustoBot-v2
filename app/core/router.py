"""结构化 Router 模块。

Router 根据标准化后的用户输入和附件特征输出 RouteDecision。当前实现采用混合路由：
确定性规则先处理图片、文件、问候、低信息和明显统计问题；其余文本优先交给
OpenAI-compatible 或 DashScope 原生 LLM 判断，LLM 不可用或输出非法时回退到规则 Router。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings
from app.models import RouteDecision, RouteType


GENERAL_KEYWORDS = ("你好", "您好", "hello", "hi", "你是谁", "帮助")
TEXT2SQL_KEYWORDS = (
    "统计",
    "多少",
    "数量",
    "计数",
    "排名",
    "top",
    "趋势",
    "平均",
    "占比",
    "最大",
    "最小",
    "最高",
    "最低",
    "前 ",
)
INGREDIENT_AMOUNT_KEYWORDS = ("用量", "放多少", "要放多少", "需要多少", "几克", "几两", "几勺", "多少克", "多少片", "多少个")
GRAPHRAG_KEYWORDS = (
    "能做什么菜",
    "能做哪些菜",
    "可以做什么菜",
    "可以做哪些菜",
    "需要哪些食材",
    "哪些菜",
    "食材",
    "步骤",
    "调料",
    "口味",
    "搭配",
    "替换",
    "相关",
    "关系",
    "哪些产品",
    "哪些商品",
    "含有",
    "配料",
    "成分",
    "过敏原",
    "营养素",
    "营养标签",
    "营养信息",
    "类别",
    "分类",
    "属于",
)
RECIPE_HOWTO_KEYWORDS = (
    "怎么做",
    "如何做",
    "做法",
    "制作方法",
    "烹饪方法",
    "怎么烧",
    "怎么煮",
    "怎么炒",
    "怎么炖",
    "怎么蒸",
)
KB_KEYWORDS = (
    "历史",
    "来历",
    "典故",
    "文化",
    "解释",
    "介绍",
    "为什么",
    "知识",
    "作用",
    "营养素",
    "营养标签",
    "是什么意思",
    "是什么",
    "区别",
    "怎么看",
    "含义",
    "字段",
)
LOW_INFO_PATTERNS = ("这个呢", "怎么做", "帮我看看", "说一下", "分析一下")
LOW_INFO_SUBJECTS = ("这个", "这个菜", "这道菜", "它", "这个呢")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class RouterLLMError(RuntimeError):
    """LLM Router 调用或输出校验失败。"""


@dataclass(slots=True)
class RouterLLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float
    temperature: float
    max_retries: int


def route_question(
    normalized_input: str,
    input_features: dict[str, Any],
) -> RouteDecision:
    """统一 Router 入口：确定性规则 -> LLM Router -> 规则兜底。"""
    text = normalized_input.strip()
    deterministic = deterministic_route_question(text, input_features)
    if deterministic is not None:
        return _with_router_metadata(deterministic, router_provider="rule", fallback_used=False)

    llm_config = _router_llm_config()
    if llm_config is None:
        fallback = rule_route_question(text, input_features)
        return _with_router_metadata(
            fallback,
            router_provider="rule",
            fallback_used=True,
            fallback_reason="llm_not_configured",
        )

    try:
        decision = llm_route_question(text, input_features, llm_config)
        if decision.confidence < settings.route_confidence_threshold:
            raise RouterLLMError(f"llm_low_confidence:{decision.confidence}")

        rule_fallback = rule_route_question(text, input_features)
        if _should_override_llm_decision(decision, rule_fallback):
            return _with_router_metadata(
                rule_fallback,
                router_provider="rule",
                router_model=llm_config.model,
                fallback_used=True,
                fallback_reason=f"llm_route_overridden:{decision.route_type.value}",
            )

        return _with_router_metadata(
            decision,
            router_provider=_router_provider_metadata(llm_config),
            router_model=llm_config.model,
            fallback_used=False,
        )
    except Exception as exc:
        fallback = rule_route_question(text, input_features)
        return _with_router_metadata(
            fallback,
            router_provider="rule",
            router_model=llm_config.model,
            fallback_used=True,
            fallback_reason=str(exc)[:200],
        )


def deterministic_route_question(
    normalized_input: str,
    input_features: dict[str, Any],
) -> RouteDecision | None:
    """只处理不需要 LLM 判断的高确定性路由。"""
    text = normalized_input.strip()
    lowered = text.lower()

    if input_features.get("has_image"):
        return RouteDecision(
            route_type=RouteType.IMAGE,
            confidence=0.95,
            reason="检测到图片附件，优先进入图片理解链路。",
            slots={"input_modality": "image"},
            need_clarification=False,
        )

    if input_features.get("has_file"):
        return RouteDecision(
            route_type=RouteType.FILE,
            confidence=0.95,
            reason="检测到文件附件，优先进入文件解析链路。",
            slots={"input_modality": "file"},
            need_clarification=False,
        )

    if _has_general_intent(text):
        return RouteDecision(
            route_type=RouteType.GENERAL,
            confidence=0.9,
            reason="问题属于问候、闲聊或系统能力咨询。",
            slots={},
            need_clarification=False,
        )

    if _is_low_info_question(text):
        return RouteDecision(
            route_type=RouteType.CLARIFY,
            confidence=0.35,
            reason="问题信息不足，无法稳定判断业务链路。",
            slots={},
            need_clarification=True,
        )

    if _has_ingredient_amount_intent(text):
        return RouteDecision(
            route_type=RouteType.GRAPHRAG,
            confidence=0.84,
            reason="问题询问某道菜中具体食材的用量，属于菜谱-食材关系查询。",
            slots={"relation_intent": True, "ingredient_amount_intent": True},
            need_clarification=False,
        )

    if _has_food_relation_intent(text):
        return RouteDecision(
            route_type=RouteType.GRAPHRAG,
            confidence=0.86,
            reason="问题询问食品商品、配料、过敏原或分类关系，属于图谱关系查询。",
            slots={"relation_intent": True, "food_relation_intent": True},
            need_clarification=False,
        )

    if _has_kb_explanation_intent(text):
        return RouteDecision(
            route_type=RouteType.KB,
            confidence=0.78,
            reason="问题偏向食品知识、概念解释或字段说明。",
            slots={"knowledge_intent": True},
            need_clarification=False,
        )

    if any(keyword in lowered for keyword in TEXT2SQL_KEYWORDS):
        return RouteDecision(
            route_type=RouteType.TEXT2SQL,
            confidence=0.82,
            reason="问题包含统计、排名、聚合或趋势分析意图。",
            slots={"analysis_intent": True},
            need_clarification=False,
        )

    return None


def rule_route_question(
    normalized_input: str,
    input_features: dict[str, Any],
) -> RouteDecision:
    """完整规则 Router，作为无模型和模型失败时的稳定兜底。"""
    deterministic = deterministic_route_question(normalized_input, input_features)
    if deterministic is not None:
        return deterministic

    text = normalized_input.strip()

    if _has_recipe_howto_intent(text):
        return RouteDecision(
            route_type=RouteType.GRAPHRAG,
            confidence=0.78,
            reason="问题包含明确菜谱做法或步骤意图。",
            slots={"relation_intent": True, "recipe_howto_intent": True},
            need_clarification=False,
        )

    if any(keyword in text for keyword in KB_KEYWORDS) and not _has_food_relation_intent(text):
        return RouteDecision(
            route_type=RouteType.KB,
            confidence=0.76,
            reason="问题偏向食品知识、营养解释、历史文化或概念说明。",
            slots={"knowledge_intent": True},
            need_clarification=False,
        )

    if any(keyword in text for keyword in GRAPHRAG_KEYWORDS):
        return RouteDecision(
            route_type=RouteType.GRAPHRAG,
            confidence=0.78,
            reason="问题涉及菜谱、食材、步骤或关系推理。",
            slots={"relation_intent": True},
            need_clarification=False,
        )

    if any(keyword in text for keyword in KB_KEYWORDS):
        return RouteDecision(
            route_type=RouteType.KB,
            confidence=0.76,
            reason="问题偏向菜谱知识、历史、文化或解释类问答。",
            slots={"knowledge_intent": True},
            need_clarification=False,
        )

    confidence = 0.55
    return RouteDecision(
        route_type=RouteType.CLARIFY,
        confidence=confidence,
        reason="未匹配到稳定路由规则，低于当前置信度阈值。",
        slots={"threshold": settings.route_confidence_threshold},
        need_clarification=confidence < settings.route_confidence_threshold,
    )


def llm_route_question(
    normalized_input: str,
    input_features: dict[str, Any],
    config: RouterLLMConfig | None = None,
) -> RouteDecision:
    """调用配置的 LLM Router 生成结构化路由结果。"""
    llm_config = config or _router_llm_config()
    if llm_config is None:
        raise RouterLLMError("llm_not_configured")

    response = _post_router_llm(
        llm_config,
        {
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "messages": [
                {"role": "system", "content": _router_system_prompt()},
                {"role": "user", "content": _router_user_prompt(normalized_input, input_features)},
            ],
        },
    )
    content = _chat_content(response.json())
    payload = _parse_json_object(content)
    return _decision_from_llm_payload(payload, input_features)


def _router_llm_config() -> RouterLLMConfig | None:
    if not settings.router_llm_enabled:
        return None
    if not settings.router_llm_base_url or not settings.router_llm_model:
        return None
    return RouterLLMConfig(
        provider=_normalize_router_provider(getattr(settings, "router_llm_provider", "openai-compatible")),
        base_url=settings.router_llm_base_url,
        model=settings.router_llm_model,
        api_key=settings.router_llm_api_key,
        timeout_seconds=settings.router_llm_timeout_seconds,
        temperature=settings.router_llm_temperature,
        max_retries=settings.router_llm_max_retries,
    )


def _router_system_prompt() -> str:
    return (
        "你是 GustoBot-v2 的结构化 Router，只负责判断问题应进入哪条业务链路。"
        "只能返回 JSON，不要 Markdown，不要解释额外文本。"
        "route_type 只能是 general、kb、graphrag、text2sql、image、file、clarify。"
        "路由边界：做法/步骤/食材/搭配/替换/关系 -> graphrag；"
        "某个食材能做什么菜、可以做哪些菜 -> graphrag；"
        "某道菜里某个食材的用量、放多少、几克、几勺 -> graphrag；"
        "历史/文化/典故/知识解释 -> kb；"
        "统计/计数/排行/趋势/聚合 -> text2sql；"
        "问候/能力介绍/闲聊 -> general；信息不足 -> clarify。"
        "例如：'宫保鸡丁怎么做'、'红烧排骨怎么做' 必须输出 graphrag；"
        "'这个怎么做' 没有菜名或附件上下文时才输出 clarify。"
        "图片和文件由程序规则识别，纯文本问题不要输出 image/file。"
    )


def _router_user_prompt(normalized_input: str, input_features: dict[str, Any]) -> str:
    enriched_features = _router_llm_input_features(normalized_input, input_features)
    # Router SFT 的训练格式是 user content 直接放 JSON；运行时保持同分布。
    return json.dumps(
        {
            "input_features": enriched_features,
            "question": normalized_input,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _router_llm_input_features(normalized_input: str, input_features: dict[str, Any]) -> dict[str, Any]:
    # SFT 训练样本包含这些轻量意图特征，运行时也补齐，避免微调模型看到分布外输入。
    text = normalized_input.strip()
    lowered = text.lower()
    attachment_types = list(input_features.get("attachment_types") or [])
    if input_features.get("has_image") and "image" not in attachment_types:
        attachment_types.append("image")
    if input_features.get("has_file") and "file" not in attachment_types:
        attachment_types.append("file")
    features = dict(input_features)
    features.update(
        {
            "attachment_types": attachment_types,
            "contains_knowledge_intent": _has_kb_explanation_intent(text),
            "contains_relation_intent": _has_food_relation_intent(text) or _has_recipe_howto_intent(text) or _has_ingredient_amount_intent(text),
            "contains_sql_mutation": bool(re.search(r"\b(delete|drop|truncate|update|insert|alter)\b", lowered)),
            "contains_statistical_intent": any(keyword in lowered for keyword in TEXT2SQL_KEYWORDS),
            "has_attachment": bool(attachment_types or input_features.get("has_attachment")),
            "is_low_context": _is_low_info_question(text),
            "language": "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en",
            "message_length": len(text),
        }
    )
    return features


def _post_router_llm(config: RouterLLMConfig, payload: dict[str, Any]) -> httpx.Response:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if config.provider == "dashscope":
        url = f"{config.base_url.rstrip('/')}/services/aigc/text-generation/generation"
        request_payload = _dashscope_router_payload(payload)
    elif config.provider == "openai-compatible":
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        request_payload = payload
    else:
        raise RouterLLMError(f"unsupported_router_provider:{config.provider}")
    last_exc: Exception | None = None
    for attempt in range(max(0, config.max_retries) + 1):
        try:
            response = httpx.post(url, headers=headers, json=request_payload, timeout=config.timeout_seconds)
            if response.status_code in RETRYABLE_STATUS_CODES and attempt < config.max_retries:
                time.sleep(0.2 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except Exception as exc:
            last_exc = exc
            if attempt >= config.max_retries:
                raise RouterLLMError(str(exc)) from exc
            time.sleep(0.2 * (attempt + 1))
    raise RouterLLMError(str(last_exc) if last_exc else "llm_request_failed")


def _decision_from_llm_payload(payload: dict[str, Any], input_features: dict[str, Any]) -> RouteDecision:
    required = {"route_type", "confidence", "reason", "slots", "need_clarification"}
    missing = required - set(payload)
    if missing:
        raise RouterLLMError(f"llm_missing_fields:{sorted(missing)}")
    if not isinstance(payload.get("slots"), dict):
        raise RouterLLMError("llm_slots_not_object")

    try:
        decision = RouteDecision(
            route_type=payload["route_type"],
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"]).strip() or "LLM Router 生成路由结果。",
            slots=payload["slots"],
            need_clarification=bool(payload["need_clarification"]),
        )
    except Exception as exc:
        raise RouterLLMError(f"llm_invalid_payload:{exc}") from exc

    if decision.route_type == RouteType.IMAGE and not input_features.get("has_image"):
        raise RouterLLMError("llm_invalid_image_route_without_attachment")
    if decision.route_type == RouteType.FILE and not input_features.get("has_file"):
        raise RouterLLMError("llm_invalid_file_route_without_attachment")
    return decision


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
            raise RouterLLMError("llm_json_parse_failed")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise RouterLLMError("llm_json_not_object")
    return data


def _chat_content(payload: dict[str, Any]) -> str:
    output = payload.get("output")
    choices = output.get("choices") if isinstance(output, dict) else payload.get("choices")
    if not choices:
        raise RouterLLMError("llm_empty_choices")
    first_choice = choices[0] if isinstance(choices[0], dict) else {}
    message = first_choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def _dashscope_router_payload(payload: dict[str, Any]) -> dict[str, Any]:
    parameters: dict[str, Any] = {"result_format": "message"}
    if "temperature" in payload:
        parameters["temperature"] = payload["temperature"]
    return {
        "model": payload["model"],
        "input": {"messages": payload["messages"]},
        "parameters": parameters,
    }


def _normalize_router_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace("_", "-")
    if normalized in {"openai", "openai-compatible", "compatible"}:
        return "openai-compatible"
    if normalized == "dashscope":
        return "dashscope"
    return normalized


def _router_provider_metadata(config: RouterLLMConfig) -> str:
    return "dashscope" if config.provider == "dashscope" else "llm"


def _with_router_metadata(
    decision: RouteDecision,
    *,
    router_provider: str,
    fallback_used: bool,
    router_model: str | None = None,
    fallback_reason: str | None = None,
) -> RouteDecision:
    slots = dict(decision.slots)
    slots["router_provider"] = router_provider
    slots["fallback_used"] = fallback_used
    if router_model:
        slots["router_model"] = router_model
    if fallback_reason:
        slots["fallback_reason"] = fallback_reason
    return decision.model_copy(update={"slots": slots})


def _should_override_llm_decision(decision: RouteDecision, rule_fallback: RouteDecision) -> bool:
    """LLM 自信但明显偏离硬规则时，用规则结果纠偏。"""
    if (
        rule_fallback.route_type == RouteType.CLARIFY
        or rule_fallback.confidence < settings.route_confidence_threshold
    ):
        return False
    if decision.route_type == RouteType.CLARIFY or decision.need_clarification:
        return True
    if rule_fallback.slots.get("recipe_howto_intent") and decision.route_type != RouteType.GRAPHRAG:
        return True
    return False


def _is_low_info_question(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) <= 3 or any(pattern == normalized for pattern in LOW_INFO_PATTERNS):
        return True
    if re.fullmatch(r"(它|这个|这个食品|该食品)(属于什么|是什么|怎么样|有哪些)\??？?", normalized):
        return True
    for keyword in RECIPE_HOWTO_KEYWORDS:
        if keyword not in normalized:
            continue
        subject = normalized.replace(keyword, "").strip(" ?？。！，,")
        if not subject or subject in LOW_INFO_SUBJECTS:
            return True
    return False


def _has_general_intent(text: str) -> bool:
    normalized = text.strip()
    lowered = normalized.lower()
    if any(keyword in normalized for keyword in ("你好", "您好", "你是谁")):
        return True
    if re.search(r"(^|[\s,，。！？?；;:：])(?:hello|hi)(?=$|[\s,，。！？?；;:：])", lowered):
        return True
    return (
        lowered in {"help", "help me"}
        or normalized in {"帮助", "帮忙"}
        or normalized.startswith(("帮助：", "帮助:"))
    )


def _has_kb_explanation_intent(text: str) -> bool:
    return any(
        keyword in text
        for keyword in ("历史", "来历", "典故", "文化", "解释", "介绍", "为什么", "知识", "作用", "是什么意思", "是什么", "区别", "怎么看", "含义", "字段")
    )


def _has_recipe_howto_intent(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return False
    for keyword in RECIPE_HOWTO_KEYWORDS:
        if keyword not in normalized:
            continue
        subject = normalized.replace(keyword, "").strip(" ?？。！，,")
        if not subject or subject in LOW_INFO_SUBJECTS:
            return False
        return True
    return False


def _has_ingredient_amount_intent(text: str) -> bool:
    normalized = text.strip()
    if not normalized or not any(keyword in normalized for keyword in INGREDIENT_AMOUNT_KEYWORDS):
        return False
    if _has_text2sql_table_subject(normalized):
        return False
    if "里" in normalized or "中" in normalized:
        return True
    return bool(re.search(r".{2,}(要)?放(多少|几[克两勺个片])\S*", normalized))


def _has_text2sql_table_subject(text: str) -> bool:
    return any(
        keyword in text
        for keyword in ("菜谱数量", "道菜谱", "道菜", "每个菜系", "数据库", "统计", "计数", "排名", "趋势", "平均", "占比")
    )


def _has_food_relation_intent(text: str) -> bool:
    base_relation = any(
        keyword in text
        for keyword in (
            "哪些产品",
            "哪些商品",
            "含有",
            "有哪些配料",
            "配料",
            "成分",
            "属于什么类别",
            "属于什么分类",
        )
    )
    if base_relation:
        return True
    if re.search(r"属于.{0,8}(类别|分类)", text):
        return True
    if _has_kb_explanation_intent(text):
        return False
    return any(keyword in text for keyword in ("营养素信息", "主要营养素", "营养标签有哪些", "营养信息有哪些"))
