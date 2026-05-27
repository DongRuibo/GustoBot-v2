"""图片理解服务模块。

该层只负责把图片、OCR 文本和 Vision LLM 结果整理成结构化文本，并生成
reroute_text 重新进入 Router；最终回答仍由 KB、GraphRAG 或 Text2SQL 链路完成。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings


@dataclass(slots=True)
class ImageUnderstandingResult:
    dish_name: str | None
    possible_ingredients: list[str]
    cooking_state: str | None
    user_intent: str
    structured_text: str
    reroute_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageUnderstandingService:
    def understand(self, user_question: str, image_attachments: list[dict[str, Any]]) -> ImageUnderstandingResult:
        first_image = _resolve_upload_attachment(image_attachments[0]) if image_attachments else {}
        raw_text = (first_image.get("text") or "").strip()
        filename = (first_image.get("filename") or "").lower()

        ocr_text, ocr_metadata = self._try_ocr(first_image)
        vision_payload, vision_metadata = self._try_vision(user_question, first_image, raw_text, ocr_text)
        strict_mode = bool(getattr(settings, "strict_external_stores", False))
        if strict_mode and not vision_metadata.get("success") and not ocr_metadata.get("success"):
            raise RuntimeError(
                "生产环境图片理解必须成功调用 Vision 或 OCR，不能退回文件名推断。"
                f"Vision 错误：{vision_metadata.get('error') or 'none'}；"
                f"OCR 错误：{ocr_metadata.get('error') or 'none'}；"
                f"Vision 模型：{vision_metadata.get('model') or 'none'}。"
            )
        vision_text = self._vision_text(vision_payload)
        combined_text = "\n".join(part for part in (raw_text, ocr_text, vision_text) if part).strip()
        fallback_filename = "" if strict_mode and not vision_metadata.get("success") else filename

        dish_name = _first_text(vision_payload.get("dish_name"), self._infer_dish_name(combined_text, fallback_filename))
        possible_ingredients = _merge_unique(
            _as_text_list(vision_payload.get("possible_ingredients")),
            self._infer_ingredients(dish_name, combined_text),
        )
        cooking_state = _first_text(vision_payload.get("cooking_state"), "已识别菜品" if dish_name else "未知")
        # 用户意图以原始问题为准；Vision 只负责识别图像内容，避免模型把“食材/做法”误改成历史文化查询。
        user_intent = self._normalize_intent(self._infer_user_intent(user_question))
        reroute_text = self._build_reroute_text(dish_name, possible_ingredients, user_intent, user_question)
        structured_text = self._build_structured_text(
            dish_name=dish_name,
            possible_ingredients=possible_ingredients,
            cooking_state=cooking_state,
            user_intent=user_intent,
            description=_first_text(vision_payload.get("description"), vision_payload.get("scene_description")),
            ocr_text=ocr_text,
        )
        return ImageUnderstandingResult(
            dish_name=dish_name,
            possible_ingredients=possible_ingredients,
            cooking_state=cooking_state,
            user_intent=user_intent,
            structured_text=structured_text,
            reroute_text=reroute_text,
            metadata={
                "filename": first_image.get("filename"),
                "content_type": first_image.get("content_type"),
                "used_attachment_text": bool(raw_text),
                "vision_used": bool(vision_metadata.get("success")),
                "vision_model": vision_metadata.get("model"),
                "vision_error": vision_metadata.get("error"),
                "ocr_used": bool(ocr_metadata.get("success")),
                "ocr_error": ocr_metadata.get("error"),
                "fallback_used": not vision_metadata.get("success"),
            },
        )

    def _try_vision(
        self,
        user_question: str,
        attachment: dict[str, Any],
        raw_text: str,
        ocr_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        base_url = getattr(settings, "vision_base_url", None)
        model = getattr(settings, "vision_model", None)
        if not base_url or not model:
            return {}, {"success": False, "error": "vision_not_configured", "model": model}
        if _is_image_generation_model(model):
            return {}, {
                "success": False,
                "error": "vision_model_not_for_understanding: qwen-image 系列用于图片生成/编辑，请改用 qwen3-vl-plus、qwen-vl-plus 或 qwen-vl-max。",
                "model": model,
            }
        image_block = _image_content_block(attachment)
        if image_block is None:
            return {}, {"success": False, "error": "image_payload_missing", "model": model}

        prompt = (
            "请识别这张餐饮/菜品图片，并只返回 JSON。字段包括："
            "dish_name 字符串或 null，possible_ingredients 字符串数组，"
            "cooking_state 字符串，user_intent 只能是 knowledge/relation/analysis/general，"
            "description 简短中文描述，ocr_text 图片中文字。"
            f"\n用户问题：{user_question}"
        )
        if raw_text or ocr_text:
            prompt += f"\n已有文本线索：{raw_text}\nOCR线索：{ocr_text}"

        request_payload = {
            "model": model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": "你是 GustoBot-v2 的图片输入理解层，只做结构化识别，不直接回答用户问题。"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_block,
                    ],
                },
            ],
        }
        try:
            response = _post_json_with_retries(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=_headers(getattr(settings, "vision_api_key", None)),
                payload=request_payload,
                timeout=float(getattr(settings, "vision_timeout_seconds", 30)),
                max_retries=int(getattr(settings, "vision_max_retries", 1)),
            )
            content = _chat_content(response.json())
            parsed = _parse_json_object(content)
            return parsed, {"success": True, "model": model}
        except Exception as exc:
            return {}, {"success": False, "error": str(exc)[:300], "model": model}

    def _try_ocr(self, attachment: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        base_url = getattr(settings, "ocr_base_url", None)
        image_base64 = attachment.get("content_base64")
        if not base_url:
            return "", {"success": False, "error": "ocr_not_configured"}
        if not image_base64:
            return "", {"success": False, "error": "image_payload_missing"}

        payload = {
            "image_base64": image_base64,
            "filename": attachment.get("filename"),
            "content_type": attachment.get("content_type"),
        }
        try:
            response = _post_json_with_retries(
                base_url.rstrip("/"),
                headers=_headers(getattr(settings, "ocr_api_key", None)),
                payload=payload,
                timeout=float(getattr(settings, "ocr_timeout_seconds", 20)),
                max_retries=int(getattr(settings, "ocr_max_retries", 1)),
            )
            data = response.json()
            text = _nested_text(data, ("text", "ocr_text", "recognized_text"))
            if not text and isinstance(data.get("output"), dict):
                text = _nested_text(data["output"], ("text", "ocr_text", "recognized_text"))
            return text.strip(), {"success": bool(text)}
        except Exception as exc:
            return "", {"success": False, "error": str(exc)[:300]}

    def _vision_text(self, payload: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in ("dish_name", "cooking_state", "description", "scene_description", "ocr_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
        parts.extend(_as_text_list(payload.get("possible_ingredients")))
        return "\n".join(parts)

    def _build_structured_text(
        self,
        *,
        dish_name: str | None,
        possible_ingredients: list[str],
        cooking_state: str | None,
        user_intent: str,
        description: str | None,
        ocr_text: str,
    ) -> str:
        parts = [
            f"图片识别结构化结果：菜品名称={dish_name or '未知'}",
            f"可能食材={'、'.join(possible_ingredients) if possible_ingredients else '未知'}",
            f"烹饪状态={cooking_state or '未知'}",
            f"用户意图={user_intent}",
        ]
        if description:
            parts.append(f"图像描述={description}")
        if ocr_text:
            parts.append(f"OCR文本={ocr_text}")
        return "；".join(parts) + "。"

    def _infer_dish_name(self, raw_text: str, filename: str) -> str | None:
        # 附件 text 字段可以承载外部 OCR/Vision 结果；文件名推断只用于本地测试和降级。
        candidates = {
            "宫保鸡丁": ("宫保鸡丁", "宫爆鸡丁", "gongbao", "kungpao", "kung pao"),
            "麻婆豆腐": ("麻婆豆腐", "mapo"),
            "白菜炖豆腐": ("白菜炖豆腐", "cabbage_tofu", "baicai"),
            "佛跳墙": ("佛跳墙", "fotiaoqiang"),
        }
        haystack = f"{raw_text} {filename}"
        for dish_name, aliases in candidates.items():
            if any(alias.lower() in haystack.lower() for alias in aliases):
                return dish_name
        return None

    def _infer_ingredients(self, dish_name: str | None, raw_text: str) -> list[str]:
        ingredient_map = {
            "宫保鸡丁": ["鸡肉", "花生", "辣椒"],
            "麻婆豆腐": ["豆腐", "辣椒", "花椒"],
            "白菜炖豆腐": ["白菜", "豆腐"],
            "佛跳墙": ["海参", "鲍鱼", "高汤"],
        }
        if dish_name in ingredient_map:
            return ingredient_map[dish_name]
        known = ["鸡肉", "花生", "辣椒", "豆腐", "白菜", "海参", "鲍鱼"]
        return [ingredient for ingredient in known if ingredient in raw_text]

    def _infer_user_intent(self, user_question: str) -> str:
        if any(keyword in user_question for keyword in ("历史", "文化", "典故", "介绍")):
            return "knowledge"
        if any(keyword in user_question for keyword in ("统计", "数量", "排名", "趋势")):
            return "analysis"
        if any(keyword in user_question for keyword in ("食材", "步骤", "怎么做", "能做什么菜")):
            return "relation"
        return "relation"

    def _normalize_intent(self, value: str | None) -> str:
        if not value:
            return "relation"
        lowered = value.strip().lower()
        if lowered in {"knowledge", "relation", "analysis", "general"}:
            return lowered
        if any(keyword in value for keyword in ("历史", "文化", "知识", "介绍")):
            return "knowledge"
        if any(keyword in value for keyword in ("统计", "分析", "趋势", "排名")):
            return "analysis"
        return "relation"

    def _build_reroute_text(
        self,
        dish_name: str | None,
        possible_ingredients: list[str],
        user_intent: str,
        user_question: str,
    ) -> str:
        subject = dish_name or ("、".join(possible_ingredients) if possible_ingredients else "图片中的菜")
        if user_intent == "knowledge":
            return f"介绍一下{subject}的历史和文化"
        if user_intent == "analysis":
            return user_question
        if "步骤" in user_question or "怎么做" in user_question:
            return f"{subject}有哪些步骤？"
        return f"{subject}需要哪些食材？"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _resolve_upload_attachment(attachment: dict[str, Any]) -> dict[str, Any]:
    uri = attachment.get("uri")
    if not isinstance(uri, str) or not uri.startswith("upload://"):
        return attachment
    try:
        from app.uploads.service import get_upload_service

        return get_upload_service().resolve_attachment(attachment)
    except Exception:
        return attachment


def _post_json_with_retries(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(max(0, max_retries) + 1):
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=timeout)
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
    raise last_exc or RuntimeError("multimodal request failed")


def _image_content_block(attachment: dict[str, Any]) -> dict[str, Any] | None:
    image_url = attachment.get("image_url") or attachment.get("url")
    if image_url:
        return {"type": "image_url", "image_url": {"url": image_url}}
    content_base64 = attachment.get("content_base64")
    if not content_base64:
        return None
    if str(content_base64).startswith("data:"):
        data_url = str(content_base64)
    else:
        content_type = attachment.get("content_type") or "image/jpeg"
        data_url = f"data:{content_type};base64,{content_base64}"
    return {"type": "image_url", "image_url": {"url": data_url}}


def _chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return str(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def _nested_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in re.split(r"[、,，;；]", value) if item.strip()]
    return []


def _merge_unique(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return merged


def _is_image_generation_model(model: str | None) -> bool:
    # 图片理解链路需要返回文本/JSON；qwen-image 系列是生成/编辑模型。
    return bool(model and model.strip().lower().startswith("qwen-image"))


_image_understanding_service = ImageUnderstandingService()


def get_image_understanding_service() -> ImageUnderstandingService:
    return _image_understanding_service
