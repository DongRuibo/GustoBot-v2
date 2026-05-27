"""多模态理解服务测试。"""

import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from app.multimodal import service as multimodal_module
from app.multimodal.service import ImageUnderstandingService


def test_vision_understanding_structures_image_and_reroutes(monkeypatch) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "settings",
        SimpleNamespace(
            vision_base_url="http://vision.local/v1",
            vision_api_key="vision-key",
            vision_model="vision-test",
            vision_timeout_seconds=5,
            vision_max_retries=0,
            ocr_base_url=None,
        ),
    )
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        assert kwargs["headers"]["Authorization"] == "Bearer vision-key"
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "dish_name": "宫保鸡丁",
                                    "possible_ingredients": ["鸡肉", "花生"],
                                    "cooking_state": "成品菜",
                                    "user_intent": "relation",
                                    "description": "一盘带花生的宫保鸡丁",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(multimodal_module.httpx, "post", fake_post)
    result = ImageUnderstandingService().understand(
        "这张图里的菜需要哪些食材？",
        [
            {
                "type": "image",
                "filename": "dish.png",
                "content_type": "image/png",
                "content_base64": base64.b64encode(b"fake-image").decode("ascii"),
            }
        ],
    )

    assert captured["url"] == "http://vision.local/v1/chat/completions"
    image_block = captured["payload"]["messages"][1]["content"][1]
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")
    assert result.dish_name == "宫保鸡丁"
    assert result.possible_ingredients == ["鸡肉", "花生", "辣椒"]
    assert result.reroute_text == "宫保鸡丁需要哪些食材？"
    assert result.metadata["vision_used"] is True


def test_image_understanding_falls_back_to_filename_when_vision_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "settings",
        SimpleNamespace(
            vision_base_url=None,
            vision_model=None,
            ocr_base_url=None,
        ),
    )

    result = ImageUnderstandingService().understand(
        "这道菜需要哪些食材？",
        [{"type": "image", "filename": "gongbao.jpg"}],
    )

    assert result.dish_name == "宫保鸡丁"
    assert "鸡肉" in result.possible_ingredients
    assert result.metadata["fallback_used"] is True


def test_image_understanding_strict_mode_rejects_filename_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "settings",
        SimpleNamespace(
            vision_base_url=None,
            vision_model=None,
            ocr_base_url=None,
            strict_external_stores=True,
        ),
    )

    with pytest.raises(RuntimeError, match="Vision 或 OCR"):
        ImageUnderstandingService().understand(
            "这道菜需要哪些食材？",
            [{"type": "image", "filename": "gongbao.jpg"}],
        )


def test_image_understanding_rejects_image_generation_model(monkeypatch) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "settings",
        SimpleNamespace(
            vision_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            vision_api_key="vision-key",
            vision_model="qwen-image-2.0-pro",
            ocr_base_url=None,
            strict_external_stores=True,
        ),
    )

    with pytest.raises(RuntimeError, match="qwen-image"):
        ImageUnderstandingService().understand(
            "这道菜是什么？",
            [
                {
                    "type": "image",
                    "filename": "dish.png",
                    "content_type": "image/png",
                    "content_base64": base64.b64encode(b"fake-image").decode("ascii"),
                }
            ],
        )


def test_image_understanding_user_question_intent_overrides_vision_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        multimodal_module,
        "settings",
        SimpleNamespace(
            vision_base_url="http://vision.local/v1",
            vision_api_key="vision-key",
            vision_model="vision-test",
            vision_timeout_seconds=5,
            vision_max_retries=0,
            ocr_base_url=None,
        ),
    )

    def fake_post(url, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "dish_name": "宫保鸡丁",
                                    "possible_ingredients": ["鸡肉", "花生"],
                                    "cooking_state": "成品菜",
                                    "user_intent": "knowledge",
                                    "description": "一盘宫保鸡丁",
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(multimodal_module.httpx, "post", fake_post)

    result = ImageUnderstandingService().understand(
        "这道菜需要哪些食材？",
        [
            {
                "type": "image",
                "filename": "dish.png",
                "content_type": "image/png",
                "content_base64": base64.b64encode(b"fake-image").decode("ascii"),
            }
        ],
    )

    assert result.user_intent == "relation"
    assert result.reroute_text == "宫保鸡丁需要哪些食材？"
