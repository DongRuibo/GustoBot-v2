"""P0 失败注入脚本测试。"""

from __future__ import annotations

import json

from scripts import p0_failure_injection as failure_injection


def test_failure_injection_selects_named_cases() -> None:
    cases = failure_injection._select_cases(["missing_model_keys", "bad_redis_runtime"])

    assert [case.name for case in cases] == ["missing_model_keys", "bad_redis_runtime"]


def test_missing_model_keys_case_clears_all_supported_key_envs() -> None:
    case = failure_injection._select_cases(["missing_model_keys"])[0]

    for name in failure_injection.API_KEY_ENV_NAMES:
        assert case.overrides[name] == ""


def test_case_passed_requires_failed_status_and_expected_stage() -> None:
    case = failure_injection._select_cases(["bad_postgres_runtime"])[0]

    assert failure_injection._case_passed(case, 1, {"status": "failed", "stage": "runtime"}) is True
    assert failure_injection._case_passed(case, 0, {"status": "ok"}) is False
    assert failure_injection._case_passed(case, 1, {"status": "failed", "stage": "config"}) is False


def test_parse_readiness_output_handles_json() -> None:
    payload = {"status": "failed", "stage": "config", "issues": ["缺少 GUSTOBOT_VISION_API_KEY。"]}

    parsed = failure_injection._parse_readiness_output(json.dumps(payload, ensure_ascii=False))

    assert parsed == payload
