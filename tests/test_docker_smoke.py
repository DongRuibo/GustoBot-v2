"""Docker smoke 脚本内部校验测试。"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from scripts import docker_smoke


def test_trace_log_allows_semantic_cache_hits_for_answer_mode() -> None:
    trace_log = _write_trace_log(
        [
            {
                "trace_id": "fresh",
                "event_type": "answer_generated",
                "payload": {"mode": "llm"},
            },
            {
                "trace_id": "cached",
                "event_type": "semantic_cache_hit",
                "payload": {"route_type": "graphrag"},
            },
        ]
    )
    summary: dict[str, object] = {"checks": []}

    try:
        docker_smoke._check_trace_log(trace_log, ["fresh", "cached"], summary, expected_answer_mode="llm")
    finally:
        trace_log.unlink(missing_ok=True)

    trace_check = summary["checks"][0]
    assert trace_check["name"] == "trace_log"
    assert trace_check["payload"]["answer_modes"] == {"fresh": "llm"}
    assert trace_check["payload"]["semantic_cache_hits"] == ["cached"]


def test_trace_log_still_rejects_wrong_fresh_answer_mode() -> None:
    trace_log = _write_trace_log(
        [
            {
                "trace_id": "fresh",
                "event_type": "answer_generated",
                "payload": {"mode": "template"},
            }
        ]
    )

    try:
        with pytest.raises(docker_smoke.SmokeFailure, match="answer_generated mode mismatch"):
            docker_smoke._check_trace_log(trace_log, ["fresh"], {"checks": []}, expected_answer_mode="llm")
    finally:
        trace_log.unlink(missing_ok=True)


def _write_trace_log(events: list[dict[str, object]]) -> Path:
    temp_dir = Path(__file__).resolve().parents[1] / "tmp" / "test_docker_smoke"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / f"{uuid4().hex}.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    return path
