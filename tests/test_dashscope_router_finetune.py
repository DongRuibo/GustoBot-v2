"""DashScope Router 微调管理脚本测试。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from scripts.data import dashscope_router_finetune as ft


def test_validate_dashscope_jsonl_accepts_messages_only(tmp_path: Path) -> None:
    path = tmp_path / "router_dashscope_train.jsonl"
    _write_jsonl(path, [_dashscope_row("kb"), _dashscope_row("text2sql")])

    result = ft.validate_dashscope_jsonl(path)

    assert result["error_count"] == 0
    assert result["row_count"] == 2
    assert result["route_counts"] == {"kb": 1, "text2sql": 1}


def test_validate_dashscope_jsonl_rejects_metadata(tmp_path: Path) -> None:
    path = tmp_path / "router_dashscope_train.jsonl"
    row = _dashscope_row("kb")
    row["metadata"] = {"route_type": "kb"}
    _write_jsonl(path, [row])

    result = ft.validate_dashscope_jsonl(path)

    assert result["error_count"] == 1
    assert "只允许包含 messages" in result["errors_preview"][0]


def test_upload_dry_run_does_not_call_httpx_and_writes_no_api_key(tmp_path: Path, monkeypatch) -> None:
    train, dev, test = _write_dataset_triplet(tmp_path)
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(ft, "resolve_project_path", lambda value, label: Path(value).resolve())
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key")

    def fail_post(*args, **kwargs):  # pragma: no cover - 只要被调用就说明 dry-run 失效。
        raise AssertionError("dry-run 不应调用 DashScope API")

    monkeypatch.setattr(ft.httpx, "post", fail_post)
    args = ft._build_parser().parse_args(
        [
            "--state",
            str(state_path),
            "--report",
            str(report_path),
            "upload",
            "--train",
            str(train),
            "--dev",
            str(dev),
            "--test",
            str(test),
            "--dry-run",
        ]
    )

    result = ft._dispatch(args)

    assert result["status"] == "dry_run"
    assert result["planned_requests"]["train"]["url"].endswith("/files")
    assert "secret-key" not in state_path.read_text(encoding="utf-8")
    assert "secret-key" not in report_path.read_text(encoding="utf-8")


def test_upload_submit_uses_file_api_and_saves_file_ids(tmp_path: Path, monkeypatch) -> None:
    train, dev, test = _write_dataset_triplet(tmp_path)
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(ft, "resolve_project_path", lambda value, label: Path(value).resolve())
    monkeypatch.setenv("DASHSCOPE_API_KEY", "secret-key")
    calls: list[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        file_name = kwargs["files"]["files"][0]
        suffix = "train" if "train" in file_name else "dev"
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"output": {"uploaded_files": [{"file_id": f"file-{suffix}"}]}},
        )

    monkeypatch.setattr(ft.httpx, "post", fake_post)
    args = ft._build_parser().parse_args(
        [
            "--state",
            str(state_path),
            "--report",
            str(report_path),
            "upload",
            "--train",
            str(train),
            "--dev",
            str(dev),
            "--test",
            str(test),
            "--submit",
        ]
    )

    result = ft._dispatch(args)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert result["status"] == "ok"
    assert len(calls) == 2
    assert all(call["url"].endswith("/files") for call in calls)
    assert all(call["headers"]["Authorization"] == "Bearer secret-key" for call in calls)
    assert all("Content-Type" not in call["headers"] for call in calls)
    assert calls[0]["data"]["purpose"] == "fine-tune"
    assert state["training_file_id"] == "file-train"
    assert state["validation_file_id"] == "file-dev"
    assert "secret-key" not in report_path.read_text(encoding="utf-8")


def test_create_dry_run_uses_fine_tune_body_from_state(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    state_path.write_text(
        json.dumps({"training_file_id": "file-train", "validation_file_id": "file-dev"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ft, "resolve_project_path", lambda value, label: Path(value).resolve())

    args = ft._build_parser().parse_args(
        [
            "--state",
            str(state_path),
            "--report",
            str(report_path),
            "create",
            "--dry-run",
        ]
    )

    result = ft._dispatch(args)
    body = result["planned_request"]["json"]

    assert result["status"] == "dry_run"
    assert result["planned_request"]["url"].endswith("/fine-tunes")
    assert body["model"] == "qwen3-8b"
    assert body["training_type"] == "sft"
    assert body["training_file_ids"] == ["file-train"]
    assert body["validation_file_ids"] == ["file-dev"]
    assert body["hyper_parameters"] == {
        "n_epochs": 3,
        "batch_size": 16,
        "max_length": 8192,
        "learning_rate": "1.6e-5",
    }


def test_status_logs_and_deploy_dry_run_requests(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    state_path.write_text(
        json.dumps({"job_id": "ft-123", "finetuned_output": "qwen3-8b-ft-router"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ft, "resolve_project_path", lambda value, label: Path(value).resolve())

    parser = ft._build_parser()
    status = ft._dispatch(parser.parse_args(["--state", str(state_path), "--report", str(report_path), "status"]))
    logs = ft._dispatch(parser.parse_args(["--state", str(state_path), "--report", str(report_path), "logs"]))
    deploy = ft._dispatch(parser.parse_args(["--state", str(state_path), "--report", str(report_path), "deploy"]))

    assert status["planned_request"]["url"].endswith("/fine-tunes/ft-123")
    assert logs["planned_request"]["url"].endswith("/fine-tunes/ft-123/logs")
    assert logs["planned_request"]["params"] == {"offset": 0, "line": 50}
    assert deploy["planned_request"]["url"].endswith("/deployments")
    assert deploy["planned_request"]["json"]["model_name"] == "qwen3-8b-ft-router"


def test_deploy_dry_run_supports_mu_parameters(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    report_path = tmp_path / "report.json"
    state_path.write_text(json.dumps({"finetuned_output": "qwen3-8b-ft-router"}), encoding="utf-8")
    monkeypatch.setattr(ft, "resolve_project_path", lambda value, label: Path(value).resolve())

    args = ft._build_parser().parse_args(
        [
            "--state",
            str(state_path),
            "--report",
            str(report_path),
            "deploy",
            "--dry-run",
            "--plan",
            "mu",
            "--deploy-spec",
            "MU1",
            "--capacity",
            "1",
            "--enable-thinking",
            "false",
            "--max-context-length",
            "8192",
            "--rpm-limit",
            "30",
            "--tpm-limit",
            "30000",
            "--suffix",
            "router",
        ]
    )

    result = ft._dispatch(args)
    body = result["planned_request"]["json"]

    assert body["plan"] == "mu"
    assert body["deploy_spec"] == "MU1"
    assert body["enable_thinking"] is False
    assert body["max_context_length"] == 8192
    assert body["rpm_limit"] == 30
    assert body["tpm_limit"] == 30000
    assert body["suffix"] == "router"


def test_report_sanitizes_dashscope_identity_fields() -> None:
    report = ft._sanitize_for_report(
        {
            "operation": "status",
            "response": {
                "job_id": "ft-123",
                "status": "SUCCEEDED",
                "finetuned_output": "qwen3-8b-ft-router",
                "workspace_id": "workspace-secret",
                "user_identity": "user-secret",
                "creator": "creator-secret",
                "modifier": "modifier-secret",
            },
        }
    )

    assert report["response"] == {
        "job_id": "ft-123",
        "status": "SUCCEEDED",
        "finetuned_output": "qwen3-8b-ft-router",
    }
    assert "workspace-secret" not in json.dumps(report, ensure_ascii=False)
    assert "user-secret" not in json.dumps(report, ensure_ascii=False)


def _write_dataset_triplet(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "router_dashscope_train.jsonl"
    dev = tmp_path / "router_dashscope_dev.jsonl"
    test = tmp_path / "router_dashscope_test.jsonl"
    _write_jsonl(train, [_dashscope_row("kb")])
    _write_jsonl(dev, [_dashscope_row("graphrag")])
    _write_jsonl(test, [_dashscope_row("text2sql")])
    return train, dev, test


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _dashscope_row(route_type: str) -> dict:
    payload = {
        "route_type": route_type,
        "confidence": 0.88,
        "reason": "Router SFT 样本。",
        "slots": {"sample_category": route_type},
        "need_clarification": route_type == "clarify",
    }
    return {
        "messages": [
            {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
            {"role": "user", "content": "{\"question\":\"蛋白质有什么作用？\"}"},
            {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
        ]
    }
