"""管理 DashScope Router SFT 云端调优任务。

所有会产生远端副作用的子命令默认 dry-run；只有显式传入 --submit 才会调用
DashScope API。状态和报告只保存 file_id/job_id/deployment_id 等非敏感信息。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import REPORTS_DIR, SFT_DIR, load_jsonl, resolve_project_path  # noqa: E402


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_MODEL = "qwen3-8b"
DEFAULT_TRAINING_TYPE = "sft"
DEFAULT_STATE = SFT_DIR / "router_training" / "dashscope_finetune_state.json"
DEFAULT_REPORT = REPORTS_DIR / "dashscope_router_finetune.json"
DEFAULT_TRAIN = SFT_DIR / "router_training" / "router_dashscope_train.jsonl"
DEFAULT_DEV = SFT_DIR / "router_training" / "router_dashscope_dev.jsonl"
DEFAULT_TEST = SFT_DIR / "router_training" / "router_dashscope_test.jsonl"
DEFAULT_TIMEOUT = 60.0
STATE_SECRET_KEYS = {"api_key", "authorization", "headers"}
SENSITIVE_RESPONSE_KEYS = {"workspace_id", "user_identity", "creator", "modifier", "group"}
REPORT_RESPONSE_KEYS = {
    "job_id",
    "job_name",
    "status",
    "finetuned_output",
    "model",
    "base_model",
    "model_name",
    "training_file_ids",
    "validation_file_ids",
    "hyper_parameters",
    "training_type",
    "create_time",
    "end_time",
    "deployment_id",
    "deployment_status",
    "deployed_model",
    "id",
    "name",
    "charge_type",
    "priority",
    "max_output_cnt",
    "output_cnt",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = _dispatch(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage DashScope Router fine-tuning workflow.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="校验 DashScope JSONL 训练文件。")
    _add_dataset_args(validate)

    upload = subparsers.add_parser("upload", help="上传 train/dev 训练文件。")
    _add_dataset_args(upload)
    _add_submit_arg(upload)

    create = subparsers.add_parser("create", help="创建 DashScope fine-tune 任务。")
    _add_submit_arg(create)
    create.add_argument("--model", default=DEFAULT_MODEL)
    create.add_argument("--training-type", default=DEFAULT_TRAINING_TYPE)
    create.add_argument("--training-file-id", default=None)
    create.add_argument("--validation-file-id", default=None)
    create.add_argument("--job-name", default=None)
    create.add_argument("--model-name", default=None)
    create.add_argument("--n-epochs", type=int, default=3)
    create.add_argument("--batch-size", type=int, default=16)
    create.add_argument("--max-length", type=int, default=8192)
    create.add_argument("--learning-rate", default="1.6e-5")

    status = subparsers.add_parser("status", help="查询 fine-tune 任务状态。")
    _add_submit_arg(status)
    status.add_argument("--job-id", default=None)

    logs = subparsers.add_parser("logs", help="查询 fine-tune 任务日志。")
    _add_submit_arg(logs)
    logs.add_argument("--job-id", default=None)
    logs.add_argument("--offset", type=int, default=0)
    logs.add_argument("--line", type=int, default=50)

    deploy = subparsers.add_parser("deploy", help="部署调优后的模型。")
    _add_submit_arg(deploy)
    deploy.add_argument("--model-name", default=None)
    deploy.add_argument("--deployment-name", default="gustobot-router-sft")
    deploy.add_argument("--plan", default="lora")
    deploy.add_argument("--capacity", type=int, default=1)
    deploy.add_argument("--deploy-spec", default=None)
    deploy.add_argument("--enable-thinking", choices=("true", "false"), default=None)
    deploy.add_argument("--max-context-length", type=int, default=None)
    deploy.add_argument("--rpm-limit", type=int, default=None)
    deploy.add_argument("--tpm-limit", type=int, default=None)
    deploy.add_argument("--suffix", default=None)
    deploy.add_argument("--display-name", default=None)

    return parser


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--test", default=str(DEFAULT_TEST))


def _add_submit_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submit", action="store_true", help="真正调用 DashScope API；默认只 dry-run。")
    parser.add_argument("--dry-run", action="store_true", help="显式 dry-run；等价于不传 --submit。")


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    state_path = resolve_project_path(args.state, label="state")
    report_path = resolve_project_path(args.report, label="report")
    state = _load_state(state_path)
    submit = bool(getattr(args, "submit", False))
    if getattr(args, "dry_run", False):
        submit = False

    if args.command == "validate":
        result = cmd_validate(args, state=state)
    elif args.command == "upload":
        result = cmd_upload(args, state=state, submit=submit)
    elif args.command == "create":
        result = cmd_create(args, state=state, submit=submit)
    elif args.command == "status":
        result = cmd_status(args, state=state, submit=submit)
    elif args.command == "logs":
        result = cmd_logs(args, state=state, submit=submit)
    elif args.command == "deploy":
        result = cmd_deploy(args, state=state, submit=submit)
    else:  # pragma: no cover - argparse 已拦截。
        raise SystemExit(f"未知子命令：{args.command}")

    result = _with_common_fields(result, args=args, submit=submit)
    updated_state = _merge_state(state, result.get("state_updates", {}))
    _write_json(state_path, _sanitize_for_state(updated_state))
    _write_json(report_path, _sanitize_for_report(result))
    return result


def cmd_validate(args: argparse.Namespace, *, state: dict[str, Any]) -> dict[str, Any]:
    paths = _dataset_paths(args)
    validation = {split: validate_dashscope_jsonl(path) for split, path in paths.items()}
    return {
        "status": "ok" if all(item["error_count"] == 0 for item in validation.values()) else "invalid",
        "operation": "validate",
        "validation": validation,
        "state_updates": {"datasets": {split: str(path) for split, path in paths.items()}},
    }


def cmd_upload(args: argparse.Namespace, *, state: dict[str, Any], submit: bool) -> dict[str, Any]:
    paths = _dataset_paths(args)
    validation = {split: validate_dashscope_jsonl(path) for split, path in paths.items()}
    if any(item["error_count"] for item in validation.values()):
        return {"status": "invalid", "operation": "upload", "validation": validation, "state_updates": {}}

    planned = {
        split: _upload_request_preview(args.base_url, path, split)
        for split, path in paths.items()
        if split in {"train", "dev"}
    }
    if not submit:
        return {
            "status": "dry_run",
            "operation": "upload",
            "validation": validation,
            "planned_requests": planned,
            "state_updates": {"datasets": {split: str(path) for split, path in paths.items()}},
        }

    api_key = _require_api_key()
    uploaded: dict[str, Any] = {}
    for split in ("train", "dev"):
        uploaded[split] = upload_file(
            base_url=args.base_url,
            api_key=api_key,
            path=paths[split],
            split=split,
            timeout=args.timeout,
        )
    return {
        "status": "ok",
        "operation": "upload",
        "validation": validation,
        "uploaded": uploaded,
        "state_updates": {
            "datasets": {split: str(path) for split, path in paths.items()},
            "training_file_id": uploaded["train"].get("file_id"),
            "validation_file_id": uploaded["dev"].get("file_id"),
        },
    }


def cmd_create(args: argparse.Namespace, *, state: dict[str, Any], submit: bool) -> dict[str, Any]:
    training_file_id = args.training_file_id or state.get("training_file_id")
    validation_file_id = args.validation_file_id or state.get("validation_file_id")
    body = fine_tune_body(
        model=args.model,
        training_type=args.training_type,
        training_file_id=training_file_id or "<training_file_id>",
        validation_file_id=validation_file_id,
        job_name=args.job_name,
        model_name=args.model_name,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )
    if not submit:
        return {
            "status": "dry_run",
            "operation": "create",
            "planned_request": {"method": "POST", "url": _url(args.base_url, "/fine-tunes"), "json": body},
            "state_updates": {},
        }
    if not training_file_id:
        raise SystemExit("create --submit 需要 training_file_id；请先 upload --submit 或显式传 --training-file-id。")

    response = http_post_json(_url(args.base_url, "/fine-tunes"), body, api_key=_require_api_key(), timeout=args.timeout)
    output = _response_output(response)
    return {
        "status": "ok",
        "operation": "create",
        "response": output,
        "state_updates": {
            "job_id": output.get("job_id"),
            "job_status": output.get("status"),
            "finetuned_output": output.get("finetuned_output"),
            "base_model": output.get("base_model") or output.get("model"),
        },
    }


def cmd_status(args: argparse.Namespace, *, state: dict[str, Any], submit: bool) -> dict[str, Any]:
    job_id = args.job_id or state.get("job_id")
    url = _url(args.base_url, f"/fine-tunes/{job_id or '<job_id>'}")
    if not submit:
        return {"status": "dry_run", "operation": "status", "planned_request": {"method": "GET", "url": url}, "state_updates": {}}
    if not job_id:
        raise SystemExit("status --submit 需要 job_id；请先 create --submit 或显式传 --job-id。")
    response = http_get_json(url, api_key=_require_api_key(), timeout=args.timeout)
    output = _response_output(response)
    return {
        "status": "ok",
        "operation": "status",
        "response": output,
        "state_updates": {
            "job_id": output.get("job_id") or job_id,
            "job_status": output.get("status"),
            "finetuned_output": output.get("finetuned_output") or state.get("finetuned_output"),
        },
    }


def cmd_logs(args: argparse.Namespace, *, state: dict[str, Any], submit: bool) -> dict[str, Any]:
    job_id = args.job_id or state.get("job_id")
    url = _url(args.base_url, f"/fine-tunes/{job_id or '<job_id>'}/logs")
    params = {"offset": args.offset, "line": args.line}
    if not submit:
        return {
            "status": "dry_run",
            "operation": "logs",
            "planned_request": {"method": "GET", "url": url, "params": params},
            "state_updates": {},
        }
    if not job_id:
        raise SystemExit("logs --submit 需要 job_id；请先 create --submit 或显式传 --job-id。")
    response = http_get_json(url, api_key=_require_api_key(), params=params, timeout=args.timeout)
    return {"status": "ok", "operation": "logs", "response": _response_output(response), "state_updates": {}}


def cmd_deploy(args: argparse.Namespace, *, state: dict[str, Any], submit: bool) -> dict[str, Any]:
    model_name = args.model_name or state.get("finetuned_output")
    body = {
        "model_name": model_name or "<finetuned_output>",
        "plan": args.plan,
        "capacity": args.capacity,
        "name": args.deployment_name,
    }
    if args.deploy_spec:
        body["deploy_spec"] = args.deploy_spec
    if args.enable_thinking is not None:
        body["enable_thinking"] = args.enable_thinking == "true"
    if args.max_context_length is not None:
        body["max_context_length"] = args.max_context_length
    if args.rpm_limit is not None:
        body["rpm_limit"] = args.rpm_limit
    if args.tpm_limit is not None:
        body["tpm_limit"] = args.tpm_limit
    if args.suffix:
        body["suffix"] = args.suffix
    if args.display_name:
        body["diasplay_name"] = args.display_name
    if not submit:
        return {
            "status": "dry_run",
            "operation": "deploy",
            "planned_request": {"method": "POST", "url": _url(args.base_url, "/deployments"), "json": body},
            "state_updates": {},
        }
    if not model_name:
        raise SystemExit("deploy --submit 需要 model_name；请先 status --submit 获取 finetuned_output 或显式传 --model-name。")
    response = http_post_json(_url(args.base_url, "/deployments"), body, api_key=_require_api_key(), timeout=args.timeout)
    output = _response_output(response)
    return {
        "status": "ok",
        "operation": "deploy",
        "response": output,
        "state_updates": {
            "deployment_id": _deployment_id(output),
            "deployment_status": output.get("status"),
            "deployed_model": output.get("deployed_model") or model_name,
            "deployment_model_name": output.get("model_name") or model_name,
        },
    }


def validate_dashscope_jsonl(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    errors: list[str] = []
    route_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        if set(row) != {"messages"}:
            errors.append(f"line {index}: 只允许包含 messages 字段")
            continue
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"line {index}: messages 至少包含 system/user/assistant")
            continue
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles[-1:] != ["assistant"] or "user" not in roles or "system" not in roles:
            errors.append(f"line {index}: messages role 不完整")
            continue
        content = messages[-1].get("content")
        try:
            payload = json.loads(content)
        except Exception:
            errors.append(f"line {index}: assistant content 不是合法 JSON")
            continue
        missing = {"route_type", "confidence", "reason", "slots", "need_clarification"} - set(payload)
        if missing:
            errors.append(f"line {index}: assistant JSON 缺少 {sorted(missing)}")
            continue
        route = str(payload.get("route_type") or "")
        route_counts[route] = route_counts.get(route, 0) + 1
    return {
        "path": str(path),
        "row_count": len(rows),
        "error_count": len(errors),
        "errors_preview": errors[:20],
        "route_counts": dict(sorted(route_counts.items())),
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
    }


def upload_file(*, base_url: str, api_key: str, path: Path, split: str, timeout: float) -> dict[str, Any]:
    with path.open("rb") as file:
        response = httpx.post(
            _url(base_url, "/files"),
            headers=_headers(api_key, content_type=None),
            files={"files": (path.name, file, "application/jsonl")},
            data={"purpose": "fine-tune", "descriptions": f"GustoBot router {split} fine-tune data"},
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()
    file_id = _extract_file_id(payload)
    return {"file_id": file_id, "name": path.name, "raw_output": _response_data_or_output(payload)}


def fine_tune_body(
    *,
    model: str,
    training_type: str,
    training_file_id: str,
    validation_file_id: str | None,
    job_name: str | None,
    model_name: str | None,
    n_epochs: int,
    batch_size: int,
    max_length: int,
    learning_rate: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "training_file_ids": [training_file_id],
        "hyper_parameters": {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
        },
        "training_type": training_type,
    }
    if validation_file_id:
        body["validation_file_ids"] = [validation_file_id]
    if job_name:
        body["job_name"] = job_name
    if model_name:
        body["model_name"] = model_name
    return body


def http_post_json(url: str, body: dict[str, Any], *, api_key: str, timeout: float) -> dict[str, Any]:
    response = httpx.post(url, headers=_headers(api_key, content_type="application/json"), json=body, timeout=timeout)
    _raise_for_status(response)
    return dict(response.json())


def http_get_json(
    url: str,
    *,
    api_key: str,
    timeout: float,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = httpx.get(url, headers=_headers(api_key, content_type="application/json"), params=params, timeout=timeout)
    _raise_for_status(response)
    return dict(response.json())


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text.strip()
        if len(detail) > 1000:
            detail = detail[:1000] + "..."
        raise SystemExit(f"DashScope API 请求失败：HTTP {response.status_code} {detail}") from exc


def _dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "train": resolve_project_path(args.train, label="train"),
        "dev": resolve_project_path(args.dev, label="dev"),
        "test": resolve_project_path(args.test, label="test"),
    }


def _upload_request_preview(base_url: str, path: Path, split: str) -> dict[str, Any]:
    return {
        "method": "POST",
        "url": _url(base_url, "/files"),
        "form": {
            "files": path.name,
            "purpose": "fine-tune",
            "descriptions": f"GustoBot router {split} fine-tune data",
        },
    }


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _headers(api_key: str, *, content_type: str | None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _require_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("GUSTOBOT_ROUTER_LLM_API_KEY")
    if not api_key:
        raise SystemExit("需要设置 DASHSCOPE_API_KEY 或 GUSTOBOT_ROUTER_LLM_API_KEY。")
    return api_key


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_state(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    merged["updated_at"] = _now()
    return merged


def _with_common_fields(result: dict[str, Any], *, args: argparse.Namespace, submit: bool) -> dict[str, Any]:
    return {
        "generated_at": _now(),
        "dry_run": not submit,
        "base_url": args.base_url,
        **result,
    }


def _sanitize_for_state(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _sanitize_for_state(value)
            for key, value in payload.items()
            if key.lower() not in STATE_SECRET_KEYS
            and key.lower() not in SENSITIVE_RESPONSE_KEYS
            and "api_key" not in key.lower()
        }
    if isinstance(payload, list):
        return [_sanitize_for_state(item) for item in payload]
    return payload


def _sanitize_for_report(payload: dict[str, Any]) -> dict[str, Any]:
    report = _sanitize_for_state(payload)
    response = report.get("response")
    if isinstance(response, dict):
        report["response"] = _report_response_summary(response, operation=str(report.get("operation") or ""))
    return report


def _report_response_summary(response: dict[str, Any], *, operation: str) -> dict[str, Any]:
    if operation == "logs":
        return {
            "total": response.get("total", 0),
            "logs": response.get("logs", []),
        }
    return {key: value for key, value in response.items() if key in REPORT_RESPONSE_KEYS}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _response_output(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("output")
    if isinstance(value, dict):
        return value
    return payload


def _response_data_or_output(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("data", "output"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _extract_file_id(payload: dict[str, Any]) -> str | None:
    data = _response_data_or_output(payload)
    uploaded = data.get("uploaded_files")
    if isinstance(uploaded, list) and uploaded:
        return uploaded[0].get("file_id") or uploaded[0].get("id")
    return data.get("file_id") or data.get("id")


def _deployment_id(output: dict[str, Any]) -> str | None:
    return output.get("deployment_id") or output.get("id") or output.get("deployed_model")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
