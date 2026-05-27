"""导出 Router SFT 为常见训练框架格式。

输入使用 GustoBot-v2 内部 Router SFT JSONL，输出 OpenAI messages、LLaMA-Factory
Alpaca、ms-swift messages 和 DashScope fine-tune 四种格式。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import SFT_DIR, clean_text, load_jsonl, resolve_project_path, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_DIR = SFT_DIR / "router_training"
REQUIRED_ROUTE_OUTPUT_KEYS = {"route_type", "confidence", "reason", "slots", "need_clarification"}
SPLITS = ("train", "dev", "test")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    input_paths = {
        "train": resolve_project_path(args.router_train, label="router-train"),
        "dev": resolve_project_path(args.router_dev, label="router-dev"),
        "test": resolve_project_path(args.router_test, label="router-test"),
    }
    output_dir = resolve_project_path(args.output_dir, label="output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {split: load_router_sft(path) for split, path in input_paths.items()}
    outputs = export_router_training_files(rows_by_split, output_dir=output_dir)
    report = build_export_report(rows_by_split, outputs=outputs)
    report_path = output_dir / "manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_llamafactory_dataset_info(output_dir)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "manifest": str(report_path),
                "total": report["total"],
                "route_counts": report["route_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Router SFT to training framework formats.")
    parser.add_argument("--router-train", default=str(SFT_DIR / "router_train.jsonl"))
    parser.add_argument("--router-dev", default=str(SFT_DIR / "router_dev.jsonl"))
    parser.add_argument("--router-test", default=str(SFT_DIR / "router_test.jsonl"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def load_router_sft(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if not rows:
        raise SystemExit(f"Router SFT 文件为空：{path}")
    errors = validate_router_rows(rows)
    if errors:
        preview = "; ".join(errors[:5])
        raise SystemExit(f"Router SFT 格式不符合预期：{preview}")
    return rows


def validate_router_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        messages = row.get("messages")
        output = row.get("output")
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"line {index}: messages 缺失")
            continue
        if not isinstance(output, dict):
            errors.append(f"line {index}: output 缺失")
            continue
        missing = REQUIRED_ROUTE_OUTPUT_KEYS - set(output)
        if missing:
            errors.append(f"line {index}: output 缺少 {sorted(missing)}")
        if not _message_content(messages, "user"):
            errors.append(f"line {index}: user content 缺失")
        if not clean_text(output.get("route_type")):
            errors.append(f"line {index}: route_type 缺失")
    return errors


def export_router_training_files(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        openai_rows = [to_openai_messages(row) for row in rows]
        swift_rows = [to_swift_messages(row) for row in rows]
        dashscope_rows = [to_dashscope_messages(row) for row in rows]
        llamafactory_rows = [to_llamafactory_alpaca(row) for row in rows]

        openai_path = output_dir / f"router_openai_{split}.jsonl"
        swift_path = output_dir / f"router_swift_{split}.jsonl"
        dashscope_path = output_dir / f"router_dashscope_{split}.jsonl"
        llamafactory_path = output_dir / f"router_llamafactory_{split}.json"
        write_jsonl(openai_path, openai_rows)
        write_jsonl(swift_path, swift_rows)
        write_jsonl(dashscope_path, dashscope_rows)
        _write_json_array(llamafactory_path, llamafactory_rows)
        outputs[split] = {
            "openai_messages": str(openai_path),
            "swift_messages": str(swift_path),
            "dashscope_messages": str(dashscope_path),
            "llamafactory_alpaca": str(llamafactory_path),
        }
    return outputs


def to_openai_messages(row: dict[str, Any]) -> dict[str, Any]:
    system = _message_content(row["messages"], "system")
    user = _message_content(row["messages"], "user")
    assistant = _assistant_json(row["output"])
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": _metadata(row),
    }


def to_swift_messages(row: dict[str, Any]) -> dict[str, Any]:
    # ms-swift 可直接消费 messages JSONL；metadata 只用于审计，不参与训练文本。
    return to_openai_messages(row)


def to_dashscope_messages(row: dict[str, Any]) -> dict[str, Any]:
    # 百炼 fine-tune 文件只保留 messages，避免 metadata 等额外字段影响平台校验。
    payload = to_openai_messages(row)
    return {"messages": payload["messages"]}


def to_llamafactory_alpaca(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": _message_content(row["messages"], "system"),
        "instruction": _message_content(row["messages"], "user"),
        "input": "",
        "output": _assistant_json(row["output"]),
        "route_type": clean_text(row["output"].get("route_type")),
    }


def build_export_report(
    rows_by_split: dict[str, list[dict[str, Any]]],
    *,
    outputs: dict[str, dict[str, str]],
) -> dict[str, Any]:
    all_rows = [row for split in SPLITS for row in rows_by_split[split]]
    route_counts = Counter(clean_text(row["output"].get("route_type")) for row in all_rows)
    split_counts = {split: len(rows_by_split[split]) for split in SPLITS}
    return {
        "status": "ok",
        "total": len(all_rows),
        "split_counts": split_counts,
        "route_counts": dict(sorted(route_counts.items())),
        "outputs": outputs,
        "file_sizes_bytes": _output_file_sizes(outputs),
        "formats": {
            "openai_messages": "JSONL，每行 messages=[system,user,assistant]，assistant 为 RouteDecision JSON 字符串。",
            "swift_messages": "JSONL，和 OpenAI messages 相同，供 ms-swift 直接指定 dataset。",
            "dashscope_messages": "JSONL，每行只保留 messages，供阿里云百炼 fine-tune 上传。",
            "llamafactory_alpaca": "JSON 数组，字段 system/instruction/input/output，配套 dataset_info.json。",
        },
    }


def write_llamafactory_dataset_info(output_dir: Path) -> None:
    dataset_info = {}
    for split in SPLITS:
        dataset_info[f"gustobot_router_{split}"] = {
            "file_name": f"router_llamafactory_{split}.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _message_content(messages: list[dict[str, Any]], role: str) -> str:
    for message in messages:
        if message.get("role") == role:
            return clean_text(message.get("content"))
    return ""


def _assistant_json(output: dict[str, Any]) -> str:
    return json.dumps(
        {
            "route_type": clean_text(output.get("route_type")),
            "confidence": output.get("confidence"),
            "reason": clean_text(output.get("reason")),
            "slots": output.get("slots") or {},
            "need_clarification": bool(output.get("need_clarification")),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _output_file_sizes(outputs: dict[str, dict[str, str]]) -> dict[str, dict[str, int]]:
    sizes: dict[str, dict[str, int]] = {}
    for split, paths in outputs.items():
        sizes[split] = {}
        for name, path_value in paths.items():
            path = Path(path_value)
            sizes[split][name] = path.stat().st_size if path.exists() else 0
    return sizes


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("output") or {}
    return {
        "route_type": clean_text(output.get("route_type")),
        "need_clarification": bool(output.get("need_clarification")),
        "source": "gustobot_router_sft",
    }


def _write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
