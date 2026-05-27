"""Router SFT 训练格式导出测试。"""

import json
from pathlib import Path

from scripts.data.export_router_sft_training import (
    build_export_report,
    export_router_training_files,
    to_dashscope_messages,
    to_llamafactory_alpaca,
    to_openai_messages,
    validate_router_rows,
    write_llamafactory_dataset_info,
)


def test_router_sft_export_formats_are_parseable(tmp_path: Path) -> None:
    row = _router_row("text2sql")

    openai_row = to_openai_messages(row)
    dashscope_row = to_dashscope_messages(row)
    llama_row = to_llamafactory_alpaca(row)

    assistant_payload = json.loads(openai_row["messages"][-1]["content"])
    dashscope_payload = json.loads(dashscope_row["messages"][-1]["content"])
    llama_payload = json.loads(llama_row["output"])
    assert assistant_payload["route_type"] == "text2sql"
    assert dashscope_payload["route_type"] == "text2sql"
    assert llama_payload["route_type"] == "text2sql"
    assert openai_row["metadata"]["route_type"] == "text2sql"
    assert set(dashscope_row) == {"messages"}
    assert llama_row["instruction"].startswith("{")


def test_export_router_training_files_writes_all_splits(tmp_path: Path) -> None:
    rows_by_split = {
        "train": [_router_row("kb"), _router_row("graphrag")],
        "dev": [_router_row("text2sql")],
        "test": [_router_row("clarify")],
    }

    outputs = export_router_training_files(rows_by_split, output_dir=tmp_path)
    write_llamafactory_dataset_info(tmp_path)
    report = build_export_report(rows_by_split, outputs=outputs)

    assert report["total"] == 4
    assert report["split_counts"] == {"train": 2, "dev": 1, "test": 1}
    assert (tmp_path / "router_openai_train.jsonl").exists()
    assert (tmp_path / "router_swift_dev.jsonl").exists()
    assert (tmp_path / "router_dashscope_dev.jsonl").exists()
    assert (tmp_path / "router_llamafactory_test.json").exists()
    assert report["file_sizes_bytes"]["train"]["dashscope_messages"] > 0
    assert "gustobot_router_train" in json.loads((tmp_path / "dataset_info.json").read_text(encoding="utf-8"))


def test_validate_router_rows_reports_missing_output() -> None:
    errors = validate_router_rows(
        [
            {
                "messages": [
                    {"role": "system", "content": "router"},
                    {"role": "user", "content": "hi"},
                ]
            }
        ]
    )

    assert errors
    assert any("output" in error for error in errors)


def _router_row(route_type: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": "统计糖分最高的前 10 个产品",
                        "input_features": {"contains_statistical_intent": True},
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "output": {
            "route_type": route_type,
            "confidence": 0.88,
            "reason": "食品数据底座 Router SFT 样本。",
            "slots": {"sample_category": route_type},
            "need_clarification": route_type == "clarify",
        },
    }
