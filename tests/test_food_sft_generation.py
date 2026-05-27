"""正式食品 SFT 生成测试。"""

import json
import random
from pathlib import Path

import pytest

from scripts.data.food_dataset import write_jsonl
from scripts.data import generate_food_sft


def test_generate_food_sft_writes_router_and_answer_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    eval_path = tmp_path / "food_eval.jsonl"
    eval_report_path = tmp_path / "eval_after_food_data.json"
    report_path = tmp_path / "sft_food_generation.json"
    router_train = tmp_path / "sft" / "router_train.jsonl"
    router_dev = tmp_path / "sft" / "router_dev.jsonl"
    router_test = tmp_path / "sft" / "router_test.jsonl"
    answer_train = tmp_path / "sft" / "answer_train.jsonl"
    answer_dev = tmp_path / "sft" / "answer_dev.jsonl"
    answer_test = tmp_path / "sft" / "answer_test.jsonl"

    write_jsonl(eval_path, _eval_rows())
    eval_report_path.write_text(json.dumps(_passing_eval_report(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(generate_food_sft, "run_chat_for_sample", _fake_chat_runner)
    monkeypatch.setattr(generate_food_sft, "resolve_project_path", lambda path_value, *, label: Path(path_value).resolve())

    generate_food_sft.main(
        [
            "--eval-samples",
            str(eval_path),
            "--eval-report",
            str(eval_report_path),
            "--report",
            str(report_path),
            "--router-count",
            "30",
            "--answer-count",
            "8",
            "--answer-candidate-limit",
            "4",
            "--router-train-output",
            str(router_train),
            "--router-dev-output",
            str(router_dev),
            "--router-test-output",
            str(router_test),
            "--answer-train-output",
            str(answer_train),
            "--answer-dev-output",
            str(answer_dev),
            "--answer-test-output",
            str(answer_test),
            "--seed",
            "20260526",
        ]
    )

    router_rows = _read_jsonl(router_train) + _read_jsonl(router_dev) + _read_jsonl(router_test)
    answer_rows = _read_jsonl(answer_train) + _read_jsonl(answer_dev) + _read_jsonl(answer_test)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["status"] == "ok"
    assert len(router_rows) == 30
    assert len(answer_rows) == 8
    assert set(report["router_sft"]["route_counts"]) == {"clarify", "general", "graphrag", "kb", "multi", "text2sql"}
    assert report["router_sft"]["splits"] == {"train": 24, "dev": 3, "test": 3}
    assert report["answer_sft"]["splits"] == {"train": 6, "dev": 1, "test": 1}
    assert all(row["messages"][0]["content"] == "你是 GustoBot-v2 的结构化 Router。" for row in router_rows)
    assert all(row["instruction"] == generate_food_sft.ANSWER_INSTRUCTION for row in answer_rows)
    assert all(_answer_has_all_evidence_sources(row) for row in answer_rows)


def test_validate_eval_report_requires_thresholds() -> None:
    report = _passing_eval_report()
    report["router_accuracy"] = 0.84

    validation = generate_food_sft.validate_eval_report(report)

    assert validation["errors"]
    assert "router_accuracy" in validation["errors"][0]


def test_router_sft_generation_is_deterministic_for_same_seed() -> None:
    samples = _eval_rows()

    first = generate_food_sft.build_router_sft(samples, 24, random.Random(7))
    second = generate_food_sft.build_router_sft(samples, 24, random.Random(7))

    assert first == second


def _eval_rows() -> list[dict]:
    return [
        _eval("food-general-001", "你好，介绍一下你能做什么", "general", ["general"]),
        _eval("food-kb-001", "解释一下蛋白质在食品营养中的作用", "kb", ["kb"]),
        _eval("food-graph-001", "Sample Milk 有哪些配料？", "graphrag", ["graph"]),
        _eval("food-sql-001", "统计糖分最高的前 10 个产品", "text2sql", ["sql"]),
        _eval("food-clarify-001", "delete from food_products", "clarify", ["guardrail"], should_block=True),
        _eval("food-multi-001", "统计糖分最高的前 5 个产品，并解释糖分是什么意思？", "multi", ["sql", "kb"], min_evidence_count=2),
    ]


def _eval(
    sample_id: str,
    message: str,
    route: str,
    sources: list[str],
    *,
    should_block: bool = False,
    min_evidence_count: int = 1,
) -> dict:
    return {
        "sample_id": sample_id,
        "category": route,
        "message": message,
        "expected_route": route,
        "expected_evidence_sources": sources,
        "expected_answer_contains": [],
        "should_block": should_block,
        "min_evidence_count": min_evidence_count,
    }


def _passing_eval_report() -> dict:
    return {
        "total": 230,
        "router_accuracy": 1.0,
        "evidence_source_coverage_rate": 1.0,
        "citation_coverage_rate": 1.0,
        "dangerous_input_block_accuracy": 1.0,
        "failure_reason_counts": {},
    }


def _fake_chat_runner(sample: dict) -> dict:
    route = sample["expected_route"]
    source_type = {"kb": "kb", "graphrag": "graph", "text2sql": "sql", "multi": "kb"}[route]
    evidences = [
        {
            "source_type": source_type,
            "source_id": f"{source_type}:{sample['sample_id']}:1",
            "score": 0.91,
            "content": f"{sample['message']} 的证据内容。",
        }
    ]
    if route == "multi":
        evidences.append(
            {
                "source_type": "sql",
                "source_id": f"sql:{sample['sample_id']}:2",
                "score": 0.88,
                "content": "统计结果证据。",
            }
        )
    source_ids = ", ".join(evidence["source_id"] for evidence in evidences)
    return {
        "answer": f"这是基于证据的回答。\n\n来源：{source_ids}",
        "route_decision": {"route_type": route},
        "evidences": evidences,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _answer_has_all_evidence_sources(row: dict) -> bool:
    output = row["output"]
    return all(evidence["source_id"] in output for evidence in row["input"]["evidence"])
