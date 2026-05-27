"""基于正式食品评估闭环生成 Router/Answer SFT 数据。

这个脚本不重新生成 eval，也不重新清洗数据。它只消费已经稳定的
data/eval/food_eval.jsonl 和 reports/eval_after_food_data.json，并通过当前
workflow 重新采集真实 Evidence 来生成 Answer SFT。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    EVAL_DIR,
    REPORTS_DIR,
    SFT_DIR,
    clean_text,
    load_jsonl,
    resolve_project_path,
    write_jsonl,
)


DEFAULT_EVAL_SAMPLES = EVAL_DIR / "food_eval.jsonl"
DEFAULT_EVAL_REPORT = REPORTS_DIR / "eval_after_food_data.json"
DEFAULT_REPORT = REPORTS_DIR / "sft_food_generation.json"
DEFAULT_SEED = 20260526
DEFAULT_ROUTER_COUNT = 800
DEFAULT_ANSWER_COUNT = 300
SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
ROUTE_ORDER = ["general", "kb", "graphrag", "text2sql", "clarify", "multi"]
ANSWER_ROUTES = {"kb", "graphrag", "text2sql", "multi"}
CITELESS_SOURCE_TYPES = {"general", "clarify", "guardrail"}
ANSWER_INSTRUCTION = "根据给定 Evidence 回答，必须引用 source_id，不要编造 Evidence 之外的信息。"
EVAL_THRESHOLDS = {
    "router_accuracy": 0.85,
    "evidence_source_coverage_rate": 0.8,
    "citation_coverage_rate": 0.9,
    "dangerous_input_block_accuracy": 1.0,
}


@dataclass(slots=True)
class WorkflowCollectionResult:
    sample: dict[str, Any]
    payload: dict[str, Any] | None = None
    error: str = ""


@dataclass(slots=True)
class AnswerBuildResult:
    rows: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    eval_samples_path = resolve_project_path(args.eval_samples, label="eval-samples")
    eval_report_path = resolve_project_path(args.eval_report, label="eval-report")
    report_path = resolve_project_path(args.report, label="report")

    samples = load_eval_samples(eval_samples_path)
    eval_report = _load_eval_report(eval_report_path)
    validation = validate_eval_report(eval_report)
    if validation["errors"] and not args.allow_unverified:
        raise SystemExit(
            "正式 eval 报告未达标，拒绝生成 SFT。"
            f" 问题：{'; '.join(validation['errors'])}。"
            " 如需调试可加 --allow-unverified。"
        )

    router_rows = build_router_sft(samples, args.router_count, random.Random(args.seed))
    answer_result = build_answer_sft(
        samples,
        args.answer_count,
        random.Random(args.seed + 1),
        chat_runner=run_chat_for_sample,
        max_workers=args.max_workers,
        candidate_limit=args.answer_candidate_limit,
    )

    router_splits = split_train_dev_test(router_rows, random.Random(args.seed + 2))
    answer_splits = split_train_dev_test(answer_result.rows, random.Random(args.seed + 3))
    output_paths = _output_paths(args)
    for split_name, rows in router_splits.items():
        write_jsonl(output_paths[f"router_{split_name}"], rows)
    for split_name, rows in answer_splits.items():
        write_jsonl(output_paths[f"answer_{split_name}"], rows)

    report = build_sft_report(
        samples=samples,
        eval_report=eval_report,
        validation=validation,
        router_splits=router_splits,
        answer_splits=answer_splits,
        answer_stats=answer_result.stats,
        seed=args.seed,
        requested_answer_count=args.answer_count,
        inputs={
            "eval_samples": str(eval_samples_path),
            "eval_report": str(eval_report_path),
        },
        outputs={key: str(path) for key, path in output_paths.items()} | {"report": str(report_path)},
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(report_path),
                "router_sft": report["router_sft"],
                "answer_sft": report["answer_sft"],
                "validation_warnings": validation["warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate official food Router/Answer SFT datasets.")
    parser.add_argument("--eval-samples", default=str(DEFAULT_EVAL_SAMPLES))
    parser.add_argument("--eval-report", default=str(DEFAULT_EVAL_REPORT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--router-count", type=int, default=DEFAULT_ROUTER_COUNT)
    parser.add_argument("--answer-count", type=int, default=DEFAULT_ANSWER_COUNT)
    parser.add_argument("--answer-candidate-limit", type=int, default=0, help="Answer SFT 采集 workflow 样本数；0 表示自动估算。")
    parser.add_argument("--max-workers", type=int, default=1, help="采集 Answer Evidence 时的并发数。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--allow-unverified", action="store_true", help="允许 eval 报告未达标时生成调试用 SFT。")
    parser.add_argument("--router-train-output", default=str(SFT_DIR / "router_train.jsonl"))
    parser.add_argument("--router-dev-output", default=str(SFT_DIR / "router_dev.jsonl"))
    parser.add_argument("--router-test-output", default=str(SFT_DIR / "router_test.jsonl"))
    parser.add_argument("--answer-train-output", default=str(SFT_DIR / "answer_train.jsonl"))
    parser.add_argument("--answer-dev-output", default=str(SFT_DIR / "answer_dev.jsonl"))
    parser.add_argument("--answer-test-output", default=str(SFT_DIR / "answer_test.jsonl"))
    return parser.parse_args(argv)


def load_eval_samples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for row in load_jsonl(path):
        message = clean_text(row.get("message"))
        route = clean_text(row.get("expected_route"))
        if not message or not route:
            continue
        rows.append(
            {
                "sample_id": clean_text(row.get("sample_id")) or f"sample-{len(rows) + 1:04d}",
                "category": clean_text(row.get("category")) or route,
                "message": message,
                "expected_route": route,
                "expected_evidence_sources": list(row.get("expected_evidence_sources") or []),
                "expected_answer_contains": list(row.get("expected_answer_contains") or []),
                "should_block": bool(row.get("should_block", False)),
                "min_evidence_count": int(row.get("min_evidence_count", 1)),
                "attachments": list(row.get("attachments") or []),
            }
        )
    if not rows:
        raise SystemExit(f"没有可用 eval 样本，无法生成 SFT：{path}")
    return rows


def validate_eval_report(report: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    for metric, threshold in EVAL_THRESHOLDS.items():
        value = _float_value(report.get(metric))
        ok = value >= threshold
        checks.append({"metric": metric, "value": value, "threshold": threshold, "ok": ok})
        if not ok:
            errors.append(f"{metric}={value:.3f} < {threshold:.3f}")
    if int(report.get("total") or 0) <= 0:
        errors.append("total 为空或为 0")
    if report.get("failure_reason_counts"):
        warnings.append("eval 报告仍包含 failure_reason_counts，建议先归因确认")
    return {"checks": checks, "errors": errors, "warnings": warnings}


def build_router_sft(samples: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    grouped = _samples_by_route(samples)
    route_targets = _allocate_counts(Counter(sample["expected_route"] for sample in samples), count)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for route in _ordered_routes(route_targets):
        route_samples = grouped.get(route, [])
        if not route_samples:
            continue
        route_target = route_targets[route]
        attempts = 0
        while sum(1 for row in rows if row["output"]["route_type"] == route) < route_target:
            sample = route_samples[attempts % len(route_samples)]
            variant_index = attempts // len(route_samples)
            question = _question_variant(sample, variant_index)
            features = _input_features(question, sample)
            user_payload = json.dumps({"question": question, "input_features": features}, ensure_ascii=False, sort_keys=True)
            attempts += 1
            if user_payload in seen:
                if attempts > route_target * 20:
                    break
                continue
            seen.add(user_payload)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
                        {"role": "user", "content": user_payload},
                    ],
                    "output": _route_decision_from_sample(route, sample, question),
                }
            )

    if len(rows) < count:
        # 极小测试集可能无法靠自然改写凑满，继续轮转所有样本补齐。
        fallback_samples = list(samples)
        rng.shuffle(fallback_samples)
        index = 0
        while len(rows) < count:
            sample = fallback_samples[index % len(fallback_samples)]
            question = _question_variant(sample, index + 1000)
            features = _input_features(question, sample) | {"augmentation_index": index}
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
                        {"role": "user", "content": json.dumps({"question": question, "input_features": features}, ensure_ascii=False)},
                    ],
                    "output": _route_decision_from_sample(sample["expected_route"], sample, question),
                }
            )
            index += 1

    rows = rows[:count]
    rng.shuffle(rows)
    return rows


def build_answer_sft(
    samples: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    *,
    chat_runner: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
    candidate_limit: int = 0,
) -> AnswerBuildResult:
    if count <= 0:
        return AnswerBuildResult(rows=[], stats={"workflow_candidate_count": 0})

    candidates = select_answer_candidates(samples, count, rng, candidate_limit=candidate_limit)
    collection = collect_workflow_payloads(candidates, chat_runner=chat_runner, max_workers=max_workers)
    actual_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped = Counter()
    failures: list[dict[str, str]] = []

    for item in collection:
        if item.error:
            skipped["workflow_error"] += 1
            failures.append({"sample_id": item.sample["sample_id"], "error": item.error[:300]})
            continue
        payload = item.payload or {}
        route = clean_text((payload.get("route_decision") or {}).get("route_type"))
        if route != item.sample["expected_route"]:
            skipped["route_mismatch"] += 1
            continue
        evidences = _citeable_evidences(payload.get("evidences") or [])
        if not evidences:
            skipped["no_citeable_evidence"] += 1
            continue
        normalized_evidences = [_normalize_evidence(evidence) for evidence in evidences[:5]]
        answer = _ensure_citations(clean_text(payload.get("answer")), _source_ids(normalized_evidences))
        if not _answer_mentions_sources(answer, normalized_evidences):
            skipped["answer_without_source_id"] += 1
            continue
        actual_rows.append(
            _answer_row(
                question=item.sample["message"],
                route=route,
                evidence=normalized_evidences,
                output=answer,
            )
        )
        for evidence in normalized_evidences:
            summary_rows.append(
                _answer_row(
                    question=item.sample["message"],
                    route=route,
                    evidence=[evidence],
                    output=_single_evidence_answer(evidence),
                )
            )

    tagged_rows = _dedupe_tagged_answer_rows(
        [(row, "workflow_answer") for row in actual_rows]
        + [(row, "evidence_summary") for row in summary_rows]
    )
    if len(tagged_rows) > count:
        tagged_rows = _route_balanced_take_tagged(tagged_rows, count, rng)
    rows = [row for row, _mode in tagged_rows]
    mode_counts = Counter(mode for _row, mode in tagged_rows)
    rng.shuffle(rows)
    stats = {
        "workflow_candidate_count": len(candidates),
        "workflow_success_count": sum(1 for item in collection if not item.error),
        "workflow_failure_count": sum(1 for item in collection if item.error),
        "skipped": dict(sorted(skipped.items())),
        "failures_preview": failures[:20],
        "generation_mode_counts": dict(sorted(mode_counts.items())),
    }
    return AnswerBuildResult(rows=rows, stats=stats)


def select_answer_candidates(
    samples: list[dict[str, Any]],
    answer_count: int,
    rng: random.Random,
    *,
    candidate_limit: int = 0,
) -> list[dict[str, Any]]:
    eligible = [
        sample
        for sample in samples
        if sample["expected_route"] in ANSWER_ROUTES and not sample.get("should_block")
    ]
    if not eligible:
        return []
    limit = candidate_limit if candidate_limit > 0 else _auto_answer_candidate_limit(answer_count, len(eligible))
    grouped = _samples_by_route(eligible)
    for route_samples in grouped.values():
        rng.shuffle(route_samples)
    selected: list[dict[str, Any]] = []
    route_positions = defaultdict(int)
    while len(selected) < min(limit, len(eligible)):
        progressed = False
        for route in ("kb", "graphrag", "text2sql", "multi"):
            route_samples = grouped.get(route, [])
            position = route_positions[route]
            if position >= len(route_samples):
                continue
            selected.append(route_samples[position])
            route_positions[route] += 1
            progressed = True
            if len(selected) >= min(limit, len(eligible)):
                break
        if not progressed:
            break
    return selected


def collect_workflow_payloads(
    samples: list[dict[str, Any]],
    *,
    chat_runner: Callable[[dict[str, Any]], dict[str, Any]],
    max_workers: int = 1,
) -> list[WorkflowCollectionResult]:
    worker_count = max(1, max_workers)
    if worker_count == 1 or len(samples) <= 1:
        return [_run_sample_safely(sample, chat_runner) for sample in samples]

    ordered: list[WorkflowCollectionResult | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=min(worker_count, len(samples))) as executor:
        futures = {
            executor.submit(_run_sample_safely, sample, chat_runner): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    return [item for item in ordered if item is not None]


def run_chat_for_sample(sample: dict[str, Any]) -> dict[str, Any]:
    from app.graph.workflow import run_chat
    from app.models import Attachment, ChatRequest

    response = run_chat(
        ChatRequest(
            message=sample["message"],
            attachments=[Attachment(**item) for item in sample.get("attachments", [])],
        )
    )
    return response.model_dump(mode="json")


def split_train_dev_test(rows: list[dict[str, Any]], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * SPLIT_RATIOS["train"])
    dev_end = int(total * (SPLIT_RATIOS["train"] + SPLIT_RATIOS["dev"]))
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }


def build_sft_report(
    *,
    samples: list[dict[str, Any]],
    eval_report: dict[str, Any],
    validation: dict[str, Any],
    router_splits: dict[str, list[dict[str, Any]]],
    answer_splits: dict[str, list[dict[str, Any]]],
    answer_stats: dict[str, Any],
    seed: int,
    requested_answer_count: int,
    inputs: dict[str, str],
    outputs: dict[str, str],
) -> dict[str, Any]:
    router_rows = [row for rows in router_splits.values() for row in rows]
    answer_rows = [row for rows in answer_splits.values() for row in rows]
    status = "ok" if not validation["errors"] else "ok_with_unverified_eval"
    if len(answer_rows) < requested_answer_count:
        status = "ok_with_warnings"
    return {
        "status": status,
        "seed": seed,
        "inputs": inputs,
        "outputs": outputs,
        "eval": {
            "sample_count": len(samples),
            "report_total": eval_report.get("total"),
            "metrics": {metric: eval_report.get(metric) for metric in EVAL_THRESHOLDS},
            "validation": validation,
        },
        "router_sft": {
            "total": len(router_rows),
            "splits": {key: len(value) for key, value in router_splits.items()},
            "route_counts": dict(sorted(Counter(row["output"]["route_type"] for row in router_rows).items())),
        },
        "answer_sft": {
            "total": len(answer_rows),
            "splits": {key: len(value) for key, value in answer_splits.items()},
            "route_counts": dict(sorted(Counter(row["input"].get("route_type") for row in answer_rows).items())),
            "evidence_source_counts": _answer_evidence_source_counts(answer_rows),
            **answer_stats,
        },
        "notes": [
            "Router SFT 来自正式 food_eval.jsonl 的标签和问题改写。",
            "Answer SFT 通过当前 workflow 重新采集 Evidence，只保留含可引用 source_id 的样本。",
        ],
    }


def _output_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "router_train": resolve_project_path(args.router_train_output, label="router-train-output"),
        "router_dev": resolve_project_path(args.router_dev_output, label="router-dev-output"),
        "router_test": resolve_project_path(args.router_test_output, label="router-test-output"),
        "answer_train": resolve_project_path(args.answer_train_output, label="answer-train-output"),
        "answer_dev": resolve_project_path(args.answer_dev_output, label="answer-dev-output"),
        "answer_test": resolve_project_path(args.answer_test_output, label="answer-test-output"),
    }


def _load_eval_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"找不到正式 eval 报告：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _samples_by_route(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["expected_route"]].append(sample)
    return grouped


def _allocate_counts(route_counts: Counter[str], total: int) -> dict[str, int]:
    source_total = sum(route_counts.values())
    if source_total <= 0:
        return {}
    raw = {route: total * count / source_total for route, count in route_counts.items()}
    allocated = {route: int(value) for route, value in raw.items()}
    remainder = total - sum(allocated.values())
    for route, _ in sorted(raw.items(), key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True)[:remainder]:
        allocated[route] += 1
    return allocated


def _ordered_routes(route_targets: dict[str, int]) -> list[str]:
    known = [route for route in ROUTE_ORDER if route in route_targets]
    extra = sorted(route for route in route_targets if route not in known)
    return known + extra


def _question_variant(sample: dict[str, Any], variant_index: int) -> str:
    question = sample["message"].strip()
    route = sample["expected_route"]
    if route == "clarify" and sample.get("should_block"):
        variants = [
            "{q}",
            "请处理这个数据库请求：{q}",
            "帮我执行一下：{q}",
            "这个 SQL 可以直接运行吗：{q}",
            "后台表操作：{q}",
        ]
    elif route == "clarify":
        variants = [
            "{q}",
            "我说的是这个：{q}",
            "帮我看看这个",
            "这个食品怎么样？",
            "请分析一下这个",
        ]
    elif route == "general":
        variants = [
            "{q}",
            "请简单回答：{q}",
            "我想了解一下：{q}",
            "{q} 可以吗？",
            "帮我说明：{q}",
        ]
    else:
        variants = [
            "{q}",
            "请帮我查一下：{q}",
            "基于食品数据，{q}",
            "我想问：{q}",
            "{q} 请给出依据。",
            "用当前食品数据回答：{q}",
        ]
    template = variants[variant_index % len(variants)]
    return template.format(q=question)


def _input_features(question: str, sample: dict[str, Any]) -> dict[str, Any]:
    lowered = question.lower()
    return {
        "language": "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en",
        "has_attachment": bool(sample.get("attachments")),
        "attachment_types": [item.get("type") for item in sample.get("attachments", []) if item.get("type")],
        "message_length": len(question),
        "contains_sql_mutation": _contains_sql_mutation(lowered),
        "contains_statistical_intent": _contains_any(question, ("统计", "排行", "top", "数量", "多少", "最高", "最低")),
        "contains_relation_intent": _contains_any(question, ("属于", "配料", "过敏原", "关系", "有哪些")),
        "contains_knowledge_intent": _contains_any(question, ("解释", "说明", "是什么", "作用", "怎么看", "介绍")),
        "is_low_context": len(question) <= 8 or question in {"这个呢", "说一下", "分析一下", "帮我看看"},
    }


def _route_decision_from_sample(route: str, sample: dict[str, Any], question: str) -> dict[str, Any]:
    should_block = bool(sample.get("should_block"))
    slots = _slots_for_route(route, sample, question)
    return {
        "route_type": route,
        "confidence": _confidence_for_route(route, should_block),
        "reason": "食品数据底座 Router SFT 样本。",
        "slots": slots,
        "need_clarification": route == "clarify" and not should_block,
    }


def _slots_for_route(route: str, sample: dict[str, Any], question: str) -> dict[str, Any]:
    slots: dict[str, Any] = {"sample_category": sample.get("category") or route}
    if route == "kb":
        slots["knowledge_intent"] = True
    elif route == "graphrag":
        slots["relation_intent"] = True
    elif route == "text2sql":
        slots["analysis_intent"] = True
    elif route == "multi":
        slots["composite_intent"] = True
        slots["expected_evidence_sources"] = sample.get("expected_evidence_sources", [])
    elif route == "clarify":
        slots["guardrail_required"] = bool(sample.get("should_block"))
        slots["low_context"] = _input_features(question, sample)["is_low_context"]
    if sample.get("attachments"):
        slots["attachment_types"] = [item.get("type") for item in sample.get("attachments", []) if item.get("type")]
    return slots


def _confidence_for_route(route: str, should_block: bool) -> float:
    if route == "clarify":
        return 0.97 if should_block else 0.42
    return {
        "general": 0.9,
        "kb": 0.86,
        "graphrag": 0.87,
        "text2sql": 0.88,
        "multi": 0.84,
    }.get(route, 0.75)


def _run_sample_safely(
    sample: dict[str, Any],
    chat_runner: Callable[[dict[str, Any]], dict[str, Any]],
) -> WorkflowCollectionResult:
    try:
        return WorkflowCollectionResult(sample=sample, payload=chat_runner(sample))
    except Exception as exc:  # pragma: no cover - 真实 Docker 链路错误只在报告中记录。
        return WorkflowCollectionResult(sample=sample, error=str(exc))


def _citeable_evidences(evidences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for evidence in evidences:
        source_type = _source_type_value(evidence.get("source_type"))
        if source_type in CITELESS_SOURCE_TYPES:
            continue
        if not clean_text(evidence.get("source_id")) or not clean_text(evidence.get("content")):
            continue
        rows.append(evidence)
    return rows


def _normalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": _source_type_value(evidence.get("source_type")),
        "source_id": clean_text(evidence.get("source_id")),
        "score": round(_float_value(evidence.get("score")), 4),
        "content": _truncate_text(clean_text(evidence.get("content")), 1200),
    }


def _answer_row(question: str, route: str, evidence: list[dict[str, Any]], output: str) -> dict[str, Any]:
    return {
        "instruction": ANSWER_INSTRUCTION,
        "input": {
            "question": question,
            "route_type": route,
            "evidence": evidence,
        },
        "output": output,
    }


def _single_evidence_answer(evidence: dict[str, Any]) -> str:
    snippet = _truncate_text(evidence["content"], 420)
    return f"根据给定 Evidence，可以确认：{snippet} 来源：{evidence['source_id']}"


def _ensure_citations(answer: str, source_ids: list[str]) -> str:
    text = answer or "根据给定 Evidence 可以回答该问题。"
    missing = [source_id for source_id in source_ids if source_id not in text]
    if missing:
        text = f"{text.rstrip()}\n\n来源：{', '.join(source_ids[:5])}"
    return text


def _answer_mentions_sources(answer: str, evidences: list[dict[str, Any]]) -> bool:
    return all(evidence["source_id"] in answer for evidence in evidences[:5])


def _source_ids(evidences: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for evidence in evidences:
        source_id = evidence["source_id"]
        if source_id not in ids:
            ids.append(source_id)
    return ids


def _dedupe_answer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen: set[str] = set()
    for row in rows:
        key = _answer_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _dedupe_tagged_answer_rows(rows: list[tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], str]]:
    deduped = []
    seen: set[str] = set()
    for row, mode in rows:
        key = _answer_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((row, mode))
    return deduped


def _answer_row_key(row: dict[str, Any]) -> str:
    evidence_ids = [item["source_id"] for item in row["input"]["evidence"]]
    return json.dumps(
        {
            "question": row["input"]["question"],
            "route_type": row["input"]["route_type"],
            "evidence_ids": evidence_ids,
            "output": row["output"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _route_balanced_take(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["input"].get("route_type", "")].append(row)
    for route_rows in grouped.values():
        rng.shuffle(route_rows)
    selected: list[dict[str, Any]] = []
    positions = defaultdict(int)
    while len(selected) < count:
        progressed = False
        for route in ("kb", "graphrag", "text2sql", "multi"):
            position = positions[route]
            route_rows = grouped.get(route, [])
            if position >= len(route_rows):
                continue
            selected.append(route_rows[position])
            positions[route] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _route_balanced_take_tagged(
    rows: list[tuple[dict[str, Any], str]],
    count: int,
    rng: random.Random,
) -> list[tuple[dict[str, Any], str]]:
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for row, mode in rows:
        grouped[row["input"].get("route_type", "")].append((row, mode))
    for route_rows in grouped.values():
        rng.shuffle(route_rows)
    selected: list[tuple[dict[str, Any], str]] = []
    positions = defaultdict(int)
    while len(selected) < count:
        progressed = False
        for route in ("kb", "graphrag", "text2sql", "multi"):
            position = positions[route]
            route_rows = grouped.get(route, [])
            if position >= len(route_rows):
                continue
            selected.append(route_rows[position])
            positions[route] += 1
            progressed = True
            if len(selected) >= count:
                break
        if not progressed:
            break
    return selected


def _answer_evidence_source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        evidence["source_type"]
        for row in rows
        for evidence in row.get("input", {}).get("evidence", [])
    )
    return dict(sorted(counts.items()))


def _auto_answer_candidate_limit(answer_count: int, eligible_count: int) -> int:
    if answer_count <= 0:
        return 0
    return min(eligible_count, max(1, int(answer_count * 0.6)))


def _contains_sql_mutation(lowered: str) -> bool:
    return bool(re.search(r"\b(delete|drop|truncate|alter|update|insert|create)\b", lowered)) or any(
        phrase in lowered for phrase in ("清空", "删除", "删表", "改表")
    )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _source_type_value(value: Any) -> str:
    text = clean_text(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _float_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
