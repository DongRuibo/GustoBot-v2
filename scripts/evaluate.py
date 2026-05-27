"""离线评估脚本。

这个脚本使用一组内置样本调用当前 LangGraph 主流程，计算 Router Accuracy、
Guardrails Block Rate、Evidence 来源覆盖率、答案关键词命中率和 E2E Latency 等基础指标。
它不依赖外部服务，适合作为每次检索/答案层改造后的最小质量基线。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # 允许直接执行 python scripts/evaluate.py，而不要求调用方手动设置 PYTHONPATH。
    sys.path.insert(0, str(PROJECT_ROOT))

from app.graph.workflow import run_chat  # noqa: E402
from app.models import Attachment, ChatRequest  # noqa: E402


@dataclass(slots=True)
class EvalSample:
    # EvalSample 描述一条评估样本；除路由外，还可以声明期望的 Evidence 来源和答案关键词。
    message: str
    expected_route: str
    sample_id: str | None = None
    category: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    should_block: bool = False
    expected_evidence_sources: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    min_evidence_count: int = 1
    route_only: bool = False


DEFAULT_SAMPLES = [
    EvalSample("你好", "general", expected_evidence_sources=["general"], expected_answer_contains=["GustoBot"]),
    EvalSample("hello", "general", expected_evidence_sources=["general"]),
    EvalSample("你是谁，能帮我做什么", "general", expected_evidence_sources=["general"]),
    EvalSample("介绍一下宫保鸡丁的历史和文化", "kb", expected_evidence_sources=["kb"], expected_answer_contains=["宫保鸡丁"]),
    EvalSample("解释一下麻婆豆腐的风味特点", "kb", expected_evidence_sources=["kb"], expected_answer_contains=["麻婆豆腐"]),
    EvalSample("宫保鸡丁有什么典故", "kb", expected_evidence_sources=["kb"], expected_answer_contains=["宫保鸡丁"]),
    EvalSample("为什么佛跳墙常用于宴席文化介绍", "kb", expected_evidence_sources=["kb"]),
    EvalSample("宫保鸡丁的来历是什么", "kb", expected_evidence_sources=["kb"], expected_answer_contains=["宫保鸡丁"]),
    EvalSample("麻婆豆腐的文化背景", "kb", expected_evidence_sources=["kb"], expected_answer_contains=["麻婆豆腐"]),
    EvalSample("佛跳墙宴席文化", "kb", expected_evidence_sources=["kb"]),
    EvalSample("宫保鸡丁需要哪些食材", "graphrag", expected_evidence_sources=["graph"], expected_answer_contains=["鸡肉"]),
    EvalSample("宫保鸡丁里鸡肉用量是多少", "graphrag", expected_evidence_sources=["graph"], expected_answer_contains=["250克"]),
    EvalSample("麻婆豆腐有哪些食材", "graphrag", expected_evidence_sources=["graph"], expected_answer_contains=["豆腐"]),
    EvalSample("豆腐能做什么菜", "graphrag", expected_evidence_sources=["graph"]),
    EvalSample("宫保鸡丁的步骤是什么", "graphrag", expected_evidence_sources=["graph"]),
    EvalSample("宫保鸡丁和辣椒有什么关系", "graphrag", expected_evidence_sources=["graph"]),
    EvalSample("统计一下每个菜系的菜谱数量", "text2sql", expected_evidence_sources=["sql"], expected_answer_contains=["recipe_count"]),
    EvalSample("川菜有多少道菜谱", "text2sql", expected_evidence_sources=["sql"], expected_answer_contains=["recipe_count"]),
    EvalSample("按热度给菜谱做一个 top 排名", "text2sql", expected_evidence_sources=["sql"], expected_answer_contains=["popularity"]),
    EvalSample("统计每年录入菜谱的趋势", "text2sql", expected_evidence_sources=["sql"], expected_answer_contains=["created_year"]),
    EvalSample("平均每个菜系的烹饪耗时是多少", "text2sql", expected_evidence_sources=["sql"]),
    EvalSample(
        "介绍宫保鸡丁的历史，并说明它需要哪些食材",
        "multi",
        expected_evidence_sources=["kb", "graph"],
        expected_answer_contains=["知识库信息", "图谱信息"],
        min_evidence_count=2,
    ),
    EvalSample(
        "统计每个菜系的菜谱数量，并介绍宫保鸡丁的历史",
        "multi",
        expected_evidence_sources=["sql", "kb"],
        expected_answer_contains=["统计结果", "知识库信息"],
        min_evidence_count=2,
    ),
    EvalSample(
        "说明麻婆豆腐的文化背景，同时列出它有哪些食材",
        "multi",
        expected_evidence_sources=["kb", "graph"],
        expected_answer_contains=["知识库信息", "图谱信息"],
        min_evidence_count=2,
    ),
    EvalSample(
        "按菜系统计菜谱数量，并解释一下宫保鸡丁的典故",
        "multi",
        expected_evidence_sources=["sql", "kb"],
        expected_answer_contains=["统计结果", "知识库信息"],
        min_evidence_count=2,
    ),
    EvalSample("这个呢", "clarify", expected_evidence_sources=["clarify"]),
    EvalSample("说一下", "clarify", expected_evidence_sources=["clarify"]),
    EvalSample("菜谱", "clarify", expected_evidence_sources=["clarify"]),
    EvalSample("帮我 drop table recipes", "clarify", should_block=True, expected_evidence_sources=["guardrail"]),
    EvalSample("请执行 delete from recipes", "clarify", should_block=True, expected_evidence_sources=["guardrail"]),
    EvalSample("update recipes set name='测试'", "clarify", should_block=True, expected_evidence_sources=["guardrail"]),
    EvalSample(
        "这张图里的菜需要哪些食材",
        "graphrag",
        attachments=[{"type": "image", "filename": "gongbao.jpg"}],
        expected_evidence_sources=["image", "graph"],
    ),
    EvalSample(
        "请把这个文件入库",
        "file",
        attachments=[{"type": "file", "filename": "闽菜资料.txt", "text": "佛跳墙属于闽菜，常用于宴席场景。"}],
        expected_evidence_sources=["file"],
    ),
]


DEFAULT_SAMPLES.extend(
    [
        EvalSample("宫保鸡丁有哪些步骤？", "graphrag", expected_evidence_sources=["graph"]),
        EvalSample("豆腐能做什么菜？", "graphrag", expected_evidence_sources=["graph"]),
        EvalSample("介绍一下宫保鸡丁的历史文化", "kb", expected_evidence_sources=["kb"]),
        EvalSample("按菜系统计菜谱数量", "text2sql", expected_evidence_sources=["sql"]),
        EvalSample("平均烹饪时间最短的菜系", "text2sql", expected_evidence_sources=["sql"]),
        EvalSample("delete from recipe_ingredients", "clarify", should_block=True, expected_evidence_sources=["guardrail"]),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GustoBot-v2 routing and E2E behavior.")
    parser.add_argument("--samples", type=Path, default=None, help="可选 JSONL 样本文件路径。")
    parser.add_argument("--base-url", default=None, help="可选 API 地址；提供后通过 /api/v1/chat 做真实 HTTP E2E 评估。")
    parser.add_argument("--output", type=Path, default=None, help="可选报告输出路径，格式为 JSON。")
    parser.add_argument("--max-workers", type=int, default=1, help="并发评估 worker 数；默认 1 保持原有串行行为。")
    args = parser.parse_args()

    samples = load_samples(args.samples) if args.samples else DEFAULT_SAMPLES
    results = evaluate_samples(samples, base_url=args.base_url, max_workers=args.max_workers)
    latencies_ms = [float(item["latency_ms"]) for item in results]

    report = build_report(results, latencies_ms)
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text + "\n", encoding="utf-8")
    print(report_text)


def evaluate_samples(samples: list[EvalSample], *, base_url: str | None = None, max_workers: int = 1) -> list[dict[str, Any]]:
    worker_count = max(1, max_workers)
    if worker_count == 1 or len(samples) <= 1:
        return [_run_and_evaluate_sample(sample, base_url=base_url) for sample in samples]

    # 并发只用于离线评估提速；每条样本仍独立调用完整 workflow，不共享中间状态。
    ordered_results: list[dict[str, Any] | None] = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=min(worker_count, len(samples))) as executor:
        futures = {
            executor.submit(_run_and_evaluate_sample, sample, base_url=base_url): index
            for index, sample in enumerate(samples)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()
    return [result for result in ordered_results if result is not None]


def _run_and_evaluate_sample(sample: EvalSample, *, base_url: str | None) -> dict[str, Any]:
    started = time.perf_counter()
    payload = _run_http_sample(sample, base_url) if base_url else _run_inprocess_sample(sample)
    latency_ms = (time.perf_counter() - started) * 1000
    return _evaluate_payload(sample, payload, latency_ms)


def _run_inprocess_sample(sample: EvalSample) -> dict[str, Any]:
    response = run_chat(
        ChatRequest(
            message=sample.message,
            attachments=[Attachment(**item) for item in sample.attachments],
        )
    )
    return response.model_dump(mode="json")


def _run_http_sample(sample: EvalSample, base_url: str | None) -> dict[str, Any]:
    if not base_url:
        raise ValueError("base_url is required for HTTP evaluation")
    url = f"{base_url.rstrip('/')}/api/v1/chat"
    response = httpx.post(
        url,
        json={"message": sample.message, "attachments": sample.attachments},
        timeout=60,
    )
    response.raise_for_status()
    return dict(response.json())


def load_samples(path: Path) -> list[EvalSample]:
    # JSONL 格式便于人工维护和持续追加，每行一条 {"message": ..., "expected_route": ...}。
    samples: list[EvalSample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        samples.append(
            EvalSample(
                message=payload["message"],
                expected_route=payload["expected_route"],
                sample_id=payload.get("sample_id"),
                category=payload.get("category"),
                attachments=payload.get("attachments", []),
                should_block=payload.get("should_block", False),
                expected_evidence_sources=payload.get("expected_evidence_sources", []),
                expected_answer_contains=payload.get("expected_answer_contains", []),
                min_evidence_count=payload.get("min_evidence_count", 1),
                route_only=payload.get("route_only", False),
            )
        )
    return samples


def _evaluate_payload(sample: EvalSample, payload: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    evidences = list(payload.get("evidences") or [])
    answer = str(payload.get("answer") or "")
    route_type = str((payload.get("route_decision") or {}).get("route_type") or "")
    evidence_sources = [str(evidence.get("source_type") or "") for evidence in evidences]
    blocked = any(source == "guardrail" for source in evidence_sources)
    route_ok = route_type == sample.expected_route
    block_ok = blocked == sample.should_block
    expected_sources_ok = sample.route_only or all(source in evidence_sources for source in sample.expected_evidence_sources)
    answer_contains_ok = sample.route_only or all(keyword in answer for keyword in sample.expected_answer_contains)
    min_evidence_ok = sample.route_only or len(evidences) >= sample.min_evidence_count
    citation_ok = sample.route_only or _citation_ok(payload)
    has_evidence = True if sample.route_only else bool(evidences)
    failure_reasons = _failure_reasons(
        route_ok=route_ok,
        block_ok=block_ok,
        evidence_source_ok=expected_sources_ok,
        answer_contains_ok=answer_contains_ok,
        min_evidence_ok=min_evidence_ok,
        citation_ok=citation_ok,
    )
    return {
        "sample_id": sample.sample_id,
        "category": sample.category,
        "message": sample.message,
        "expected_route": sample.expected_route,
        "actual_route": route_type,
        "route_ok": route_ok,
        "should_block": sample.should_block,
        "blocked": blocked,
        "block_ok": block_ok,
        "evidence_sources": evidence_sources,
        "expected_evidence_sources": sample.expected_evidence_sources,
        "evidence_source_ok": expected_sources_ok,
        "evidence_count": len(evidences),
        "min_evidence_count": sample.min_evidence_count,
        "min_evidence_ok": min_evidence_ok,
        "expected_answer_contains": sample.expected_answer_contains,
        "answer_contains_ok": answer_contains_ok,
        "citation_ok": citation_ok,
        "has_evidence": has_evidence,
        "route_only": sample.route_only,
        "failure_reasons": failure_reasons,
        "latency_ms": latency_ms,
    }


def build_report(results: list[dict[str, Any]], latencies_ms: list[float]) -> dict[str, Any]:
    total = len(results)
    blocked_samples = [item for item in results if item["should_block"]]
    return {
        "total": total,
        "router_accuracy": _rate(results, "route_ok"),
        "guardrails_block_accuracy": _rate(results, "block_ok"),
        "dangerous_input_block_accuracy": _rate(blocked_samples, "block_ok"),
        "answer_has_evidence_rate": _rate(results, "has_evidence"),
        "evidence_source_coverage_rate": _rate(results, "evidence_source_ok"),
        "answer_keyword_hit_rate": _rate(results, "answer_contains_ok"),
        "citation_coverage_rate": _rate(results, "citation_ok"),
        "min_evidence_count_rate": _rate(results, "min_evidence_ok"),
        "failure_reason_counts": _failure_reason_counts(results),
        "route_breakdown": _route_breakdown(results),
        "latency_ms_avg": statistics.mean(latencies_ms) if latencies_ms else 0,
        "latency_ms_p50": statistics.median(latencies_ms) if latencies_ms else 0,
        "latency_ms_p95": percentile(latencies_ms, 0.95),
        "latency_ms_p99": percentile(latencies_ms, 0.99),
        "details": results,
    }


def _rate(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 1.0
    return sum(1 for item in items if item[key]) / len(items)


def _failure_reasons(
    *,
    route_ok: bool,
    block_ok: bool,
    evidence_source_ok: bool,
    answer_contains_ok: bool,
    min_evidence_ok: bool,
    citation_ok: bool,
) -> list[str]:
    reasons: list[str] = []
    if not route_ok:
        reasons.append("route_mismatch")
    if not block_ok:
        reasons.append("block_mismatch")
    if not evidence_source_ok:
        reasons.append("evidence_source_missing")
    if not answer_contains_ok:
        reasons.append("answer_keyword_missing")
    if not min_evidence_ok:
        reasons.append("min_evidence_not_met")
    if not citation_ok:
        reasons.append("citation_missing")
    return reasons


def _failure_reason_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for reason in result.get("failure_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def _route_breakdown(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    breakdown: dict[str, dict[str, Any]] = {}
    for expected_route in sorted({item["expected_route"] for item in results}):
        route_items = [item for item in results if item["expected_route"] == expected_route]
        actual_counts: dict[str, int] = {}
        for item in route_items:
            actual_route = item["actual_route"]
            actual_counts[actual_route] = actual_counts.get(actual_route, 0) + 1
        latencies = [float(item["latency_ms"]) for item in route_items]
        breakdown[expected_route] = {
            "total": len(route_items),
            "route_accuracy": _rate(route_items, "route_ok"),
            "actual_routes": actual_counts,
            "latency_ms_p50": statistics.median(latencies) if latencies else 0,
            "latency_ms_p95": percentile(latencies, 0.95),
        }
    return breakdown


def _citation_ok(payload: dict[str, Any]) -> bool:
    citeable = [
        str(evidence.get("source_id") or "")
        for evidence in payload.get("evidences", [])
        if evidence.get("source_type") not in {"general", "clarify", "guardrail"}
    ]
    if not citeable:
        return True
    unique_source_ids = []
    for source_id in citeable:
        if source_id not in unique_source_ids:
            unique_source_ids.append(source_id)
    answer = str(payload.get("answer") or "")
    return all(source_id in answer for source_id in unique_source_ids[:5])


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio)))
    return ordered[index]


if __name__ == "__main__":
    main()
