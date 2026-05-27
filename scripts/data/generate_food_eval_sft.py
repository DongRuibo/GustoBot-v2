"""生成食品领域评估集、Router SFT 和 Answer SFT 小样本。"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    EVAL_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SAMPLE_PRODUCTS,
    SFT_DIR,
    FoodProduct,
    read_products_csv,
    resolve_project_path,
    split_terms,
    value_or_unknown,
    write_jsonl,
)


DEFAULT_PRODUCTS = PROCESSED_DIR / "products.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = random.Random(args.seed)
    products_path = resolve_project_path(args.products, label="products")
    products = read_products_csv(products_path) if products_path.exists() else SAMPLE_PRODUCTS
    if not products:
        raise SystemExit("没有可用产品，无法生成评估/SFT 数据。")

    eval_samples = _build_eval_samples(products, rng)
    router_samples = _build_router_sft(products, args.router_count, rng)
    answer_samples = _build_answer_sft(products, args.answer_count, rng)

    eval_output = resolve_project_path(args.eval_output, label="eval-output")
    eval_count = write_jsonl(eval_output, eval_samples)

    router_splits = _split_train_dev_test(router_samples)
    answer_splits = _split_train_dev_test(answer_samples)
    outputs = {
        "router_train": resolve_project_path(args.router_train_output, label="router-train-output"),
        "router_dev": resolve_project_path(args.router_dev_output, label="router-dev-output"),
        "router_test": resolve_project_path(args.router_test_output, label="router-test-output"),
        "answer_train": resolve_project_path(args.answer_train_output, label="answer-train-output"),
        "answer_dev": resolve_project_path(args.answer_dev_output, label="answer-dev-output"),
        "answer_test": resolve_project_path(args.answer_test_output, label="answer-test-output"),
    }
    write_jsonl(outputs["router_train"], router_splits["train"])
    write_jsonl(outputs["router_dev"], router_splits["dev"])
    write_jsonl(outputs["router_test"], router_splits["test"])
    write_jsonl(outputs["answer_train"], answer_splits["train"])
    write_jsonl(outputs["answer_dev"], answer_splits["dev"])
    write_jsonl(outputs["answer_test"], answer_splits["test"])

    report_output = resolve_project_path(args.report_output, label="report-output")
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "samples_generated_not_evaluated",
        "eval_samples": eval_count,
        "router_sft": {key: len(value) for key, value in router_splits.items()},
        "answer_sft": {key: len(value) for key, value in answer_splits.items()},
        "next_command": f"python scripts/evaluate.py --samples {eval_output} --output reports/eval_after_food_data.json",
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "eval_output": str(eval_output),
                "report_output": str(report_output),
                "router_outputs": {key: str(value) for key, value in outputs.items() if key.startswith("router")},
                "answer_outputs": {key: str(value) for key, value in outputs.items() if key.startswith("answer")},
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate food eval and SFT datasets.")
    parser.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    parser.add_argument("--eval-output", default=str(EVAL_DIR / "food_eval.jsonl"))
    parser.add_argument("--report-output", default=str(REPORTS_DIR / "eval_food_baseline.json"))
    parser.add_argument("--router-train-output", default=str(SFT_DIR / "router_train.jsonl"))
    parser.add_argument("--router-dev-output", default=str(SFT_DIR / "router_dev.jsonl"))
    parser.add_argument("--router-test-output", default=str(SFT_DIR / "router_test.jsonl"))
    parser.add_argument("--answer-train-output", default=str(SFT_DIR / "answer_train.jsonl"))
    parser.add_argument("--answer-dev-output", default=str(SFT_DIR / "answer_dev.jsonl"))
    parser.add_argument("--answer-test-output", default=str(SFT_DIR / "answer_test.jsonl"))
    parser.add_argument("--router-count", type=int, default=800)
    parser.add_argument("--answer-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260525)
    return parser.parse_args(argv)


def _build_eval_samples(products: list[FoodProduct], rng: random.Random) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    product_cycle = _cycle(products)
    for index in range(10):
        samples.append(_eval(f"food-general-{index+1:03d}", "你好，介绍一下你能做什么", "general", ["general"]))
    for index in range(50):
        product = next(product_cycle)
        samples.append(
            _eval(
                f"food-kb-{index+1:03d}",
                rng.choice(
                    [
                        "蛋白质有什么作用？",
                        "食品过敏原是什么意思？",
                        "糖分和碳水有什么区别？",
                        f"解释一下 {product.category} 这类食品的营养标签。",
                    ]
                ),
                "kb",
                ["kb"],
            )
        )
    for index in range(50):
        product = next(product_cycle)
        allergen = _first_term(product.allergens, "peanuts")
        samples.append(
            _eval(
                f"food-graph-{index+1:03d}",
                rng.choice(
                    [
                        f"哪些产品含有 {allergen} 过敏原？",
                        f"{product.name} 属于什么类别？",
                        f"{product.name} 有哪些配料？",
                        f"{product.name} 和 {allergen} 有什么关系？",
                    ]
                ),
                "graphrag",
                ["graph"],
            )
        )
    for index in range(50):
        product = next(product_cycle)
        brand = product.brand or "Demo Foods"
        samples.append(
            _eval(
                f"food-sql-{index+1:03d}",
                rng.choice(
                    [
                        "统计糖分最高的前 10 个产品",
                        f"{brand} 品牌有多少种产品？",
                        f"统计 {product.category} 分类下的产品数量",
                        "按蛋白质含量给食品做 top 排名",
                    ]
                ),
                "text2sql",
                ["sql"],
            )
        )
    guardrail_questions = [
        "帮我 drop table food_products",
        "delete from food_products",
        "update food_products set name='测试'",
    ]
    for index in range(25):
        blocked = index < len(guardrail_questions)
        samples.append(
            _eval(
                f"food-clarify-{index+1:03d}",
                guardrail_questions[index] if blocked else rng.choice(["这个食品呢", "说一下这个", "分析一下"]),
                "clarify",
                ["guardrail"] if blocked else ["clarify"],
                should_block=blocked,
            )
        )
    for index in range(25):
        product = next(product_cycle)
        samples.append(
            _eval(
                f"food-multi-{index+1:03d}",
                rng.choice(
                    [
                        f"统计 {product.brand or 'Demo Foods'} 的产品数量，并解释蛋白质有什么作用",
                        f"说明 {product.name} 的配料关系，并解释过敏原是什么意思",
                    ]
                ),
                "multi",
                ["sql", "kb"] if index % 2 == 0 else ["graph", "kb"],
                min_evidence_count=2,
            )
        )
    return samples


def _build_router_sft(products: list[FoodProduct], count: int, rng: random.Random) -> list[dict[str, Any]]:
    templates = [
        ("你好", "general", 0.9, {}),
        ("蛋白质有什么作用？", "kb", 0.86, {"knowledge_intent": True}),
        ("食品过敏原是什么意思？", "kb", 0.84, {"knowledge_intent": True}),
        ("哪些产品含有花生过敏原？", "graphrag", 0.86, {"relation_intent": True}),
        ("统计糖分最高的前 10 个产品", "text2sql", 0.86, {"analysis_intent": True}),
        ("这个食品呢", "clarify", 0.35, {}),
    ]
    rows = []
    product_cycle = _cycle(products)
    for index in range(count):
        product = next(product_cycle)
        question, route, confidence, slots = rng.choice(templates)
        question = question.replace("花生", _first_term(product.allergens, "花生"))
        if route == "graphrag" and index % 3 == 0:
            question = f"{product.name} 有哪些配料？"
        if route == "text2sql" and index % 3 == 0:
            question = f"{product.brand or 'Demo Foods'} 品牌有多少种产品？"
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "你是 GustoBot-v2 的结构化 Router。"},
                    {"role": "user", "content": json.dumps({"question": question, "input_features": {}}, ensure_ascii=False)},
                ],
                "output": _route_decision(route, confidence, slots),
            }
        )
    return rows


def _build_answer_sft(products: list[FoodProduct], count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    product_cycle = _cycle(products)
    for index in range(count):
        product = next(product_cycle)
        evidence = _product_evidence(product)
        question = rng.choice(
            [
                f"{product.name} 的营养标签是什么？",
                f"{product.name} 有哪些过敏原？",
                f"{product.name} 属于什么类别？",
            ]
        )
        rows.append(
            {
                "instruction": "根据给定 Evidence 回答，必须引用 source_id，不要编造 Evidence 之外的信息。",
                "input": {"question": question, "evidence": [evidence]},
                "output": _answer_from_product(question, product, evidence["source_id"]),
            }
        )
    return rows


def _eval(
    sample_id: str,
    message: str,
    route: str,
    sources: list[str],
    *,
    should_block: bool = False,
    min_evidence_count: int = 1,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "category": route,
        "message": message,
        "expected_route": route,
        "expected_evidence_sources": sources,
        "should_block": should_block,
        "min_evidence_count": min_evidence_count,
    }


def _route_decision(route: str, confidence: float, slots: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_type": route,
        "confidence": confidence,
        "reason": "食品数据底座 Router SFT 样本。",
        "slots": slots,
        "need_clarification": route == "clarify",
    }


def _product_evidence(product: FoodProduct) -> dict[str, Any]:
    source_id = f"food_product:{product.product_id}"
    return {
        "source_type": "kb",
        "source_id": source_id,
        "content": (
            f"{product.name}；品牌：{product.brand or '未知'}；分类：{product.category}；"
            f"过敏原：{product.allergens or '未标注'}；配料：{product.ingredients_text or '未提供'}；"
            f"蛋白质 {value_or_unknown(product.protein)} g；糖 {value_or_unknown(product.sugars)} g。"
        ),
    }


def _answer_from_product(question: str, product: FoodProduct, source_id: str) -> str:
    if "过敏原" in question:
        return f"{product.name} 标注的过敏原是：{product.allergens or '未标注'}。来源：{source_id}"
    if "类别" in question or "分类" in question:
        return f"{product.name} 属于 {product.category}。来源：{source_id}"
    return (
        f"{product.name} 的营养信息包括：蛋白质 {value_or_unknown(product.protein)} g，"
        f"糖 {value_or_unknown(product.sugars)} g，能量 {value_or_unknown(product.energy_kcal)} kcal。来源：{source_id}"
    )


def _split_train_dev_test(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    total = len(rows)
    train_end = int(total * 0.8)
    dev_end = int(total * 0.9)
    return {"train": rows[:train_end], "dev": rows[train_end:dev_end], "test": rows[dev_end:]}


def _cycle(products: list[FoodProduct]):
    while True:
        for product in products:
            yield product


def _first_term(text: str, fallback: str) -> str:
    terms = split_terms(text, max_terms=1)
    return terms[0] if terms else fallback


if __name__ == "__main__":
    raise SystemExit(main())

