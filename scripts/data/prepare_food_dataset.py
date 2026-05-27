"""把 USDA + FoodOn 原始下载转换成 GustoBot-v2 可消费的 processed 数据。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    FoodOnTerm,
    FoodProduct,
    PROCESSED_DIR,
    build_graph,
    build_kb_documents,
    clean_text,
    dedupe_products,
    ensure_parent,
    parse_float,
    product_from_usda,
    write_graph_csv,
    write_jsonl,
    write_nutrients_csv,
    write_products_csv,
)


DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "external"
DEFAULT_OUTPUT_DIR = PROCESSED_DIR
GRAPH_NUTRIENT_COLUMNS = ("energy_kcal", "protein", "fat", "carbohydrates")


@dataclass(slots=True)
class SourceResult:
    name: str
    path: Path
    selected_count: int
    food_row_count: int
    product_count: int
    products: list[FoodProduct]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    raw_root = _resolve_path(args.raw_root)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usda_root = raw_root / "usda"
    foodon_root = raw_root / "foodon"
    foundation_path = _find_usda_source(usda_root, "foundation")
    sr_path = _find_usda_source(usda_root, "sr_legacy")
    branded_zip = _find_usda_source(usda_root, "branded", prefer_zip=True)
    foodon_tsv = foodon_root / "foodon-synonyms.tsv"
    foodon_owl = foodon_root / "foodon.owl"

    source_results: list[SourceResult] = []
    source_results.append(_prepare_foundation(foundation_path))
    source_results.append(_prepare_sr_legacy(sr_path, limit=args.sr_limit))
    source_results.append(_prepare_branded(branded_zip, limit=args.branded_limit))

    products = dedupe_products(product for result in source_results for product in result.products)
    foodon_terms, foodon_stats = _read_foodon_terms(foodon_tsv, limit=args.max_foodon_terms)
    graph_products = products[: args.graph_product_limit] if args.graph_product_limit > 0 else products

    food_products_path = output_dir / "food_products.csv"
    food_nutrients_path = output_dir / "food_nutrients.csv"
    kb_documents_path = output_dir / "kb_documents.jsonl"
    graph_nodes_path = output_dir / "graph_nodes.csv"
    graph_edges_path = output_dir / "graph_edges.csv"
    manifest_path = output_dir / "dataset_manifest.json"
    stats_path = output_dir / "food_dataset_stats.json"

    product_count = write_products_csv(food_products_path, products)
    nutrient_count = write_nutrients_csv(food_nutrients_path, products)
    nodes, edges = build_graph(
        graph_products,
        foodon_terms=foodon_terms,
        max_ingredients_per_product=args.graph_max_ingredients,
        nutrient_columns=GRAPH_NUTRIENT_COLUMNS,
    )
    write_graph_csv(graph_nodes_path, graph_edges_path, nodes, edges)
    documents = build_kb_documents(
        products,
        max_product_docs=args.max_kb_products,
        foodon_terms=foodon_terms,
        max_foodon_docs=args.max_foodon_docs,
    )
    kb_document_count = write_jsonl(kb_documents_path, documents)

    # 兼容旧脚本默认入参，避免这一步改动问答/导入链路。
    _copy_alias(food_products_path, output_dir / "products.csv")
    _copy_alias(food_nutrients_path, output_dir / "nutrients.csv")

    manifest = _build_manifest(
        raw_root=raw_root,
        output_dir=output_dir,
        source_results=source_results,
        products=products,
        nutrient_count=nutrient_count,
        graph_node_count=len(nodes),
        graph_edge_count=len(edges),
        kb_document_count=kb_document_count,
        foodon_stats=foodon_stats,
        foodon_owl=foodon_owl,
        output_paths={
            "food_products": food_products_path,
            "food_nutrients": food_nutrients_path,
            "kb_documents": kb_documents_path,
            "graph_nodes": graph_nodes_path,
            "graph_edges": graph_edges_path,
            "dataset_manifest": manifest_path,
            "food_dataset_stats": stats_path,
            "products_alias": output_dir / "products.csv",
            "nutrients_alias": output_dir / "nutrients.csv",
        },
        args=args,
        product_count=product_count,
    )
    ensure_parent(manifest_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats_path.write_text(json.dumps(_stats_from_manifest(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare official USDA + FoodOn food dataset for GustoBot-v2.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sr-limit", type=int, default=8000)
    parser.add_argument("--branded-limit", type=int, default=8000)
    parser.add_argument("--graph-product-limit", type=int, default=12000)
    parser.add_argument("--graph-max-ingredients", type=int, default=3)
    parser.add_argument("--max-kb-products", type=int, default=2000)
    parser.add_argument("--max-foodon-terms", type=int, default=1200)
    parser.add_argument("--max-foodon-docs", type=int, default=400)
    return parser.parse_args(argv)


def _prepare_foundation(path: Path) -> SourceResult:
    with _CsvSource(path) as source:
        selected_ids = _read_id_table(source, "foundation_food.csv", limit=0)
        return _load_usda_products("usda_foundation", path, source, selected_ids)


def _prepare_sr_legacy(path: Path, *, limit: int) -> SourceResult:
    with _CsvSource(path) as source:
        selected_ids = _read_id_table(source, "sr_legacy_food.csv", limit=limit)
        return _load_usda_products("usda_sr_legacy", path, source, selected_ids)


def _prepare_branded(path: Path, *, limit: int) -> SourceResult:
    with _CsvSource(path) as source:
        branded_rows = _select_branded_rows(source, limit=limit)
        selected_ids = list(branded_rows)
        return _load_usda_products("usda_branded", path, source, selected_ids, branded_rows=branded_rows)


def _load_usda_products(
    source_name: str,
    path: Path,
    source: "_CsvSource",
    selected_ids: list[str],
    *,
    branded_rows: dict[str, dict[str, str]] | None = None,
) -> SourceResult:
    selected_set = set(selected_ids)
    food_rows = _read_food_rows(source, selected_ids)
    nutrient_names = _read_nutrient_names(source)
    common_nutrients, detailed_nutrients = _read_food_nutrients(source, selected_set, nutrient_names)
    branded = branded_rows if branded_rows is not None else _read_branded_food(source, selected_set)
    categories = _read_food_categories(source)

    products: list[FoodProduct] = []
    for row in food_rows:
        fdc_id = clean_text(row.get("fdc_id"))
        brand_info = branded.get(fdc_id, {})
        category = clean_text(
            brand_info.get("branded_food_category")
            or categories.get(row.get("food_category_id", ""))
            or row.get("data_type")
        )
        product = product_from_usda(
            row,
            common_nutrients.get(fdc_id, {}),
            brand=clean_text(brand_info.get("brand_owner") or brand_info.get("brand_name")),
            ingredients=clean_text(brand_info.get("ingredients")),
            category=category,
        )
        if product is None:
            continue
        product.source = source_name
        product.metadata.update({"fdc_id": fdc_id, "data_type": row.get("data_type", "")})
        product.nutrient_values = detailed_nutrients.get(fdc_id, {})
        products.append(product)

    return SourceResult(
        name=source_name,
        path=path,
        selected_count=len(selected_ids),
        food_row_count=len(food_rows),
        product_count=len(products),
        products=products,
    )


def _read_id_table(source: "_CsvSource", name: str, *, limit: int) -> list[str]:
    ids: list[str] = []
    for row in source.iter_csv(name):
        fdc_id = clean_text(row.get("fdc_id"))
        if not fdc_id:
            continue
        ids.append(fdc_id)
        if limit > 0 and len(ids) >= limit:
            break
    return ids


def _select_branded_rows(source: "_CsvSource", *, limit: int) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    for row in source.iter_csv("branded_food.csv"):
        fdc_id = clean_text(row.get("fdc_id"))
        if not fdc_id:
            continue
        if not _branded_row_has_signal(row):
            continue
        selected[fdc_id] = row
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def _branded_row_has_signal(row: dict[str, str]) -> bool:
    return bool(
        clean_text(row.get("branded_food_category"))
        or clean_text(row.get("ingredients"))
        or clean_text(row.get("brand_owner"))
        or clean_text(row.get("brand_name"))
    )


def _read_food_rows(source: "_CsvSource", selected_ids: list[str]) -> list[dict[str, str]]:
    selected_set = set(selected_ids)
    rows_by_id: dict[str, dict[str, str]] = {}
    for row in source.iter_csv("food.csv"):
        fdc_id = clean_text(row.get("fdc_id"))
        if fdc_id not in selected_set:
            continue
        rows_by_id[fdc_id] = row
        if len(rows_by_id) >= len(selected_set):
            break
    return [rows_by_id[fdc_id] for fdc_id in selected_ids if fdc_id in rows_by_id]


def _read_nutrient_names(source: "_CsvSource") -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    for row in source.iter_csv("nutrient.csv"):
        nutrient_id = clean_text(row.get("id") or row.get("nutrient_id"))
        if nutrient_id:
            names[nutrient_id] = (clean_text(row.get("name")), clean_text(row.get("unit_name")))
    return names


def _read_food_nutrients(
    source: "_CsvSource",
    selected_ids: set[str],
    nutrient_names: dict[str, tuple[str, str]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, str]]]]:
    common: dict[str, dict[str, float]] = {}
    details: dict[str, dict[str, tuple[float, str]]] = {}
    for row in source.iter_csv("food_nutrient.csv"):
        fdc_id = clean_text(row.get("fdc_id"))
        if fdc_id not in selected_ids:
            continue
        amount = parse_float(row.get("amount"))
        if amount is None:
            continue
        name, unit = nutrient_names.get(clean_text(row.get("nutrient_id")), ("", ""))
        if not name or not unit:
            continue
        common_key = _common_nutrient_key(name, unit)
        if common_key:
            common.setdefault(fdc_id, {}).setdefault(common_key, amount)
        detail_key = _detail_nutrient_key(name, unit)
        if detail_key:
            details.setdefault(fdc_id, {}).setdefault(detail_key, (amount, _detail_unit(detail_key, unit)))
    return common, details


def _read_branded_food(source: "_CsvSource", selected_ids: set[str]) -> dict[str, dict[str, str]]:
    branded = {}
    if not source.exists("branded_food.csv"):
        return branded
    for row in source.iter_csv("branded_food.csv"):
        fdc_id = clean_text(row.get("fdc_id"))
        if fdc_id in selected_ids:
            branded[fdc_id] = row
            if len(branded) >= len(selected_ids):
                break
    return branded


def _read_food_categories(source: "_CsvSource") -> dict[str, str]:
    categories = {}
    if not source.exists("food_category.csv"):
        return categories
    for row in source.iter_csv("food_category.csv"):
        category_id = clean_text(row.get("id") or row.get("food_category_id"))
        if category_id:
            categories[category_id] = clean_text(row.get("description") or row.get("code"))
    return categories


def _common_nutrient_key(name: str, unit: str) -> str | None:
    lowered = name.lower()
    unit_lower = unit.lower()
    if lowered.startswith("energy") and unit_lower == "kcal":
        return "energy_kcal"
    if lowered == "protein" or lowered.startswith("protein"):
        return "protein"
    if "total lipid" in lowered or lowered == "fat":
        return "fat"
    if lowered.startswith("carbohydrate"):
        return "carbohydrates"
    if "sugars" in lowered:
        return "sugars"
    if lowered.startswith("sodium"):
        return "sodium_mg"
    return None


def _detail_nutrient_key(name: str, unit: str) -> str | None:
    lowered = name.lower()
    unit_lower = unit.lower()
    if lowered.startswith("energy") and unit_lower == "kcal":
        return "energy_kcal"
    if lowered.startswith("protein"):
        return "protein"
    if "total lipid" in lowered or lowered == "fat":
        return "fat"
    if "fatty acids, total saturated" in lowered:
        return "saturated_fat"
    if "fatty acids, total trans" in lowered:
        return "trans_fat"
    if lowered.startswith("carbohydrate"):
        return "carbohydrates"
    if "sugars" in lowered:
        return "sugars"
    if lowered.startswith("fiber"):
        return "fiber"
    if lowered.startswith("sodium"):
        return "sodium"
    if lowered.startswith("calcium"):
        return "calcium"
    if lowered.startswith("iron"):
        return "iron"
    if lowered.startswith("potassium"):
        return "potassium"
    if lowered.startswith("cholesterol"):
        return "cholesterol"
    if lowered.startswith("vitamin c"):
        return "vitamin_c"
    if lowered.startswith("vitamin d"):
        return "vitamin_d"
    return None


def _detail_unit(detail_key: str, unit: str) -> str:
    if detail_key == "sodium":
        return "mg"
    return clean_text(unit).lower()


def _read_foodon_terms(path: Path, *, limit: int) -> tuple[list[FoodOnTerm], dict[str, Any]]:
    if not path.exists():
        return [], {"path": str(path), "exists": False, "term_count": 0, "edge_count": 0}

    labels: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    raw_rows = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            raw_rows += 1
            term_id = _clean_foodon_id(row.get("?class") or row.get("class") or "")
            if not term_id or "FOODON_" not in term_id:
                continue
            parent_id = _clean_foodon_id(row.get("?parent") or row.get("parent") or "")
            row_type = clean_text(row.get("?type") or row.get("type")).strip('"').lower()
            label = _clean_foodon_label(row.get("?label") or row.get("label") or "")
            if parent_id and "FOODON_" in parent_id and term_id not in parents:
                parents[term_id] = parent_id
            if not label:
                continue
            if row_type == "label" and term_id not in labels:
                labels[term_id] = label
            elif "synonym" in row_type or "alternative" in row_type:
                aliases.setdefault(term_id, [])
                if label not in aliases[term_id]:
                    aliases[term_id].append(label)

    terms: list[FoodOnTerm] = []
    for term_id, label in labels.items():
        terms.append(FoodOnTerm(term_id=term_id, label=label, parent_id=parents.get(term_id, ""), aliases=aliases.get(term_id, [])))
        if limit > 0 and len(terms) >= limit:
            break
    edge_count = sum(1 for term in terms if term.parent_id)
    return terms, {
        "path": str(path),
        "exists": True,
        "raw_rows": raw_rows,
        "term_count": len(terms),
        "edge_count": edge_count,
        "alias_count": sum(len(term.aliases) for term in terms),
    }


def _clean_foodon_id(value: str) -> str:
    text = clean_text(value).strip("<>").strip()
    return text


def _clean_foodon_label(value: str) -> str:
    text = clean_text(value).strip('"')
    if "@" in text:
        text = text.rsplit("@", 1)[0].strip('"')
    return text.strip()


def _build_manifest(
    *,
    raw_root: Path,
    output_dir: Path,
    source_results: list[SourceResult],
    products: list[FoodProduct],
    nutrient_count: int,
    graph_node_count: int,
    graph_edge_count: int,
    kb_document_count: int,
    foodon_stats: dict[str, Any],
    foodon_owl: Path,
    output_paths: dict[str, Path],
    args: argparse.Namespace,
    product_count: int,
) -> dict[str, Any]:
    source_counts = Counter(product.source for product in products)
    return {
        "status": "ok",
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "product_count": product_count,
        "nutrient_row_count": nutrient_count,
        "graph_node_count": graph_node_count,
        "graph_edge_count": graph_edge_count,
        "kb_document_count": kb_document_count,
        "field_missing_rate": _field_missing_rates(products),
        "source_counts": dict(sorted(source_counts.items())),
        "source_ratio": {
            source: round(count / product_count, 4) if product_count else 0
            for source, count in sorted(source_counts.items())
        },
        "source_inputs": [
            {
                "name": result.name,
                "path": str(result.path),
                "selected_count": result.selected_count,
                "food_row_count": result.food_row_count,
                "product_count": result.product_count,
            }
            for result in source_results
        ],
        "foodon": {
            **foodon_stats,
            "owl": {
                "path": str(foodon_owl),
                "exists": foodon_owl.exists(),
                "bytes": foodon_owl.stat().st_size if foodon_owl.exists() else 0,
            },
        },
        "limits": {
            "sr_limit": args.sr_limit,
            "branded_limit": args.branded_limit,
            "graph_product_limit": args.graph_product_limit,
            "graph_max_ingredients": args.graph_max_ingredients,
            "max_kb_products": args.max_kb_products,
            "max_foodon_terms": args.max_foodon_terms,
            "max_foodon_docs": args.max_foodon_docs,
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
    }


def _stats_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_count": manifest["product_count"],
        "nutrient_row_count": manifest["nutrient_row_count"],
        "graph_node_count": manifest["graph_node_count"],
        "graph_edge_count": manifest["graph_edge_count"],
        "kb_document_count": manifest["kb_document_count"],
        "source_counts": manifest["source_counts"],
        "products_output": manifest["outputs"]["products_alias"],
        "nutrients_output": manifest["outputs"]["nutrients_alias"],
        "graph_nodes_output": manifest["outputs"]["graph_nodes"],
        "graph_edges_output": manifest["outputs"]["graph_edges"],
        "kb_documents_output": manifest["outputs"]["kb_documents"],
        "manifest_output": manifest["outputs"]["dataset_manifest"],
        "status": manifest["status"],
    }


def _field_missing_rates(products: list[FoodProduct]) -> dict[str, float]:
    if not products:
        return {}
    fields = [
        "brand",
        "category",
        "ingredients_text",
        "energy_kcal",
        "protein",
        "fat",
        "carbohydrates",
        "sugars",
        "salt",
    ]
    rates: dict[str, float] = {}
    for field in fields:
        missing = 0
        for product in products:
            value = getattr(product, field)
            if value is None or value == "":
                missing += 1
        rates[field] = round(missing / len(products), 4)
    return rates


def _find_usda_source(usda_root: Path, kind: str, *, prefer_zip: bool = False) -> Path:
    patterns = {
        "foundation": ("foundation_food_csv_*", "*foundation*_csv*.zip"),
        "sr_legacy": ("sr_legacy_food_csv_*", "*sr_legacy*_csv*.zip"),
        "branded": ("branded_food_csv_*", "*branded*_csv*.zip"),
    }[kind]
    candidates: list[Path] = []
    if prefer_zip:
        candidates.extend(sorted(usda_root.glob(patterns[1])))
        candidates.extend(sorted(path for path in usda_root.glob(patterns[0]) if path.is_dir()))
    else:
        candidates.extend(sorted(path for path in usda_root.glob(patterns[0]) if path.is_dir()))
        candidates.extend(sorted(usda_root.glob(patterns[1])))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"找不到 USDA {kind} 原始数据：{usda_root}")


def _copy_alias(source: Path, target: Path) -> None:
    ensure_parent(target)
    shutil.copyfile(source, target)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class _CsvSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None

    def __enter__(self) -> "_CsvSource":
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._zip is not None:
            self._zip.close()

    def exists(self, name: str) -> bool:
        if self._zip is not None:
            return self._zip_name(name) is not None
        return self._dir_csv_path(name) is not None

    def iter_csv(self, name: str) -> Iterable[dict[str, str]]:
        if self._zip is not None:
            zip_name = self._zip_name(name)
            if zip_name is None:
                return
            with self._zip.open(zip_name, "r") as raw_file:
                text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
                yield from csv.DictReader(text_file)
            return
        csv_path = self._dir_csv_path(name)
        if csv_path is None:
            return
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            yield from csv.DictReader(file)

    def _zip_name(self, name: str) -> str | None:
        if self._zip is None:
            return None
        target = name.lower()
        for candidate in self._zip.namelist():
            if Path(candidate).name.lower() == target:
                return candidate
        return None

    def _dir_csv_path(self, name: str) -> Path | None:
        direct = self.path / name
        if direct.exists():
            return direct
        matches = sorted(self.path.rglob(name))
        return matches[0] if matches else None


if __name__ == "__main__":
    raise SystemExit(main())
