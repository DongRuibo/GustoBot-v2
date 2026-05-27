"""清洗 USDA FoodData Central CSV 下载包，输出统一食品商品 CSV。"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import (  # noqa: E402
    PROCESSED_DIR,
    RAW_DIR,
    SAMPLE_PRODUCTS,
    clean_text,
    dedupe_products,
    product_from_usda,
    resolve_project_path,
    write_nutrients_csv,
    write_products_csv,
)


DEFAULT_INPUT = RAW_DIR / "usda" / "FoodData_Central_csv.zip"
DEFAULT_PRODUCTS_OUTPUT = PROCESSED_DIR / "products_usda.csv"
DEFAULT_NUTRIENTS_OUTPUT = PROCESSED_DIR / "nutrients_usda.csv"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = resolve_project_path(args.output, label="output")
    nutrients_output = resolve_project_path(args.nutrients_output, label="nutrients-output")

    if args.sample:
        products = [product for product in SAMPLE_PRODUCTS if product.product_id in {"sample:oat_milk", "sample:yogurt"}]
    else:
        input_path = resolve_project_path(args.input, label="input")
        if not input_path.exists():
            raise SystemExit(f"找不到 USDA 原始文件：{input_path}。请从 FoodData Central 下载 CSV 包，或加 --sample 做链路自检。")
        products = _load_usda_products(input_path, limit=args.limit)

    products = dedupe_products(products)
    product_count = write_products_csv(output, products)
    nutrient_count = write_nutrients_csv(nutrients_output, products)
    print(
        json.dumps(
            {
                "products_output": str(output),
                "nutrients_output": str(nutrients_output),
                "product_count": product_count,
                "nutrient_row_count": nutrient_count,
                "status": "ok",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare USDA FoodData Central CSV subset for GustoBot-v2.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="USDA CSV zip 或解压后的目录。")
    parser.add_argument("--output", default=str(DEFAULT_PRODUCTS_OUTPUT))
    parser.add_argument("--nutrients-output", default=str(DEFAULT_NUTRIENTS_OUTPUT))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--sample", action="store_true", help="使用内置小样本生成输出，便于无网络自检。")
    return parser.parse_args(argv)


def _load_usda_products(path: Path, *, limit: int) -> list:
    source = _CsvSource(path)
    food_rows = _read_limited_rows(source, "food.csv", limit=limit)
    selected_ids = {row["fdc_id"] for row in food_rows if row.get("fdc_id")}
    nutrient_names = _read_nutrient_names(source)
    nutrient_values = _read_food_nutrients(source, selected_ids, nutrient_names)
    branded = _read_branded_food(source, selected_ids)
    categories = _read_food_categories(source)

    products = []
    for row in food_rows:
        fdc_id = row.get("fdc_id", "")
        brand_info = branded.get(fdc_id, {})
        category = clean_text(
            brand_info.get("branded_food_category")
            or categories.get(row.get("food_category_id", ""))
            or row.get("data_type")
        )
        product = product_from_usda(
            row,
            nutrient_values.get(fdc_id, {}),
            brand=clean_text(brand_info.get("brand_owner") or brand_info.get("brand_name")),
            ingredients=clean_text(brand_info.get("ingredients")),
            category=category,
        )
        if product is not None:
            products.append(product)
    return products


def _read_limited_rows(source: "_CsvSource", name: str, *, limit: int) -> list[dict[str, str]]:
    rows = []
    for row in source.iter_csv(name):
        if limit > 0 and len(rows) >= limit:
            break
        rows.append(row)
    return rows


def _read_nutrient_names(source: "_CsvSource") -> dict[str, tuple[str, str]]:
    names = {}
    for row in source.iter_csv("nutrient.csv"):
        nutrient_id = row.get("id") or row.get("nutrient_id")
        if nutrient_id:
            names[nutrient_id] = (clean_text(row.get("name")), clean_text(row.get("unit_name")))
    return names


def _read_food_nutrients(
    source: "_CsvSource",
    selected_ids: set[str],
    nutrient_names: dict[str, tuple[str, str]],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for row in source.iter_csv("food_nutrient.csv"):
        fdc_id = row.get("fdc_id", "")
        if fdc_id not in selected_ids:
            continue
        key = _nutrient_key(nutrient_names.get(row.get("nutrient_id", ""), ("", "")))
        if key is None:
            continue
        try:
            amount = float(row.get("amount") or "")
        except ValueError:
            continue
        values.setdefault(fdc_id, {})[key] = amount
    return values


def _read_branded_food(source: "_CsvSource", selected_ids: set[str]) -> dict[str, dict[str, str]]:
    branded = {}
    if not source.exists("branded_food.csv"):
        return branded
    for row in source.iter_csv("branded_food.csv"):
        fdc_id = row.get("fdc_id", "")
        if fdc_id in selected_ids:
            branded[fdc_id] = row
    return branded


def _read_food_categories(source: "_CsvSource") -> dict[str, str]:
    categories = {}
    if not source.exists("food_category.csv"):
        return categories
    for row in source.iter_csv("food_category.csv"):
        category_id = row.get("id") or row.get("food_category_id")
        if category_id:
            categories[category_id] = clean_text(row.get("description") or row.get("code"))
    return categories


def _nutrient_key(name_and_unit: tuple[str, str]) -> str | None:
    name, unit = name_and_unit
    lowered = name.lower()
    unit_lower = unit.lower()
    if lowered.startswith("energy") and unit_lower == "kcal":
        return "energy_kcal"
    if "protein" in lowered:
        return "protein"
    if "total lipid" in lowered or lowered == "fat":
        return "fat"
    if "carbohydrate" in lowered:
        return "carbohydrates"
    if "sugars" in lowered:
        return "sugars"
    if lowered.startswith("sodium"):
        return "sodium_mg"
    return None


class _CsvSource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._zip: zipfile.ZipFile | None = zipfile.ZipFile(path) if path.suffix.lower() == ".zip" else None

    def exists(self, name: str) -> bool:
        if self._zip is not None:
            return self._zip_name(name) is not None
        return (self.path / name).exists()

    def iter_csv(self, name: str):
        if self._zip is not None:
            zip_name = self._zip_name(name)
            if zip_name is None:
                return
            with self._zip.open(zip_name, "r") as raw_file:
                text_file = io.TextIOWrapper(raw_file, encoding="utf-8-sig", newline="")
                yield from csv.DictReader(text_file)
            return
        csv_path = self.path / name
        if not csv_path.exists():
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


if __name__ == "__main__":
    raise SystemExit(main())
