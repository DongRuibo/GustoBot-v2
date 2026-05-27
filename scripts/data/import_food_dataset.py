"""把食品数据底座导入 PostgreSQL，并可选写入 KB。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.food_dataset import PROCESSED_DIR, resolve_project_path  # noqa: E402


DEFAULT_PRODUCTS = PROCESSED_DIR / "products.csv"
DEFAULT_NUTRIENTS = PROCESSED_DIR / "nutrients.csv"
DEFAULT_KB_DOCS = PROCESSED_DIR / "kb_documents.jsonl"
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env_file(resolve_project_path(args.env_file, label="env-file") if args.env_file else None)
    if args.postgres_dsn:
        os.environ["GUSTOBOT_POSTGRES_DSN"] = args.postgres_dsn
        os.environ.setdefault("GUSTOBOT_TEXT2SQL_POSTGRES_DSN", args.postgres_dsn)

    from app.core.config import settings

    dsn = args.postgres_dsn or settings.text2sql_postgres_dsn or settings.postgres_dsn
    if not dsn:
        raise SystemExit("缺少 PostgreSQL DSN，无法导入食品数据。")

    products_path = resolve_project_path(args.products, label="products")
    nutrients_path = resolve_project_path(args.nutrients, label="nutrients")
    if not products_path.exists():
        raise SystemExit(f"找不到 products.csv：{products_path}")
    if not nutrients_path.exists():
        raise SystemExit(f"找不到 nutrients.csv：{nutrients_path}")

    psycopg, dict_row = _ensure_postgres_driver()
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            _ensure_schema(cursor)
            if args.reset:
                cursor.execute("DELETE FROM food_nutrients")
                cursor.execute("DELETE FROM food_products")
            product_count = _upsert_products(cursor, products_path)
            nutrient_count = _upsert_nutrients(cursor, nutrients_path)
            _upsert_schema_catalog(cursor)

    kb_documents = 0
    kb_chunks = 0
    kb_deleted_documents = 0
    if args.ingest_kb:
        if args.reset or args.reset_kb:
            kb_deleted_documents = _delete_food_kb_documents(dsn)
        kb_path = resolve_project_path(args.kb_documents, label="kb-documents")
        kb_documents, kb_chunks = _ingest_kb_documents(kb_path)

    payload = {
        "postgres_dsn_configured": True,
        "products": str(products_path),
        "nutrients": str(nutrients_path),
        "product_count": product_count,
        "nutrient_row_count": nutrient_count,
        "schema_catalog_updated": True,
        "kb_deleted_documents": kb_deleted_documents,
        "kb_ingested_documents": kb_documents,
        "kb_ingested_chunks": kb_chunks,
        "status": "ok",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import food dataset into PostgreSQL.")
    parser.add_argument("--products", default=str(DEFAULT_PRODUCTS))
    parser.add_argument("--nutrients", default=str(DEFAULT_NUTRIENTS))
    parser.add_argument("--kb-documents", default=str(DEFAULT_KB_DOCS))
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--ingest-kb", action="store_true")
    parser.add_argument("--reset-kb", action="store_true", help="导入 KB 前清理旧的 food_dataset 文档。")
    return parser.parse_args(argv)


def _ensure_schema(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_catalog (
            table_name text PRIMARY KEY,
            table_comment text NOT NULL,
            business_meaning text NOT NULL,
            module text NOT NULL,
            columns jsonb NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS food_products (
            product_id text PRIMARY KEY,
            name text NOT NULL,
            brand text,
            category text NOT NULL,
            country text,
            ingredients_text text,
            allergens text,
            energy_kcal numeric(12,4),
            protein numeric(12,4),
            fat numeric(12,4),
            carbohydrates numeric(12,4),
            sugars numeric(12,4),
            salt numeric(12,4),
            source text NOT NULL,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS food_nutrients (
            product_id text NOT NULL REFERENCES food_products(product_id) ON DELETE CASCADE,
            nutrient_name text NOT NULL,
            value numeric(12,4) NOT NULL,
            unit text NOT NULL,
            source text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (product_id, nutrient_name)
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_products_category ON food_products(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_products_brand ON food_products(brand)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_food_products_source ON food_products(source)")


def _upsert_products(cursor: Any, path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cursor.execute(
                """
                INSERT INTO food_products
                    (product_id, name, brand, category, country, ingredients_text, allergens,
                     energy_kcal, protein, fat, carbohydrates, sugars, salt, source, metadata)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (product_id) DO UPDATE
                SET name = EXCLUDED.name,
                    brand = EXCLUDED.brand,
                    category = EXCLUDED.category,
                    country = EXCLUDED.country,
                    ingredients_text = EXCLUDED.ingredients_text,
                    allergens = EXCLUDED.allergens,
                    energy_kcal = EXCLUDED.energy_kcal,
                    protein = EXCLUDED.protein,
                    fat = EXCLUDED.fat,
                    carbohydrates = EXCLUDED.carbohydrates,
                    sugars = EXCLUDED.sugars,
                    salt = EXCLUDED.salt,
                    source = EXCLUDED.source,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                """,
                (
                    row["product_id"],
                    row["name"],
                    row.get("brand") or None,
                    row["category"],
                    row.get("country") or None,
                    row.get("ingredients_text") or None,
                    row.get("allergens") or None,
                    _nullable_float(row.get("energy_kcal")),
                    _nullable_float(row.get("protein")),
                    _nullable_float(row.get("fat")),
                    _nullable_float(row.get("carbohydrates")),
                    _nullable_float(row.get("sugars")),
                    _nullable_float(row.get("salt")),
                    row["source"],
                    json.dumps({"source_script": "import_food_dataset"}, ensure_ascii=False),
                ),
            )
            count += 1
    return count


def _upsert_nutrients(cursor: Any, path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            cursor.execute(
                """
                INSERT INTO food_nutrients (product_id, nutrient_name, value, unit, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (product_id, nutrient_name) DO UPDATE
                SET value = EXCLUDED.value,
                    unit = EXCLUDED.unit,
                    source = EXCLUDED.source,
                    updated_at = now()
                """,
                (
                    row["product_id"],
                    row["nutrient_name"],
                    _nullable_float(row["value"]),
                    row["unit"],
                    row["source"],
                ),
            )
            count += 1
    return count


def _upsert_schema_catalog(cursor: Any) -> None:
    rows = [
        (
            "food_products",
            "食品商品主表，保存商品名称、品牌、分类、配料、过敏原和常用营养标签。",
            "用于食品商品筛选、品牌统计、分类统计、糖分/蛋白质/能量排序和营养标签查询。",
            "food_analytics",
            [
                ("product_id", "text", "商品唯一编号，带 source 前缀"),
                ("name", "text", "商品或标准食物名称"),
                ("brand", "text", "品牌"),
                ("category", "text", "食品分类"),
                ("country", "text", "国家或地区"),
                ("ingredients_text", "text", "配料文本"),
                ("allergens", "text", "过敏原文本"),
                ("energy_kcal", "numeric", "每 100g 能量 kcal"),
                ("protein", "numeric", "每 100g 蛋白质 g"),
                ("fat", "numeric", "每 100g 脂肪 g"),
                ("carbohydrates", "numeric", "每 100g 碳水 g"),
                ("sugars", "numeric", "每 100g 糖 g"),
                ("salt", "numeric", "每 100g 盐 g"),
                ("source", "text", "数据来源"),
            ],
        ),
        (
            "food_nutrients",
            "食品营养素长表，每个商品每种营养素一行。",
            "用于扩展营养素明细、按营养素名称过滤和与 food_products 关联分析。",
            "food_analytics",
            [
                ("product_id", "text", "商品唯一编号"),
                ("nutrient_name", "text", "营养素名称"),
                ("value", "numeric", "营养素值"),
                ("unit", "text", "单位"),
                ("source", "text", "数据来源"),
            ],
        ),
    ]
    for table_name, comment, meaning, module, columns in rows:
        cursor.execute(
            """
            INSERT INTO schema_catalog
                (table_name, table_comment, business_meaning, module, columns, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (table_name) DO UPDATE
            SET table_comment = EXCLUDED.table_comment,
                business_meaning = EXCLUDED.business_meaning,
                module = EXCLUDED.module,
                columns = EXCLUDED.columns,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                table_name,
                comment,
                meaning,
                module,
                json.dumps(
                    [
                        {"name": name, "data_type": data_type, "comment": col_comment, "sample_values": []}
                        for name, data_type, col_comment in columns
                    ],
                    ensure_ascii=False,
                ),
                json.dumps({"source": "food_dataset_import"}, ensure_ascii=False),
            ),
        )


def _ingest_kb_documents(path: Path) -> tuple[int, int]:
    if not path.exists():
        raise SystemExit(f"找不到 KB 文档 JSONL：{path}")
    from app.kb.service import get_kb_service

    service = get_kb_service()
    document_count = 0
    chunk_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        result = service.ingest_document(
            title=payload["title"],
            content=payload["content"],
            source_id=payload.get("source_id"),
            metadata=payload.get("metadata", {}),
        )
        document_count += 1
        chunk_count += result.chunk_count
    return document_count, chunk_count


def _delete_food_kb_documents(dsn: str) -> int:
    psycopg, dict_row = _ensure_postgres_driver()
    with psycopg.connect(dsn, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM kb_documents WHERE metadata->>'source' = 'food_dataset' RETURNING document_id")
            return len(cursor.fetchall())


def _nullable_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_postgres_driver():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 psycopg，无法导入 PostgreSQL。") from exc
    return psycopg, dict_row


if __name__ == "__main__":
    raise SystemExit(main())
