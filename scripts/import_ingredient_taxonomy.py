"""把 YAML 食材 taxonomy seed 幂等导入 PostgreSQL。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_SEED_FILE = PROJECT_ROOT / "data" / "taxonomy" / "ingredient_categories.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_env_file(_resolve_optional_project_path(args.env_file, label="env-file"))
    _ensure_project_on_path()

    from app.core.config import settings
    from app.graphrag.ingredient_taxonomy import load_taxonomy

    postgres_dsn = args.postgres_dsn or settings.taxonomy_postgres_dsn or settings.text2sql_postgres_dsn or settings.postgres_dsn
    if not postgres_dsn:
        raise SystemExit("缺少 PostgreSQL DSN，无法导入食材 taxonomy。")

    seed_file = _resolve_project_path(args.seed_file, label="seed-file")
    taxonomy = load_taxonomy(seed_path=seed_file, use_cache=False)
    psycopg = _ensure_postgres_driver()
    with psycopg.connect(postgres_dsn) as connection:
        with connection.cursor() as cursor:
            _create_taxonomy_tables(cursor)
            stats = _upsert_taxonomy(cursor, taxonomy.rules, taxonomy.assignments)
        connection.commit()

    _print_json({"seed_file": str(seed_file), **stats, "status": "ok"})
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import ingredient taxonomy YAML seed into PostgreSQL.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_FILE))
    parser.add_argument("--postgres-dsn", default=None)
    parser.add_argument("--seed-file", default=str(DEFAULT_SEED_FILE))
    return parser.parse_args(argv)


def _upsert_taxonomy(cursor: Any, rules: Sequence[Any], assignments: Sequence[Any]) -> dict[str, int]:
    alias_count = 0
    pattern_count = 0
    hierarchy_count = 0
    assignment_count = 0

    for rule in rules:
        cursor.execute(
            """
            INSERT INTO ingredient_categories (slug, name, enabled, priority, metadata)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (slug) DO UPDATE
            SET name = EXCLUDED.name,
                enabled = EXCLUDED.enabled,
                priority = EXCLUDED.priority,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (rule.slug, rule.name, rule.enabled, rule.priority, _json({"source": "yaml_seed"})),
        )

    for rule in rules:
        for index, alias in enumerate(rule.aliases):
            cursor.execute(
                """
                INSERT INTO ingredient_category_aliases (category_slug, alias, enabled, priority)
                VALUES (%s, %s, true, %s)
                ON CONFLICT (category_slug, alias) DO UPDATE
                SET enabled = EXCLUDED.enabled,
                    priority = EXCLUDED.priority,
                    updated_at = now()
                """,
                (rule.slug, alias, rule.priority + index),
            )
            alias_count += 1

        for index, parent_slug in enumerate(rule.parent_slugs):
            cursor.execute(
                """
                INSERT INTO ingredient_category_hierarchy (child_slug, parent_slug, enabled, priority)
                VALUES (%s, %s, true, %s)
                ON CONFLICT (child_slug, parent_slug) DO UPDATE
                SET enabled = EXCLUDED.enabled,
                    priority = EXCLUDED.priority,
                    updated_at = now()
                """,
                (rule.slug, parent_slug, rule.priority + index),
            )
            hierarchy_count += 1

        for index, pattern in enumerate(rule.name_patterns):
            _upsert_pattern(cursor, rule.slug, name_pattern=pattern, source_category_pattern=None, priority=rule.priority + index)
            pattern_count += 1
        for index, pattern in enumerate(rule.source_category_patterns):
            _upsert_pattern(cursor, rule.slug, name_pattern=None, source_category_pattern=pattern, priority=rule.priority + index)
            pattern_count += 1

    for assignment in assignments:
        cursor.execute(
            """
            INSERT INTO ingredient_category_assignments
                (ingredient_name, source_category, category_slug, enabled, priority, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (assignment_key) DO UPDATE
            SET enabled = EXCLUDED.enabled,
                priority = EXCLUDED.priority,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (
                assignment.ingredient_name,
                assignment.source_category,
                assignment.category_slug,
                assignment.enabled,
                assignment.priority,
                _json({"source": "yaml_seed"}),
            ),
        )
        assignment_count += 1

    return {
        "category_count": len(rules),
        "alias_count": alias_count,
        "pattern_count": pattern_count,
        "hierarchy_count": hierarchy_count,
        "assignment_count": assignment_count,
    }


def _upsert_pattern(
    cursor: Any,
    category_slug: str,
    *,
    name_pattern: str | None,
    source_category_pattern: str | None,
    priority: int,
) -> None:
    cursor.execute(
        """
        INSERT INTO ingredient_category_patterns
            (category_slug, name_pattern, source_category_pattern, enabled, priority)
        VALUES (%s, %s, %s, true, %s)
        ON CONFLICT (category_slug, pattern_key) DO UPDATE
        SET enabled = EXCLUDED.enabled,
            priority = EXCLUDED.priority,
            updated_at = now()
        """,
        (category_slug, name_pattern, source_category_pattern, priority),
    )


def _create_taxonomy_tables(cursor: Any) -> None:
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_categories (
            slug text PRIMARY KEY,
            name text NOT NULL,
            enabled boolean NOT NULL DEFAULT true,
            priority integer NOT NULL DEFAULT 100,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_category_aliases (
            category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
            alias text NOT NULL,
            enabled boolean NOT NULL DEFAULT true,
            priority integer NOT NULL DEFAULT 100,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (category_slug, alias)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_category_patterns (
            id bigserial PRIMARY KEY,
            category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
            name_pattern text,
            source_category_pattern text,
            pattern_key text GENERATED ALWAYS AS (coalesce(name_pattern, '') || '|' || coalesce(source_category_pattern, '')) STORED,
            enabled boolean NOT NULL DEFAULT true,
            priority integer NOT NULL DEFAULT 100,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CHECK (name_pattern IS NOT NULL OR source_category_pattern IS NOT NULL),
            UNIQUE (category_slug, pattern_key)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_category_hierarchy (
            child_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
            parent_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
            enabled boolean NOT NULL DEFAULT true,
            priority integer NOT NULL DEFAULT 100,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (child_slug, parent_slug)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredient_category_assignments (
            id bigserial PRIMARY KEY,
            ingredient_name text,
            source_category text,
            category_slug text NOT NULL REFERENCES ingredient_categories(slug) ON DELETE CASCADE,
            assignment_key text GENERATED ALWAYS AS (coalesce(ingredient_name, '') || '|' || coalesce(source_category, '') || '|' || category_slug) STORED,
            enabled boolean NOT NULL DEFAULT true,
            priority integer NOT NULL DEFAULT 0,
            metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            CHECK (ingredient_name IS NOT NULL OR source_category IS NOT NULL),
            UNIQUE (assignment_key)
        )
        """
    )


def _resolve_project_path(path_value: str, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} 必须位于 GustoBot-v2 项目目录内：{resolved}") from exc
    return resolved


def _resolve_optional_project_path(path_value: str, *, label: str) -> Path | None:
    resolved = _resolve_project_path(path_value, label=label)
    return resolved if resolved.exists() else None


def _load_env_file(path: Path | None) -> None:
    if path is None:
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


def _ensure_project_on_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_postgres_driver():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 psycopg，无法写入 PostgreSQL taxonomy。") from exc
    return psycopg


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
