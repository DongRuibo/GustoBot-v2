"""GraphRAG 食材类别与上位词归一化规则。

生产环境优先从 PostgreSQL taxonomy 表读取；本地开发和测试可退回 YAML seed 或内置 seed。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CATEGORY_LABEL = "IngredientCategory"
BELONGS_TO_CATEGORY = "BELONGS_TO_CATEGORY"
IS_A_CATEGORY = "IS_A"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_PATH = PROJECT_ROOT / "data" / "taxonomy" / "ingredient_categories.yaml"


@dataclass(frozen=True, slots=True)
class IngredientCategoryRule:
    slug: str
    name: str
    aliases: tuple[str, ...] = ()
    parent_slugs: tuple[str, ...] = ()
    name_patterns: tuple[str, ...] = ()
    source_category_patterns: tuple[str, ...] = ()
    enabled: bool = True
    priority: int = 100

    @property
    def node_id(self) -> str:
        return f"ingredient_category:{self.slug}"


@dataclass(frozen=True, slots=True)
class IngredientCategoryAssignment:
    ingredient_name: str | None
    source_category: str | None
    category_slug: str
    enabled: bool = True
    priority: int = 0


@dataclass(frozen=True, slots=True)
class IngredientTaxonomyData:
    rules: tuple[IngredientCategoryRule, ...]
    assignments: tuple[IngredientCategoryAssignment, ...] = ()
    source: str = "builtin"


_TAXONOMY_CACHE: IngredientTaxonomyData | None = None


_BUILTIN_SEED: dict[str, Any] = {
    "categories": [
        {
            "slug": "meat",
            "name": "肉类",
            "aliases": ["肉", "荤菜"],
            "source_category_patterns": ["肉类"],
            "priority": 10,
        },
        {
            "slug": "vegetable",
            "name": "蔬菜",
            "aliases": ["素菜", "青蔬"],
            "source_category_patterns": ["蔬菜"],
            "priority": 20,
        },
        {
            "slug": "seafood",
            "name": "海鲜",
            "aliases": ["水产", "水产品"],
            "source_category_patterns": ["海鲜", "水产"],
            "priority": 30,
        },
        {
            "slug": "pork",
            "name": "猪肉",
            "aliases": ["大肉"],
            "parent_slugs": ["meat"],
            "name_patterns": ["猪", "五花肉", "排骨", "肋排", "小排", "仔排", "肘子", "猪肝", "猪腰", "猪心", "肥肠", "^(?!.*[牛羊鸡]).*(里脊|通脊).*"],
            "priority": 40,
        },
        {
            "slug": "lean_meat",
            "name": "瘦肉",
            "aliases": ["精瘦肉", "纯瘦肉", "瘦肉类"],
            "parent_slugs": ["meat"],
            "name_patterns": ["瘦肉", "精瘦", "纯瘦", "猪肉里脊", "猪里脊", "通脊", "^(?!.*[牛羊鸡]).*里脊.*"],
            "priority": 45,
        },
        {
            "slug": "beef",
            "name": "牛肉",
            "aliases": ["肥牛"],
            "parent_slugs": ["meat"],
            "name_patterns": ["牛肉", "牛腩", "牛里脊", "牛排", "肥牛", "牛肚", "牛柳", "牛腱"],
            "priority": 50,
        },
        {
            "slug": "chicken",
            "name": "鸡肉",
            "aliases": ["鸡胸肉", "鸡腿肉"],
            "parent_slugs": ["meat"],
            "name_patterns": ["鸡肉", "鸡胸", "鸡腿", "鸡翅", "鸡丁", "鸡柳", "鸡排"],
            "priority": 60,
        },
        {
            "slug": "fish",
            "name": "鱼类",
            "aliases": ["鱼", "鱼肉"],
            "parent_slugs": ["seafood"],
            "name_patterns": ["鱼", "三文鱼", "鳕鱼", "鲈鱼", "草鱼", "鲫鱼", "龙利鱼"],
            "priority": 70,
        },
        {
            "slug": "shrimp",
            "name": "虾类",
            "aliases": ["虾", "虾仁"],
            "parent_slugs": ["seafood"],
            "name_patterns": ["虾", "虾仁", "基围虾", "河虾"],
            "priority": 80,
        },
        {
            "slug": "leafy_vegetable",
            "name": "绿叶菜",
            "aliases": ["青菜", "叶菜"],
            "parent_slugs": ["vegetable"],
            "name_patterns": ["白菜", "小白菜", "上海青", "油菜", "青菜", "菠菜", "生菜", "油麦", "空心菜", "鸡毛菜", "菜心", "茼蒿"],
            "priority": 90,
        },
        {
            "slug": "mushroom",
            "name": "菌菇",
            "aliases": ["蘑菇", "菇类"],
            "parent_slugs": ["vegetable"],
            "name_patterns": ["菇", "蘑菇", "木耳", "银耳"],
            "priority": 100,
        },
        {
            "slug": "soy_product",
            "name": "豆制品",
            "aliases": ["豆腐类"],
            "name_patterns": ["豆腐", "豆皮", "腐竹", "千张", "豆干", "豆泡"],
            "priority": 110,
        },
        {
            "slug": "root_vegetable",
            "name": "根茎类蔬菜",
            "aliases": ["根茎菜"],
            "parent_slugs": ["vegetable"],
            "name_patterns": ["土豆", "萝卜", "胡萝卜", "山药", "莲藕", "藕", "红薯", "芋头"],
            "priority": 120,
        },
    ],
    "assignments": [],
}


def reset_taxonomy_cache_for_tests() -> None:
    global _TAXONOMY_CACHE
    _TAXONOMY_CACHE = None


def load_taxonomy(
    dsn: str | None = None,
    *,
    seed_path: Path | None = None,
    use_cache: bool = True,
) -> IngredientTaxonomyData:
    explicit_source = dsn is not None or seed_path is not None
    if use_cache and not explicit_source and _TAXONOMY_CACHE is not None:
        return _TAXONOMY_CACHE

    taxonomy_dsn = dsn if dsn is not None else _configured_taxonomy_dsn()
    if taxonomy_dsn:
        try:
            data = _load_taxonomy_from_postgres(taxonomy_dsn)
            return _cache_if_needed(data, use_cache=use_cache, explicit_source=explicit_source)
        except Exception:
            pass

    yaml_path = seed_path or DEFAULT_SEED_PATH
    try:
        data = _load_taxonomy_from_yaml(yaml_path)
        return _cache_if_needed(data, use_cache=use_cache, explicit_source=explicit_source)
    except Exception:
        data = _taxonomy_from_seed(_BUILTIN_SEED, source="builtin")
        return _cache_if_needed(data, use_cache=use_cache, explicit_source=explicit_source)


def category_rules(
    dsn: str | None = None,
    *,
    taxonomy_data: IngredientTaxonomyData | None = None,
) -> tuple[IngredientCategoryRule, ...]:
    return (taxonomy_data or load_taxonomy(dsn)).rules


def category_node_id(name: str) -> str:
    by_name = {rule.name: rule for rule in category_rules()}
    return by_name[name].node_id


def category_node_data(
    dsn: str | None = None,
    *,
    taxonomy_data: IngredientTaxonomyData | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "node_id": rule.node_id,
            "label": CATEGORY_LABEL,
            "name": rule.name,
            "aliases": _dedupe([rule.name, *rule.aliases, *_literal_alias_patterns(rule.name_patterns)]),
            "properties": {"slug": rule.slug, "priority": rule.priority},
        }
        for rule in category_rules(dsn, taxonomy_data=taxonomy_data)
    ]


def category_hierarchy_edges(
    dsn: str | None = None,
    *,
    taxonomy_data: IngredientTaxonomyData | None = None,
) -> list[dict[str, object]]:
    rules = category_rules(dsn, taxonomy_data=taxonomy_data)
    category_by_slug = {rule.slug: rule for rule in rules}
    edges: list[dict[str, object]] = []
    for rule in rules:
        for parent_slug in rule.parent_slugs:
            parent = category_by_slug.get(parent_slug)
            if parent is None:
                continue
            edges.append(
                {
                    "edge_id": f"edge:{rule.node_id}:is_a:{parent.node_id}",
                    "source_id": rule.node_id,
                    "target_id": parent.node_id,
                    "relation": IS_A_CATEGORY,
                    "properties": {},
                }
            )
    return edges


def aliases_for_ingredient(name: str) -> list[str]:
    aliases = [name]
    if name == "猪肉里脊":
        aliases.append("猪里脊")
    if name == "猪里脊":
        aliases.append("猪肉里脊")
    if name == "猪排骨":
        aliases.append("排骨")
    return list(dict.fromkeys(alias for alias in aliases if alias))


def categories_for_ingredient(
    name: str,
    source_category: str | None = None,
    dsn: str | None = None,
    *,
    taxonomy_data: IngredientTaxonomyData | None = None,
) -> list[IngredientCategoryRule]:
    data = taxonomy_data or load_taxonomy(dsn)
    category_by_slug = {rule.slug: rule for rule in data.rules}
    matched_slugs: list[str] = []

    for assignment in sorted(data.assignments, key=lambda item: item.priority):
        if _assignment_matches(assignment, name, source_category):
            matched_slugs.append(assignment.category_slug)

    for rule in data.rules:
        if source_category and any(_matches_pattern(pattern, source_category) for pattern in rule.source_category_patterns):
            matched_slugs.append(rule.slug)

    for rule in data.rules:
        if any(_matches_pattern(pattern, name) for pattern in rule.name_patterns):
            matched_slugs.append(rule.slug)

    return [
        category_by_slug[slug]
        for slug in dict.fromkeys(matched_slugs)
        if slug in category_by_slug
    ]


def category_terms_for_query(question: str) -> list[str]:
    matched: list[str] = []
    for rule in category_rules():
        candidates = (rule.name, *rule.aliases)
        if any(candidate and candidate in question for candidate in candidates):
            matched.append(rule.name)
    return list(dict.fromkeys(matched))


def _configured_taxonomy_dsn() -> str | None:
    try:
        from app.core.config import settings
    except Exception:
        return None
    return settings.taxonomy_postgres_dsn or settings.text2sql_postgres_dsn or settings.postgres_dsn


def _cache_if_needed(
    data: IngredientTaxonomyData,
    *,
    use_cache: bool,
    explicit_source: bool,
) -> IngredientTaxonomyData:
    if use_cache and not explicit_source:
        global _TAXONOMY_CACHE
        _TAXONOMY_CACHE = data
    return data


def _load_taxonomy_from_yaml(path: Path) -> IngredientTaxonomyData:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 PyYAML，无法读取 taxonomy seed。") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _taxonomy_from_seed(payload, source=f"yaml:{path}")


def _load_taxonomy_from_postgres(dsn: str) -> IngredientTaxonomyData:
    psycopg, dict_row = _ensure_postgres_driver()
    with psycopg.connect(dsn, row_factory=dict_row, connect_timeout=2) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT slug, name, enabled, priority
                FROM ingredient_categories
                WHERE enabled = true
                ORDER BY priority, slug
                """
            )
            categories = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT category_slug, alias, enabled, priority
                FROM ingredient_category_aliases
                WHERE enabled = true
                ORDER BY priority, alias
                """
            )
            aliases = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT child_slug, parent_slug, enabled, priority
                FROM ingredient_category_hierarchy
                WHERE enabled = true
                ORDER BY priority, child_slug, parent_slug
                """
            )
            hierarchy = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT category_slug, name_pattern, source_category_pattern, enabled, priority
                FROM ingredient_category_patterns
                WHERE enabled = true
                ORDER BY priority, category_slug
                """
            )
            patterns = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                """
                SELECT ingredient_name, source_category, category_slug, enabled, priority
                FROM ingredient_category_assignments
                WHERE enabled = true
                ORDER BY priority, ingredient_name NULLS LAST, source_category NULLS LAST
                """
            )
            assignments = [dict(row) for row in cursor.fetchall()]

    if not categories:
        raise RuntimeError("taxonomy_empty")
    return _taxonomy_from_rows(categories, aliases, hierarchy, patterns, assignments, source="postgres")


def _taxonomy_from_rows(
    categories: list[dict[str, Any]],
    aliases: list[dict[str, Any]],
    hierarchy: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    source: str,
) -> IngredientTaxonomyData:
    aliases_by_slug: dict[str, list[str]] = {}
    parents_by_slug: dict[str, list[str]] = {}
    name_patterns_by_slug: dict[str, list[str]] = {}
    source_patterns_by_slug: dict[str, list[str]] = {}

    for row in aliases:
        aliases_by_slug.setdefault(str(row["category_slug"]), []).append(str(row["alias"]))
    for row in hierarchy:
        parents_by_slug.setdefault(str(row["child_slug"]), []).append(str(row["parent_slug"]))
    for row in patterns:
        category_slug = str(row["category_slug"])
        if row.get("name_pattern"):
            name_patterns_by_slug.setdefault(category_slug, []).append(str(row["name_pattern"]))
        if row.get("source_category_pattern"):
            source_patterns_by_slug.setdefault(category_slug, []).append(str(row["source_category_pattern"]))

    rules = tuple(
        IngredientCategoryRule(
            slug=str(row["slug"]),
            name=str(row["name"]),
            aliases=tuple(_dedupe(aliases_by_slug.get(str(row["slug"]), []))),
            parent_slugs=tuple(_dedupe(parents_by_slug.get(str(row["slug"]), []))),
            name_patterns=tuple(_dedupe(name_patterns_by_slug.get(str(row["slug"]), []))),
            source_category_patterns=tuple(_dedupe(source_patterns_by_slug.get(str(row["slug"]), []))),
            enabled=bool(row.get("enabled", True)),
            priority=int(row.get("priority") or 100),
        )
        for row in categories
        if bool(row.get("enabled", True))
    )
    assignment_rows = tuple(
        IngredientCategoryAssignment(
            ingredient_name=_optional_text(row.get("ingredient_name")),
            source_category=_optional_text(row.get("source_category")),
            category_slug=str(row["category_slug"]),
            enabled=bool(row.get("enabled", True)),
            priority=int(row.get("priority") or 0),
        )
        for row in assignments
        if bool(row.get("enabled", True))
    )
    return IngredientTaxonomyData(rules=rules, assignments=assignment_rows, source=source)


def _taxonomy_from_seed(payload: dict[str, Any], *, source: str) -> IngredientTaxonomyData:
    categories = payload.get("categories") or []
    assignments = payload.get("assignments") or []
    rules = tuple(
        IngredientCategoryRule(
            slug=str(row["slug"]),
            name=str(row["name"]),
            aliases=tuple(str(item) for item in row.get("aliases", []) if item),
            parent_slugs=tuple(str(item) for item in row.get("parent_slugs", []) if item),
            name_patterns=tuple(str(item) for item in row.get("name_patterns", []) if item),
            source_category_patterns=tuple(str(item) for item in row.get("source_category_patterns", []) if item),
            enabled=bool(row.get("enabled", True)),
            priority=int(row.get("priority", 100)),
        )
        for row in categories
        if bool(row.get("enabled", True))
    )
    if not rules:
        raise RuntimeError("taxonomy_seed_empty")
    assignment_rows = tuple(
        IngredientCategoryAssignment(
            ingredient_name=_optional_text(row.get("ingredient_name")),
            source_category=_optional_text(row.get("source_category")),
            category_slug=str(row["category_slug"]),
            enabled=bool(row.get("enabled", True)),
            priority=int(row.get("priority", 0)),
        )
        for row in assignments
        if bool(row.get("enabled", True))
    )
    return IngredientTaxonomyData(
        rules=tuple(sorted(rules, key=lambda item: (item.priority, item.slug))),
        assignments=tuple(sorted(assignment_rows, key=lambda item: item.priority)),
        source=source,
    )


def _assignment_matches(
    assignment: IngredientCategoryAssignment,
    name: str,
    source_category: str | None,
) -> bool:
    name_matches = assignment.ingredient_name is None or assignment.ingredient_name == name
    source_matches = assignment.source_category is None or assignment.source_category == source_category
    if assignment.ingredient_name is None and assignment.source_category is None:
        return False
    return name_matches and source_matches


def _matches_pattern(pattern: str, text: str) -> bool:
    if not pattern or not text:
        return False
    try:
        return re.search(pattern, text) is not None
    except re.error:
        return pattern in text


def _literal_alias_patterns(patterns: tuple[str, ...]) -> list[str]:
    # 仅把普通词条提升为实体链接别名，避免把正则表达式或过短词写进图谱 alias。
    regex_chars = set("^$.*+?{}[]\\|()")
    return [
        pattern
        for pattern in patterns
        if len(pattern) >= 2 and not any(char in pattern for char in regex_chars)
    ]


def _optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _ensure_postgres_driver():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError("当前环境缺少 psycopg，无法读取 PostgreSQL taxonomy。") from exc
    return psycopg, dict_row
