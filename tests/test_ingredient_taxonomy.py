"""食材 taxonomy loader 测试。"""

from pathlib import Path

from app.graphrag import ingredient_taxonomy as taxonomy_module
from app.graphrag.ingredient_taxonomy import (
    DEFAULT_SEED_PATH,
    categories_for_ingredient,
    category_hierarchy_edges,
    category_node_data,
    load_taxonomy,
    reset_taxonomy_cache_for_tests,
)


def test_taxonomy_loads_yaml_seed() -> None:
    data = load_taxonomy(seed_path=DEFAULT_SEED_PATH, use_cache=False)

    assert data.source.startswith("yaml:")
    assert any(rule.slug == "pork" and "大肉" in rule.aliases for rule in data.rules)
    assert any(rule.slug == "leafy_vegetable" and "白菜" in rule.name_patterns for rule in data.rules)
    leafy_node = next(node for node in category_node_data() if node["node_id"] == "ingredient_category:leafy_vegetable")
    assert "白菜" in leafy_node["aliases"]
    assert any(edge["source_id"] == "ingredient_category:pork" for edge in category_hierarchy_edges())
    assert any(node["node_id"] == "ingredient_category:leafy_vegetable" for node in category_node_data())


def test_taxonomy_postgres_rows_map_to_rules() -> None:
    data = taxonomy_module._taxonomy_from_rows(
        categories=[
            {"slug": "vegetable", "name": "蔬菜", "enabled": True, "priority": 10},
            {"slug": "leafy", "name": "叶菜", "enabled": True, "priority": 20},
        ],
        aliases=[{"category_slug": "leafy", "alias": "青菜"}],
        hierarchy=[{"child_slug": "leafy", "parent_slug": "vegetable"}],
        patterns=[{"category_slug": "leafy", "name_pattern": "白菜", "source_category_pattern": None}],
        assignments=[{"ingredient_name": "小油菜", "source_category": None, "category_slug": "leafy", "priority": 0}],
        source="postgres",
    )

    leafy = next(rule for rule in data.rules if rule.slug == "leafy")
    assert leafy.aliases == ("青菜",)
    assert leafy.parent_slugs == ("vegetable",)
    assert leafy.name_patterns == ("白菜",)
    assert data.assignments[0].ingredient_name == "小油菜"


def test_taxonomy_postgres_failure_falls_back_to_yaml(monkeypatch) -> None:
    def fail_loader(dsn: str):
        raise RuntimeError("postgres unavailable")

    reset_taxonomy_cache_for_tests()
    monkeypatch.setattr(taxonomy_module, "_load_taxonomy_from_postgres", fail_loader)

    data = load_taxonomy("postgresql://taxonomy.local/db", seed_path=DEFAULT_SEED_PATH, use_cache=False)

    assert data.source.startswith("yaml:")
    assert any(rule.slug == "pork" for rule in data.rules)


def test_taxonomy_categories_keep_existing_generalization() -> None:
    reset_taxonomy_cache_for_tests()

    assert [rule.slug for rule in categories_for_ingredient("猪排骨")] == ["pork"]
    assert [rule.slug for rule in categories_for_ingredient("猪肉里脊")] == ["pork", "lean_meat"]
    assert [rule.slug for rule in categories_for_ingredient("白菜")] == ["leafy_vegetable"]


def test_taxonomy_manual_assignment_has_priority() -> None:
    test_dir = Path(__file__).resolve().parents[1] / "tmp" / "taxonomy-tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    seed = test_dir / "ingredient_categories_manual.yaml"
    seed.write_text(
        """
categories:
  - slug: source_rule
    name: 来源分类
    source_category_patterns: [调料]
    priority: 10
  - slug: manual_rule
    name: 手工分类
    name_patterns: [盐]
    priority: 20
assignments:
  - ingredient_name: 盐
    category_slug: manual_rule
    priority: 0
""".strip(),
        encoding="utf-8",
    )
    data = load_taxonomy(seed_path=seed, use_cache=False)
    taxonomy_module._TAXONOMY_CACHE = data

    try:
        assert [rule.slug for rule in categories_for_ingredient("盐", "调料")] == ["manual_rule", "source_rule"]
    finally:
        reset_taxonomy_cache_for_tests()
