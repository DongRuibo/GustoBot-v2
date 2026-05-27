"""食品数据底座公共处理函数。

这里集中放清洗、字段映射、图谱构建和 JSONL/CSV 读写逻辑，避免每个脚本各写一套字段规则。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
SFT_DIR = DATA_DIR / "sft"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRODUCT_FIELDS = [
    "product_id",
    "name",
    "brand",
    "category",
    "country",
    "ingredients_text",
    "allergens",
    "energy_kcal",
    "protein",
    "fat",
    "carbohydrates",
    "sugars",
    "salt",
    "source",
]

NUTRIENT_FIELDS = ["product_id", "nutrient_name", "value", "unit", "source"]
GRAPH_NODE_FIELDS = ["node_id", "label", "name", "aliases", "properties"]
GRAPH_EDGE_FIELDS = ["edge_id", "source_id", "target_id", "relation", "properties"]

NUTRIENT_COLUMNS = {
    "energy_kcal": ("energy_kcal", "kcal"),
    "protein": ("protein", "g"),
    "fat": ("fat", "g"),
    "carbohydrates": ("carbohydrates", "g"),
    "sugars": ("sugars", "g"),
    "salt": ("salt", "g"),
}


@dataclass(slots=True)
class FoodProduct:
    product_id: str
    name: str
    brand: str = ""
    category: str = ""
    country: str = ""
    ingredients_text: str = ""
    allergens: str = ""
    energy_kcal: float | None = None
    protein: float | None = None
    fat: float | None = None
    carbohydrates: float | None = None
    sugars: float | None = None
    salt: float | None = None
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    nutrient_values: dict[str, tuple[float, str]] = field(default_factory=dict)

    def to_csv_row(self) -> dict[str, str]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category,
            "country": self.country,
            "ingredients_text": self.ingredients_text,
            "allergens": self.allergens,
            "energy_kcal": _format_number(self.energy_kcal),
            "protein": _format_number(self.protein),
            "fat": _format_number(self.fat),
            "carbohydrates": _format_number(self.carbohydrates),
            "sugars": _format_number(self.sugars),
            "salt": _format_number(self.salt),
            "source": self.source,
        }


@dataclass(slots=True)
class FoodOnTerm:
    term_id: str
    label: str
    parent_id: str = ""
    aliases: list[str] = field(default_factory=list)


SAMPLE_PRODUCTS = [
    FoodProduct(
        product_id="sample:peanut_bar",
        name="Peanut Protein Bar",
        brand="Demo Foods",
        category="Protein bars",
        country="United States",
        ingredients_text="peanuts, soy protein isolate, cocoa, sugar, salt",
        allergens="peanuts, soy",
        energy_kcal=420,
        protein=28,
        fat=16,
        carbohydrates=42,
        sugars=18,
        salt=0.6,
        source="sample",
    ),
    FoodProduct(
        product_id="sample:oat_milk",
        name="燕麦奶",
        brand="示例品牌",
        category="Plant-based drinks",
        country="China",
        ingredients_text="水, 燕麦, 菜籽油, 海盐",
        allergens="gluten",
        energy_kcal=58,
        protein=1.2,
        fat=1.5,
        carbohydrates=9.0,
        sugars=3.2,
        salt=0.12,
        source="sample",
    ),
    FoodProduct(
        product_id="sample:yogurt",
        name="Greek Yogurt",
        brand="Demo Dairy",
        category="Yogurts",
        country="United States",
        ingredients_text="milk, cream, live cultures",
        allergens="milk",
        energy_kcal=97,
        protein=9.0,
        fat=5.0,
        carbohydrates=3.6,
        sugars=3.6,
        salt=0.1,
        source="sample",
    ),
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def product_from_openfoodfacts(raw: dict[str, Any]) -> FoodProduct | None:
    """把 Open Food Facts JSONL 中的一行规整为统一食品商品字段。"""

    code = clean_text(raw.get("code") or raw.get("_id") or "")
    name = first_text(
        raw.get("product_name_zh"),
        raw.get("product_name_en"),
        raw.get("product_name"),
        _localized_value(raw.get("product_name")),
    )
    category = first_text(
        raw.get("categories"),
        raw.get("main_category"),
        _join_tags(raw.get("categories_tags")),
        _join_tags(raw.get("food_groups_tags")),
    )
    ingredients_text = first_text(
        raw.get("ingredients_text_zh"),
        raw.get("ingredients_text_en"),
        raw.get("ingredients_text"),
        _localized_value(raw.get("ingredients_text")),
    )
    nutriments = raw.get("nutriments") if isinstance(raw.get("nutriments"), dict) else {}
    allergens = first_text(raw.get("allergens"), _join_tags(raw.get("allergens_tags")))
    product = FoodProduct(
        product_id=f"off:{code}" if code else stable_id("off", name, raw.get("brands"), category),
        name=name,
        brand=clean_text(raw.get("brands")),
        category=category,
        country=first_text(raw.get("countries"), _join_tags(raw.get("countries_tags"))),
        ingredients_text=ingredients_text,
        allergens=allergens,
        energy_kcal=parse_float(_nutriment(nutriments, "energy-kcal")),
        protein=parse_float(_nutriment(nutriments, "proteins")),
        fat=parse_float(_nutriment(nutriments, "fat")),
        carbohydrates=parse_float(_nutriment(nutriments, "carbohydrates")),
        sugars=parse_float(_nutriment(nutriments, "sugars")),
        salt=parse_float(_nutriment(nutriments, "salt")),
        source="openfoodfacts",
        metadata={"lang": raw.get("lang")},
    )
    return product if product_passes_quality(product) else None


def product_from_usda(
    food: dict[str, str],
    nutrients: dict[str, float],
    *,
    brand: str = "",
    ingredients: str = "",
    category: str = "",
) -> FoodProduct | None:
    """把 USDA FoodData Central CSV 记录规整为统一食品商品字段。"""

    fdc_id = clean_text(food.get("fdc_id"))
    name = first_text(food.get("description"), food.get("lowercase_description"))
    data_type = clean_text(food.get("data_type"))
    sodium_mg = nutrients.get("sodium_mg")
    salt = round(sodium_mg * 2.5 / 1000, 4) if sodium_mg is not None else None
    product = FoodProduct(
        product_id=f"usda:{fdc_id}" if fdc_id else stable_id("usda", name, brand, data_type),
        name=name,
        brand=brand,
        category=first_text(category, data_type, "USDA Food"),
        country="United States",
        ingredients_text=ingredients,
        allergens="",
        energy_kcal=nutrients.get("energy_kcal"),
        protein=nutrients.get("protein"),
        fat=nutrients.get("fat"),
        carbohydrates=nutrients.get("carbohydrates"),
        sugars=nutrients.get("sugars"),
        salt=salt,
        source="usda",
        metadata={"data_type": data_type},
    )
    return product if product_passes_quality(product) else None


def product_passes_quality(product: FoodProduct) -> bool:
    if not product.name or not product.category:
        return False
    has_ingredients = bool(product.ingredients_text.strip())
    has_nutrients = any(
        value is not None
        for value in (
            product.energy_kcal,
            product.protein,
            product.fat,
            product.carbohydrates,
            product.sugars,
            product.salt,
        )
    )
    if not has_ingredients and not has_nutrients:
        return False
    if len(product.name) > 240 or len(product.category) > 240:
        return False
    return True


def dedupe_products(products: Iterable[FoodProduct]) -> list[FoodProduct]:
    deduped: dict[str, FoodProduct] = {}
    seen_signatures: set[str] = set()
    for product in products:
        if product.product_id in deduped:
            continue
        signature = "|".join(
            [
                product.name.strip().lower(),
                product.brand.strip().lower(),
                product.category.strip().lower(),
            ]
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped[product.product_id] = product
    return list(deduped.values())


def write_products_csv(path: Path, products: Iterable[FoodProduct]) -> int:
    ensure_parent(path)
    rows = [product.to_csv_row() for product in products]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PRODUCT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def read_products_csv(path: Path) -> list[FoodProduct]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [_product_from_row(row) for row in reader]


def write_nutrients_csv(path: Path, products: Iterable[FoodProduct]) -> int:
    ensure_parent(path)
    rows = list(nutrient_rows(products))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=NUTRIENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def nutrient_rows(products: Iterable[FoodProduct]) -> Iterable[dict[str, str]]:
    for product in products:
        row = product.to_csv_row()
        seen_names: set[str] = set()
        for nutrient_name, value_unit in product.nutrient_values.items():
            value, unit = value_unit
            value_text = _format_number(parse_float(value))
            clean_name = clean_text(nutrient_name)
            clean_unit = clean_text(unit)
            if not clean_name or not clean_unit or not value_text:
                continue
            if clean_name in seen_names:
                continue
            seen_names.add(clean_name)
            yield {
                "product_id": product.product_id,
                "nutrient_name": clean_name,
                "value": value_text,
                "unit": clean_unit,
                "source": product.source,
            }
        for column, (name, unit) in NUTRIENT_COLUMNS.items():
            if name in seen_names:
                continue
            value = row[column]
            if value:
                yield {
                    "product_id": product.product_id,
                    "nutrient_name": name,
                    "value": value,
                    "unit": unit,
                    "source": product.source,
                }


def build_graph(
    products: Iterable[FoodProduct],
    *,
    foodon_terms: Iterable[FoodOnTerm] | None = None,
    max_ingredients_per_product: int = 20,
    nutrient_columns: Iterable[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """从食品商品构建 Product/Ingredient/Allergen/Category/Nutrient 图谱 CSV。"""

    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    graph_nutrient_columns = tuple(nutrient_columns or NUTRIENT_COLUMNS.keys())
    root_category_id = "ingredient_category:food_ingredient"
    nodes[root_category_id] = _node(root_category_id, "IngredientCategory", "食品配料", ["ingredient"])

    for product in products:
        product_id = f"product:{_safe_node_key(product.product_id)}"
        nodes[product_id] = _node(
            product_id,
            "Product",
            product.name,
            [product.name, product.brand] if product.brand else [product.name],
            {
                "product_id": product.product_id,
                "brand": product.brand,
                "country": product.country,
                "source": product.source,
            },
        )

        category_name = clean_category(product.category)
        if category_name:
            category_id = f"food_category:{stable_slug(category_name)}"
            nodes.setdefault(category_id, _node(category_id, "FoodCategory", category_name, [category_name]))
            edges[f"edge:{product_id}:category"] = _edge(product_id, category_id, "BELONGS_TO")

        for allergen in split_terms(product.allergens, max_terms=12):
            allergen_id = f"allergen:{stable_slug(allergen)}"
            nodes.setdefault(allergen_id, _node(allergen_id, "Allergen", allergen, _allergen_aliases(allergen)))
            edges[f"edge:{product_id}:allergen:{stable_slug(allergen)}"] = _edge(product_id, allergen_id, "HAS_ALLERGEN")

        for ingredient in split_terms(product.ingredients_text, max_terms=max_ingredients_per_product):
            ingredient_id = f"ingredient:{stable_slug(ingredient)}"
            nodes.setdefault(ingredient_id, _node(ingredient_id, "Ingredient", ingredient, [ingredient]))
            edges[f"edge:{product_id}:ingredient:{stable_slug(ingredient)}"] = _edge(
                product_id,
                ingredient_id,
                "HAS_INGREDIENT",
            )
            category = ingredient_category(ingredient)
            category_id = f"ingredient_category:{category['slug']}"
            nodes.setdefault(
                category_id,
                _node(category_id, "IngredientCategory", category["name"], category["aliases"]),
            )
            edges[f"edge:{ingredient_id}:is_a:{category['slug']}"] = _edge(
                ingredient_id,
                category_id,
                "IS_A",
                {"source": "food_dataset_rules"},
            )
            if category_id != root_category_id:
                edges[f"edge:{category_id}:root"] = _edge(category_id, root_category_id, "IS_A")

        for column in graph_nutrient_columns:
            nutrient_name, unit = NUTRIENT_COLUMNS[column]
            value = getattr(product, column)
            if value is None:
                continue
            nutrient_id = f"nutrient:{nutrient_name}"
            nodes.setdefault(nutrient_id, _node(nutrient_id, "Nutrient", nutrient_name, [nutrient_name]))
            edges[f"edge:{product_id}:nutrient:{nutrient_name}"] = _edge(
                product_id,
                nutrient_id,
                "HAS_NUTRIENT",
                {"value": value, "unit": unit},
            )

    for term in foodon_terms or []:
        term_node_id = _foodon_node_id(term.term_id)
        aliases = [term.label, *term.aliases]
        nodes.setdefault(
            term_node_id,
            _node(
                term_node_id,
                "FoodCategory",
                term.label,
                aliases,
                {"source": "foodon", "term_id": term.term_id},
            ),
        )
        if term.parent_id:
            parent_node_id = _foodon_node_id(term.parent_id)
            nodes.setdefault(
                parent_node_id,
                _node(
                    parent_node_id,
                    "FoodCategory",
                    _foodon_label_from_id(term.parent_id),
                    [_foodon_label_from_id(term.parent_id)],
                    {"source": "foodon", "term_id": term.parent_id},
                ),
            )
            edges[f"edge:{term_node_id}:foodon_parent:{parent_node_id}"] = _edge(
                term_node_id,
                parent_node_id,
                "IS_A",
                {"source": "foodon"},
            )

    return _graph_rows(nodes.values(), GRAPH_NODE_FIELDS), _graph_rows(edges.values(), GRAPH_EDGE_FIELDS)


def write_graph_csv(
    nodes_path: Path,
    edges_path: Path,
    nodes: list[dict[str, str]],
    edges: list[dict[str, str]],
) -> None:
    ensure_parent(nodes_path)
    ensure_parent(edges_path)
    with nodes_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=GRAPH_NODE_FIELDS)
        writer.writeheader()
        writer.writerows(nodes)
    with edges_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=GRAPH_EDGE_FIELDS)
        writer.writeheader()
        writer.writerows(edges)


def build_kb_documents(
    products: list[FoodProduct],
    *,
    max_product_docs: int = 2000,
    foodon_terms: Iterable[FoodOnTerm] | None = None,
    max_foodon_docs: int = 400,
) -> list[dict[str, Any]]:
    """生成食品说明、营养素说明、过敏原说明和少量商品说明 KB 文档。"""

    documents = [
        _kb_doc(
            "food_category_overview",
            "食品分类说明",
            "食品分类用于把商品或标准食物归入饮料、乳制品、谷物、零食、调味品、肉类、蔬果等可检索集合。"
            "在 GustoBot-v2 中，分类字段进入 PostgreSQL 统计表，也作为 Neo4j 的 FoodCategory 节点。",
            {"doc_type": "food_category"},
        ),
        _kb_doc(
            "nutrient_protein",
            "蛋白质作用说明",
            "蛋白质是人体组织修复、肌肉合成和酶/激素合成的重要营养素。食品标签中的 protein 通常表示每 100g 蛋白质含量，单位为克。",
            {"doc_type": "nutrient"},
        ),
        _kb_doc(
            "nutrient_sugars",
            "糖分说明",
            "糖分 sugars 表示食品中单糖和双糖的含量。对控糖或能量管理场景，常需要结合总碳水、食用份量和配料表一起判断。",
            {"doc_type": "nutrient"},
        ),
        _kb_doc(
            "allergen_overview",
            "过敏原解释",
            "过敏原字段记录食品标签中声明的花生、坚果、牛奶、大豆、麸质、鸡蛋、鱼、甲壳类等风险来源。回答过敏相关问题时必须引用商品标签或图谱证据。",
            {"doc_type": "allergen"},
        ),
        _kb_doc(
            "usda_field_overview",
            "USDA FoodData 字段说明",
            "USDA FoodData Central 的 fdc_id 是食品记录唯一编号，data_type 区分 Foundation、SR Legacy、Branded 等数据集。"
            "description 是食品名称，food_category_id 或 branded_food_category 表示分类，food_nutrient 保存每种营养素的数值和单位。",
            {"doc_type": "usda_field"},
        ),
        _kb_doc(
            "foodon_overview",
            "FoodOn 类别和同义词说明",
            "FoodOn 是食品本体，提供食品类别、食材、食品形态及其同义词和上位词关系。"
            "在 GustoBot-v2 中，FoodOn 类别进入 Neo4j 的 FoodCategory 节点，也用于 KB 解释型问答。",
            {"doc_type": "foodon"},
        ),
    ]
    for product in products[:max_product_docs]:
        content = (
            f"商品名称：{product.name}\n"
            f"品牌：{product.brand or '未知'}\n"
            f"分类：{product.category}\n"
            f"国家/地区：{product.country or '未知'}\n"
            f"配料：{product.ingredients_text or '未提供'}\n"
            f"过敏原：{product.allergens or '未标注'}\n"
            f"营养标签：能量 {value_or_unknown(product.energy_kcal)} kcal，蛋白质 {value_or_unknown(product.protein)} g，"
            f"脂肪 {value_or_unknown(product.fat)} g，碳水 {value_or_unknown(product.carbohydrates)} g，"
            f"糖 {value_or_unknown(product.sugars)} g，盐 {value_or_unknown(product.salt)} g。"
        )
        documents.append(
            _kb_doc(
                f"food_product_{stable_slug(product.product_id)}",
                f"食品商品说明：{product.name}",
                content,
                {
                    "doc_type": "food_product",
                    "product_id": product.product_id,
                    "product_source": product.source,
                },
            )
        )
    for term in list(foodon_terms or [])[:max_foodon_docs]:
        aliases = "、".join(term.aliases[:6]) if term.aliases else "未提供"
        parent = term.parent_id or "未提供"
        documents.append(
            _kb_doc(
                f"foodon_{stable_slug(term.term_id)}",
                f"FoodOn 类别说明：{term.label}",
                f"FoodOn 类别：{term.label}\n同义词：{aliases}\n上位词 ID：{parent}\n"
                "该类别可用于解释食品分类、食材别名和食品图谱中的 IS_A 关系。",
                {"doc_type": "foodon_term", "term_id": term.term_id},
            )
        )
    return documents


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_project_path(path_value: str | Path, *, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"{label} 必须位于 GustoBot-v2 项目目录内：{resolved}") from exc
    return resolved


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = ", ".join(clean_text(item) for item in value if clean_text(item))
    if isinstance(value, dict):
        value = _localized_value(value)
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def first_text(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(str(value).replace(",", "."))
    except ValueError:
        return None
    if number < 0 or number > 100000:
        return None
    return round(number, 4)


def split_terms(text: str, *, max_terms: int) -> list[str]:
    cleaned = clean_text(text)
    cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    parts = re.split(r"[,，;；、/|]+", cleaned)
    terms: list[str] = []
    for part in parts:
        term = clean_tag(part)
        if not term or len(term) > 80:
            continue
        if term.lower() in {"and", "or", "contains", "may contain", "en"}:
            continue
        if term not in terms:
            terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def clean_tag(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^[a-z]{2,3}:", "", text)
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_category(value: str) -> str:
    terms = split_terms(value, max_terms=4)
    return terms[-1] if terms else clean_tag(value)


def stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(clean_text(part).lower() for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def stable_slug(value: str) -> str:
    cleaned = clean_tag(value).lower()
    if not cleaned:
        return "unknown"
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if ascii_slug:
        return ascii_slug[:80]
    return hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:16]


def ingredient_category(ingredient: str) -> dict[str, Any]:
    lowered = ingredient.lower()
    rules = [
        (("peanut", "花生"), "peanut", "花生及坚果", ["peanut", "nuts", "花生", "坚果"]),
        (("milk", "cream", "cheese", "yogurt", "牛奶", "奶", "乳"), "dairy", "乳制品", ["milk", "dairy", "乳制品"]),
        (("soy", "soya", "大豆", "豆"), "soy", "大豆制品", ["soy", "大豆"]),
        (("wheat", "oat", "barley", "gluten", "小麦", "燕麦", "麸质"), "grain", "谷物", ["grain", "gluten", "谷物"]),
        (("sugar", "syrup", "糖", "蜂蜜"), "sweetener", "甜味剂", ["sweetener", "糖"]),
        (("oil", "fat", "butter", "油", "脂"), "oil_fat", "油脂", ["oil", "fat", "油脂"]),
        (("chicken", "beef", "pork", "fish", "meat", "鸡", "牛", "猪", "鱼", "肉"), "protein_food", "肉蛋水产", ["meat", "protein", "肉蛋水产"]),
        (("apple", "berry", "banana", "fruit", "苹果", "水果"), "fruit", "水果", ["fruit", "水果"]),
        (("vegetable", "tomato", "白菜", "蔬菜"), "vegetable", "蔬菜", ["vegetable", "蔬菜"]),
        (("salt", "sodium", "盐"), "seasoning", "调味品", ["seasoning", "salt", "调味品"]),
    ]
    for keywords, slug, name, aliases in rules:
        if any(keyword in lowered or keyword in ingredient for keyword in keywords):
            return {"slug": slug, "name": name, "aliases": aliases}
    return {"slug": "other", "name": "其他配料", "aliases": ["other", "其他配料"]}


def _allergen_aliases(allergen: str) -> list[str]:
    aliases = [allergen]
    lowered = allergen.lower()
    mapping = {
        "peanut": ["花生", "peanuts"],
        "soy": ["大豆", "soybeans", "soya"],
        "milk": ["牛奶", "乳", "dairy"],
        "gluten": ["麸质", "小麦"],
        "wheat": ["小麦", "麸质"],
        "egg": ["鸡蛋", "蛋"],
        "fish": ["鱼"],
        "shellfish": ["甲壳类", "虾", "蟹"],
        "tree nut": ["坚果"],
    }
    for keyword, values in mapping.items():
        if keyword in lowered:
            aliases.extend(values)
    return list(dict.fromkeys(alias for alias in aliases if alias))


def value_or_unknown(value: float | None) -> str:
    return "未知" if value is None else _format_number(value)


def _product_from_row(row: dict[str, str]) -> FoodProduct:
    return FoodProduct(
        product_id=row["product_id"],
        name=row["name"],
        brand=row.get("brand", ""),
        category=row.get("category", ""),
        country=row.get("country", ""),
        ingredients_text=row.get("ingredients_text", ""),
        allergens=row.get("allergens", ""),
        energy_kcal=parse_float(row.get("energy_kcal")),
        protein=parse_float(row.get("protein")),
        fat=parse_float(row.get("fat")),
        carbohydrates=parse_float(row.get("carbohydrates")),
        sugars=parse_float(row.get("sugars")),
        salt=parse_float(row.get("salt")),
        source=row.get("source", ""),
    )


def _foodon_node_id(term_id: str) -> str:
    compact = re.sub(r"^.*[/#]", "", clean_text(term_id))
    return f"foodon_category:{stable_slug(compact or term_id)}"


def _foodon_label_from_id(term_id: str) -> str:
    compact = re.sub(r"^.*[/#]", "", clean_text(term_id))
    return compact.replace("_", " ") if compact else "FoodOn category"


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return ("%f" % value).rstrip("0").rstrip(".")


def _nutriment(nutriments: dict[str, Any], name: str) -> Any:
    for key in (f"{name}_100g", name, f"{name}_value"):
        if key in nutriments:
            return nutriments[key]
    return None


def _join_tags(value: Any) -> str:
    if not isinstance(value, list):
        return clean_text(value)
    return ", ".join(clean_tag(str(item)) for item in value if clean_tag(str(item)))


def _localized_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("zh", "cn", "en", "main"):
            if value.get(key):
                return clean_text(value[key])
        for item in value.values():
            text = clean_text(item)
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                text = first_text(item.get("text"), item.get("value"))
                if text:
                    return text
    return clean_text(value)


def _safe_node_key(value: str) -> str:
    return stable_slug(value.replace(":", "_"))


def _node(
    node_id: str,
    label: str,
    name: str,
    aliases: list[str] | None = None,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "label": label,
        "name": name,
        "aliases": list(dict.fromkeys(alias for alias in (aliases or [name]) if alias)),
        "properties": properties or {},
    }


def _edge(
    source_id: str,
    target_id: str,
    relation: str,
    properties: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_id": f"edge:{hashlib.sha1(f'{source_id}|{relation}|{target_id}'.encode('utf-8')).hexdigest()[:20]}",
        "source_id": source_id,
        "target_id": target_id,
        "relation": relation,
        "properties": properties or {},
    }


def _graph_rows(items: Iterable[dict[str, Any]], fieldnames: list[str]) -> list[dict[str, str]]:
    rows = []
    for item in items:
        row = {}
        for field in fieldnames:
            value = item.get(field)
            row[field] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value or "")
        rows.append(row)
    return sorted(rows, key=lambda row: row[fieldnames[0]])


def _kb_doc(source_id: str, title: str, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": f"food:{source_id}",
        "title": title,
        "content": content,
        "metadata": {"source": "food_dataset", **metadata},
    }
