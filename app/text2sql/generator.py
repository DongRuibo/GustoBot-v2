"""SQL 生成模块。

这个文件根据用户问题和 Schema Catalog 检索结果生成候选 SQL。
当前使用规则生成器保证可控和可测试，后续可以替换为低温 LLM 生成候选 SQL，但仍必须经过校验器。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.text2sql.schema import SchemaMatch


@dataclass(slots=True)
class GeneratedSQL:
    # GeneratedSQL 保存候选 SQL 和生成原因，便于 Evidence 中追踪“为什么生成这条 SQL”。
    sql: str
    reason: str


@dataclass(slots=True)
class SQLGenerationFeedback:
    # 校验失败反馈只提供上一轮非法 SQL 和校验原因，不包含数据库执行结果。
    invalid_sql: str
    validation_reason: str
    attempt: int


class SQLGenerator(Protocol):
    def generate(
        self,
        question: str,
        schema_matches: list[SchemaMatch],
        feedback: SQLGenerationFeedback | None = None,
    ) -> GeneratedSQL:
        ...


class RuleBasedSQLGenerator:
    # 规则生成器只覆盖第三阶段最小统计问题，目标是先把安全执行链路打通。
    # 真正接入 LLM 后，这里可以变成 prompt + model 输出，但安全校验和只读执行不能省略。
    def generate(
        self,
        question: str,
        schema_matches: list[SchemaMatch],
        feedback: SQLGenerationFeedback | None = None,
    ) -> GeneratedSQL:
        if not schema_matches:
            return GeneratedSQL(
                sql="SELECT name, cuisine FROM recipes LIMIT 10",
                reason="未检索到明确 schema，使用菜谱主表的安全预览查询。",
            )

        lowered = question.lower()
        if _has_table(schema_matches, "food_products") and _is_food_question(question):
            return _generate_food_product_sql(question, lowered)

        if "平均" in question and ("耗时" in question or "时间" in question):
            return GeneratedSQL(
                sql=(
                    "SELECT cuisine, ROUND(AVG(cooking_time_minutes), 1) AS avg_cooking_time_minutes "
                    "FROM recipes GROUP BY cuisine ORDER BY avg_cooking_time_minutes ASC"
                ),
                reason="问题包含平均耗时意图，按菜系统计平均烹饪时间。",
            )

        cuisine = _extract_cuisine(question)
        if cuisine and ("多少" in question or "数量" in question or "统计" in question):
            return GeneratedSQL(
                sql=f"SELECT COUNT(*) AS recipe_count FROM recipes WHERE cuisine = '{cuisine}'",
                reason=f"问题询问 {cuisine} 菜谱数量，生成带菜系过滤的计数 SQL。",
            )

        if "菜系" in question and ("数量" in question or "统计" in question or "排名" in question):
            return GeneratedSQL(
                sql=(
                    "SELECT cuisine, COUNT(*) AS recipe_count "
                    "FROM recipes GROUP BY cuisine ORDER BY recipe_count DESC"
                ),
                reason="问题包含按菜系统计数量意图，生成 GROUP BY 聚合 SQL。",
            )

        if "排名" in question or "top" in lowered or "最多" in question or "热度" in question:
            return GeneratedSQL(
                sql=(
                    "SELECT name, cuisine, popularity "
                    "FROM recipes ORDER BY popularity DESC LIMIT 5"
                ),
                reason="问题包含排名或热度意图，生成按 popularity 降序的只读查询。",
            )

        if "趋势" in question or "年份" in question:
            return GeneratedSQL(
                sql=(
                    "SELECT created_year, COUNT(*) AS recipe_count "
                    "FROM recipes GROUP BY created_year ORDER BY created_year ASC"
                ),
                reason="问题包含趋势或年份意图，生成按年份聚合的统计 SQL。",
            )

        return GeneratedSQL(
            sql="SELECT name, cuisine, difficulty, cooking_time_minutes FROM recipes LIMIT 10",
            reason="未命中特定统计模板，返回菜谱主表的安全只读预览查询。",
        )


@dataclass(slots=True)
class LLMSQLGenerator:
    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 20
    temperature: float = 0.0
    fallback: SQLGenerator | None = None
    fallback_on_error: bool = True
    max_retries: int = 1

    def generate(
        self,
        question: str,
        schema_matches: list[SchemaMatch],
        feedback: SQLGenerationFeedback | None = None,
    ) -> GeneratedSQL:
        fallback = self.fallback or RuleBasedSQLGenerator()
        if not schema_matches:
            if not self.fallback_on_error:
                raise RuntimeError("text2sql schema matches empty and rule fallback is disabled")
            return fallback.generate(question, schema_matches)
        if _has_table(schema_matches, "food_products") and _is_food_question(question):
            return fallback.generate(question, schema_matches)
        try:
            generated = self._generate_with_llm(question, schema_matches, feedback)
        except Exception as exc:
            if not self.fallback_on_error:
                raise RuntimeError(f"text2sql llm request failed: {str(exc)[:160]}") from exc
            fallback_sql = fallback.generate(question, schema_matches)
            return GeneratedSQL(
                sql=fallback_sql.sql,
                reason=f"llm_text2sql_failed_fallback: {str(exc)[:160]}; fallback_reason={fallback_sql.reason}",
            )
        if not generated.sql.strip():
            if not self.fallback_on_error:
                raise RuntimeError("text2sql llm returned empty sql and rule fallback is disabled")
            fallback_sql = fallback.generate(question, schema_matches)
            return GeneratedSQL(
                sql=fallback_sql.sql,
                reason=f"llm_text2sql_empty_fallback; fallback_reason={fallback_sql.reason}",
            )
        return generated

    def _generate_with_llm(
        self,
        question: str,
        schema_matches: list[SchemaMatch],
        feedback: SQLGenerationFeedback | None,
    ) -> GeneratedSQL:
        response = self._post(
            {
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是 GustoBot-v2 的 Text2SQL 候选 SQL 生成器。"
                            "只生成 PostgreSQL 只读 SELECT 查询；不得生成 INSERT、UPDATE、DELETE、DROP、ALTER、TRUNCATE。"
                            "只能使用给定 schema 中的表和字段。返回 JSON：{\"sql\":\"...\",\"reason\":\"...\"}，不要 Markdown。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(question, schema_matches, feedback),
                    },
                ],
            }
        )
        content = response.json()["choices"][0]["message"]["content"]
        payload = _parse_json_object(str(content))
        return GeneratedSQL(
            sql=str(payload.get("sql") or "").strip(),
            reason=f"llm_text2sql: {str(payload.get('reason') or '').strip()}",
        )

    def _build_prompt(
        self,
        question: str,
        schema_matches: list[SchemaMatch],
        feedback: SQLGenerationFeedback | None = None,
    ) -> str:
        schema_lines: list[str] = []
        for match in schema_matches:
            columns = ", ".join(
                f"{column.name} {column.data_type}".strip() for column in match.table.columns
            )
            schema_lines.append(
                f"- table={match.table.name}; description={match.table.comment}; columns=[{columns}]"
            )
        feedback_text = ""
        if feedback:
            feedback_text = (
                "\n上一轮 SQL 未通过安全校验，请修正后重新生成。\n"
                f"上一轮 SQL：{feedback.invalid_sql}\n"
                f"校验失败原因：{feedback.validation_reason}\n"
                f"当前修正轮次：{feedback.attempt}\n"
            )
        return (
            f"用户问题：{question}\n"
            f"可用 schema：\n{chr(10).join(schema_lines)}\n"
            f"{feedback_text}"
            "要求：返回一个单条 SELECT SQL；需要聚合时使用 COUNT/AVG/GROUP BY；需要排序时加 ORDER BY；"
            "除非问题明确要求更多结果，否则 LIMIT 不超过 50。"
        )

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        last_exc: Exception | None = None
        for attempt in range(max(0, self.max_retries) + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                time.sleep(0.2 * (attempt + 1))
        raise last_exc or RuntimeError("text2sql llm request failed")


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I).strip()
        stripped = re.sub(r"\s*```$", "", stripped).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.S)
        data = json.loads(match.group(0)) if match else {}
    return data if isinstance(data, dict) else {}


def _extract_cuisine(question: str) -> str | None:
    # 当前先支持常见菜系的确定性抽取，后续可以交给 Router slots 或 schema catalog 样例值检索增强。
    for cuisine in ("川菜", "粤菜", "闽菜", "鲁菜", "家常菜"):
        if cuisine in question:
            return cuisine
    return None


def _has_table(schema_matches: list[SchemaMatch], table_name: str) -> bool:
    return any(match.table.name == table_name for match in schema_matches)


def _is_food_question(question: str) -> bool:
    return any(
        keyword in question
        for keyword in ("食品", "商品", "产品", "品牌", "糖分", "含糖", "蛋白质", "营养", "过敏原", "分类", "配料")
    )


def _generate_food_product_sql(question: str, lowered: str) -> GeneratedSQL:
    limit = _extract_limit(question, default=10)
    if "糖" in question or "sugar" in lowered:
        return GeneratedSQL(
            sql=(
                "SELECT name, brand, sugars "
                "FROM food_products WHERE sugars IS NOT NULL "
                f"ORDER BY sugars DESC LIMIT {limit}"
            ),
            reason="问题询问食品糖分排名，按 food_products.sugars 降序查询。",
        )
    if "蛋白" in question or "protein" in lowered:
        return GeneratedSQL(
            sql=(
                "SELECT name, brand, protein "
                "FROM food_products WHERE protein IS NOT NULL "
                f"ORDER BY protein DESC LIMIT {limit}"
            ),
            reason="问题询问蛋白质含量排名，按 food_products.protein 降序查询。",
        )

    brand = _extract_brand(question)
    if brand and ("多少" in question or "数量" in question or "统计" in question):
        return GeneratedSQL(
            sql=f"SELECT COUNT(*) AS product_count FROM food_products WHERE brand ILIKE '%{_sql_literal_like(brand)}%'",
            reason=f"问题询问 {brand} 品牌产品数量，生成品牌过滤计数 SQL。",
        )

    category = _extract_food_category(question)
    if category and ("多少" in question or "数量" in question or "统计" in question):
        return GeneratedSQL(
            sql=f"SELECT COUNT(*) AS product_count FROM food_products WHERE category ILIKE '%{_sql_literal_like(category)}%'",
            reason=f"问题询问 {category} 分类产品数量，生成分类过滤计数 SQL。",
        )

    if "统计" in question or "数量" in question or "多少" in question:
        return GeneratedSQL(
            sql="SELECT category, COUNT(*) AS product_count FROM food_products GROUP BY category ORDER BY product_count DESC LIMIT 20",
            reason="问题包含食品统计意图，按分类统计产品数量。",
        )

    return GeneratedSQL(
        sql="SELECT name, brand, category, source FROM food_products LIMIT 10",
        reason="未命中特定食品统计模板，返回食品商品主表的安全预览查询。",
    )


def _extract_limit(question: str, *, default: int) -> int:
    match = re.search(r"(?:top|前)\s*(\d+)", question, flags=re.IGNORECASE)
    if not match:
        return default
    return max(1, min(int(match.group(1)), 50))


def _extract_brand(question: str) -> str | None:
    patterns = (
        r"(.+?)\s*品牌.*?(?:多少|数量|统计)",
        r"统计\s*(.+?)\s*品牌",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            brand = _clean_extracted_text(match.group(1))
            return brand or None
    return None


def _extract_food_category(question: str) -> str | None:
    patterns = (
        r"统计\s*(.+?)\s*(?:分类|类别)",
        r"(.+?)\s*(?:分类|类别).*?(?:多少|数量|统计)",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            category = _clean_extracted_text(match.group(1))
            return category or None
    return None


def _clean_extracted_text(value: str) -> str:
    text = re.sub(r"[，,。?？!！]", " ", value).strip()
    for prefix in ("请", "帮我", "一下", "这个"):
        text = text.replace(prefix, "")
    return re.sub(r"\s+", " ", text).strip()


def _sql_literal_like(value: str) -> str:
    return value.replace("'", "''").replace("%", "\\%").replace("_", "\\_")
