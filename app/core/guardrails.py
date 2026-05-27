"""Guardrails 安全检查模块。

这个文件负责在用户请求进入 Router 前做全局安全拦截，并在答案返回前做最小输出校验。
当前实现是确定性规则版本，后续可以扩展为规则和小模型结合的安全策略。
"""

from __future__ import annotations

from app.models import GuardrailResult


# 第一阶段先用确定性关键词规则兜底，后续可扩展为规则 + 小模型混合 Guardrails。
BLOCK_RULES: dict[str, tuple[str, ...]] = {
    "dangerous_database_write": (
        "drop table",
        "truncate table",
        "delete from",
        "update ",
        "insert into",
        "alter table",
        "删除所有数据",
        "全部删除",
        "删除表",
        "清空数据库",
    ),
    "unsafe_illegal": (
        "制作爆炸物",
        "制毒",
        "绕过风控",
        "盗取账号",
    ),
}


def run_global_guardrails(text: str) -> GuardrailResult:
    # Global Guardrails 必须在 Router 前执行，避免危险请求进入检索、SQL 或图查询链路。
    normalized = text.lower().strip()
    if not normalized:
        return GuardrailResult(
            allowed=False,
            reason="输入为空，无法进入业务路由。",
            categories=["empty_input"],
        )

    categories: list[str] = []
    # 收集全部命中的风险类别，便于后续日志、评估和策略调优。
    for category, keywords in BLOCK_RULES.items():
        if any(keyword in normalized for keyword in keywords):
            categories.append(category)

    if categories:
        return GuardrailResult(
            allowed=False,
            reason="请求包含危险操作或不适合执行的内容，已在全局 Guardrails 阶段拦截。",
            categories=categories,
        )

    return GuardrailResult(allowed=True, reason="通过全局 Guardrails。", categories=[])


def run_answer_guardrails(answer: str) -> GuardrailResult:
    # 输出 Guardrails 当前只做最小校验，后续可补充来源引用、幻觉检测等规则。
    if not answer.strip():
        return GuardrailResult(
            allowed=False,
            reason="答案为空，已被输出 Guardrails 拦截。",
            categories=["empty_answer"],
        )

    return GuardrailResult(allowed=True, reason="通过答案 Guardrails。", categories=[])
