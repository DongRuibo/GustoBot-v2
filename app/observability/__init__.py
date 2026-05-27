"""可观测性子包。

这个子包实现第四阶段的日志追踪能力，当前通过 JSONL 文件记录关键 trace 事件。
后续可以把同一套事件结构上报到 OpenTelemetry、ELK、LangSmith 或其他观测平台。
"""

