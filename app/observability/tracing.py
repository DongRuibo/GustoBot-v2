"""日志追踪模块。

这个文件负责把一次请求的关键生命周期事件写入 JSONL 日志，包括请求开始、缓存命中、
路由结果、最终响应等信息。日志统一携带 trace_id，方便排查问题和做离线评估。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from app.core.config import settings


_trace_lock = RLock()


def record_trace_event(trace_id: str, event_type: str, payload: dict[str, Any]) -> None:
    # 追踪日志是“旁路能力”，写日志失败不能影响主问答流程，因此这里吞掉文件系统异常。
    if not settings.trace_enabled:
        return

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event_type": event_type,
        "payload": payload,
    }
    try:
        path = Path(settings.trace_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _trace_lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        return

