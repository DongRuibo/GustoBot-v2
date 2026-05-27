"""缓存存储模块。

这个文件定义缓存协议，并提供内存缓存和 Redis 缓存两个实现。
工作流通过这个抽象读写热点问答缓存，避免把 Redis 客户端调用散落到业务节点里。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from app.core.config import settings


class CacheStore(Protocol):
    # CacheStore 是缓存层的最小协议，后续语义缓存、节点结果缓存都可以复用这两个方法。
    store_type: str

    def get_json(self, key: str) -> dict[str, Any] | None:
        ...

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        ...


@dataclass(slots=True)
class _MemoryCacheItem:
    # 内存缓存项需要保存过期时间，避免长时间运行时一直返回过旧答案。
    value: dict[str, Any]
    expires_at: float


class InMemoryCacheStore:
    # 内存缓存适合本地开发和测试，不依赖 Redis 服务，但进程重启后缓存会丢失。
    store_type = "memory"

    def __init__(self) -> None:
        self._items: dict[str, _MemoryCacheItem] = {}
        self._lock = RLock()

    def get_json(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item.expires_at < time.time():
                self._items.pop(key, None)
                return None
            return dict(item.value)

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        with self._lock:
            self._items[key] = _MemoryCacheItem(
                value=dict(value),
                expires_at=time.time() + ttl_seconds,
            )


class RedisCacheStore:
    # Redis 缓存用于生产或多进程部署场景；这里采用惰性导入，避免未安装 redis 时影响本地测试。
    store_type = "redis"

    def __init__(self, redis_url: str) -> None:
        redis_module = self._ensure_driver()
        self.client = redis_module.Redis.from_url(redis_url, decode_responses=True)
        # 初始化时 ping 一次，提前暴露 Redis 配置错误；调用方会捕获后降级到内存缓存。
        self.client.ping()

    def get_json(self, key: str) -> dict[str, Any] | None:
        raw_value = self.client.get(key)
        if raw_value is None:
            return None
        return json.loads(raw_value)

    def set_json(self, key: str, value: dict[str, Any], *, ttl_seconds: int) -> None:
        self.client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))

    @staticmethod
    def _ensure_driver():
        try:
            import redis
        except ImportError as exc:
            raise RuntimeError(
                "已配置 GUSTOBOT_REDIS_URL，但当前环境缺少 redis 包。"
                "请安装 redis，或取消该环境变量改用内存缓存。"
            ) from exc
        return redis


_cache_store: CacheStore | None = None
_cache_lock = RLock()


def get_cache_store() -> CacheStore:
    # 缓存使用惰性单例，避免应用导入时就连接 Redis；Redis 不可用时自动回退到内存缓存。
    global _cache_store
    if _cache_store is None:
        with _cache_lock:
            if _cache_store is None:
                _cache_store = _build_cache_store()
    return _cache_store


def reset_cache_store_for_tests(store: CacheStore | None = None) -> None:
    # 测试隔离用函数，生产代码不要调用。
    global _cache_store
    with _cache_lock:
        _cache_store = store


def _build_cache_store() -> CacheStore:
    if settings.redis_url:
        try:
            return RedisCacheStore(settings.redis_url)
        except Exception:
            if settings.strict_external_stores:
                raise
            # Redis 是工程化增强，不应因为 Redis 暂时不可用导致开发/测试主流程启动失败。
            return InMemoryCacheStore()
    if settings.strict_external_stores:
        raise RuntimeError("生产环境必须配置 GUSTOBOT_REDIS_URL，不能退回内存缓存。")
    return InMemoryCacheStore()
