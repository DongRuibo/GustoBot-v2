"""FastAPI 应用创建模块。

这个文件负责创建 ASGI 应用实例并挂载 API 路由，是 uvicorn 启动服务时加载的入口。
它不直接包含业务逻辑，业务处理会下沉到 LangGraph 工作流和 KB 服务中。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core import p0_readiness
from app.core.config import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    if p0_readiness.should_run_startup_check(settings):
        result = p0_readiness.assert_p0_startup_ready(settings)
        logger.info(
            "P0 strict readiness passed: %s",
            json.dumps(result.get("snapshot", {}), ensure_ascii=False, sort_keys=True),
        )
    yield


def create_app() -> FastAPI:
    # FastAPI 应用只负责挂载接口，业务编排统一放在 LangGraph 工作流中。
    app = FastAPI(title="GustoBot-v2", version="0.1.0", lifespan=lifespan)
    # 阶段一前端为本地 Vite 应用，先开放开发端口跨域访问。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


# uvicorn 默认从这里加载 ASGI 应用。
app = create_app()
