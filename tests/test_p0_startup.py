"""P0 strict 启动门禁测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.core.p0_readiness import P0ReadinessError


def test_startup_readiness_skips_dev_mode(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(main_module, "settings", SimpleNamespace(environment="dev", strict_external_stores=False))
    monkeypatch.setattr(
        main_module.p0_readiness,
        "assert_p0_startup_ready",
        lambda settings: calls.append(settings),
    )

    with TestClient(main_module.create_app()) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert calls == []


def test_startup_readiness_runs_in_prod_strict(monkeypatch) -> None:
    calls: list[object] = []
    strict_settings = SimpleNamespace(environment="prod", strict_external_stores=True)
    monkeypatch.setattr(main_module, "settings", strict_settings)

    def fake_assert(settings):
        calls.append(settings)
        return {"status": "ok", "snapshot": {"kb": {"store_type": "postgres_pgvector"}}}

    monkeypatch.setattr(main_module.p0_readiness, "assert_p0_startup_ready", fake_assert)

    with TestClient(main_module.create_app()) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert calls == [strict_settings]


def test_startup_readiness_fails_fast_in_prod_strict(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "settings", SimpleNamespace(environment="prod", strict_external_stores=True))

    def fake_assert(settings):
        raise P0ReadinessError("P0 strict readiness failed at runtime")

    monkeypatch.setattr(main_module.p0_readiness, "assert_p0_startup_ready", fake_assert)

    with pytest.raises(P0ReadinessError, match="runtime"):
        with TestClient(main_module.create_app()):
            pass
