from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.api


def test_root_returns_service_metadata(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "MediScan AI API",
        "status": "running",
        "docs": "/docs",
    }


def test_health_ready_when_models_are_loaded(
    client: TestClient,
    models_loaded: None,
) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "mediscan-ai",
        "models_loaded": True,
    }


def test_health_not_ready_when_models_are_missing(
    client: TestClient,
    models_not_loaded: None,
) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "not_ready",
        "service": "mediscan-ai",
        "models_loaded": False,
    }
