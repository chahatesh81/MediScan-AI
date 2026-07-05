from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.inference_service import (
    inference_service,
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """
    HTTP client for deterministic API contract tests.

    The TestClient is intentionally not used as a context
    manager so the production lifespan does not eagerly load
    TensorFlow model artifacts during isolated API tests.
    """
    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as test_client:
        yield test_client


@pytest.fixture
def models_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Represent loaded models without loading TensorFlow artifacts.
    """

    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        object(),
    )
    monkeypatch.setattr(
        inference_service,
        "_v3_model",
        object(),
    )


@pytest.fixture
def models_not_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inference_service,
        "_v1_model",
        None,
    )
    monkeypatch.setattr(
        inference_service,
        "_v3_model",
        None,
    )
