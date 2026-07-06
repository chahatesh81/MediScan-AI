from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app import main


pytestmark = pytest.mark.unit


def test_application_startup_validates_module_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validate_mock = Mock()

    monkeypatch.setattr(
        main,
        "validate_module_runtime",
        validate_mock,
    )

    with TestClient(
        main.app,
        raise_server_exceptions=False,
    ):
        pass

    validate_mock.assert_called_once_with()


def test_invalid_module_runtime_prevents_application_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_validation() -> None:
        raise RuntimeError(
            "invalid module runtime"
        )

    monkeypatch.setattr(
        main,
        "validate_module_runtime",
        fail_validation,
    )

    with pytest.raises(
        RuntimeError,
        match="invalid module runtime",
    ):
        with TestClient(main.app):
            pass
