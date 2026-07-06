from __future__ import annotations

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import module_analysis
from backend.app.modules.dispatcher import (
    ModuleDispatchResult,
)
from backend.app.modules.registry import (
    PNEUMONIA_DETECTION,
)
from backend.app.modules.results import (
    BinaryClassificationResult,
)


pytestmark = pytest.mark.api


def make_dispatch_result() -> ModuleDispatchResult:
    return ModuleDispatchResult(
        module=PNEUMONIA_DETECTION,
        result=BinaryClassificationResult(
            negative_label="NORMAL",
            positive_label="PNEUMONIA",
            predicted_label="PNEUMONIA",
            probability=0.81,
            threshold=0.53,
        ),
    )


def test_module_route_uses_dispatcher_as_authorization_boundary(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_mock = Mock(
        return_value=make_dispatch_result()
    )

    monkeypatch.setattr(
        module_analysis,
        "dispatch_module_analysis",
        dispatch_mock,
    )

    response = client.post(
        "/api/v1/modules/pneumonia_detection/analyze",
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    dispatch_mock.assert_called_once_with(
        "pneumonia_detection",
        b"valid-image-bytes",
    )


def test_unknown_module_error_from_dispatcher_maps_to_404(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.modules.execution import (
        UnknownModuleError,
    )

    monkeypatch.setattr(
        module_analysis,
        "dispatch_module_analysis",
        Mock(
            side_effect=UnknownModuleError(
                "unknown_module"
            )
        ),
    )

    response = client.post(
        "/api/v1/modules/unknown_module/analyze",
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 404


def test_non_executable_error_from_dispatcher_maps_to_409(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.app.modules.execution import (
        ModuleNotExecutableError,
    )
    from backend.app.modules.registry import (
        BRAIN_TUMOR_MRI,
    )

    monkeypatch.setattr(
        module_analysis,
        "dispatch_module_analysis",
        Mock(
            side_effect=ModuleNotExecutableError(
                BRAIN_TUMOR_MRI
            )
        ),
    )

    response = client.post(
        "/api/v1/modules/brain_tumor_mri/analyze",
        files={
            "file": (
                "scan.png",
                b"valid-image-bytes",
                "image/png",
            )
        },
    )

    assert response.status_code == 409
