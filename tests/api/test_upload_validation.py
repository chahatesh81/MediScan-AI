from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.api


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/predict",
        "/api/v1/analyze",
        "/api/v1/explain",
        "/api/v1/explain/overlay",
    ],
)
def test_rejects_unsupported_media_type(
    client: TestClient,
    endpoint: str,
) -> None:
    response = client.post(
        endpoint,
        files={
            "file": (
                "report.txt",
                b"not-an-image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "detail": (
            "Unsupported image type. "
            "Use JPEG or PNG."
        )
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/predict",
        "/api/v1/analyze",
        "/api/v1/explain",
        "/api/v1/explain/overlay",
    ],
)
def test_rejects_empty_upload(
    client: TestClient,
    endpoint: str,
) -> None:
    response = client.post(
        endpoint,
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Uploaded image is empty."
    }


@pytest.mark.parametrize(
    "endpoint",
    [
        "/api/v1/predict",
        "/api/v1/analyze",
        "/api/v1/explain",
        "/api/v1/explain/overlay",
    ],
)
def test_rejects_upload_larger_than_ten_megabytes(
    client: TestClient,
    endpoint: str,
) -> None:
    oversized = b"x" * (
        10 * 1024 * 1024 + 1
    )

    response = client.post(
        endpoint,
        files={
            "file": (
                "large.png",
                oversized,
                "image/png",
            )
        },
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": (
            "Image exceeds the 10 MB "
            "upload limit."
        )
    }
