from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.api


def test_model_info_preserves_frozen_deployment_policy(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["primary_model"]["name"] == (
        "baseline_cnn_v1"
    )
    assert body["primary_model"]["role"] == (
        "primary_classifier"
    )

    assert body["secondary_model"]["name"] == (
        "advanced_v3"
    )
    assert (
        body["secondary_model"][
            "automatic_override_allowed"
        ]
        is False
    )

    policy = body["deployment_policy"]

    assert policy["primary_prediction_source"] == (
        "baseline_cnn_v1"
    )
    assert (
        policy["automatic_override_allowed"]
        is False
    )
    assert (
        policy["automatic_ensemble_allowed"]
        is False
    )
    assert policy["warning_condition"] == (
        "v1_predicts_normal_and_v3_predicts_pneumonia"
    )

    assert "Not for clinical use" in body["disclaimer"]


def test_model_info_exposes_numeric_thresholds(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/model-info")

    body = response.json()

    assert isinstance(
        body["primary_model"]["threshold"],
        float,
    )
    assert isinstance(
        body["secondary_model"]["threshold"],
        float,
    )
