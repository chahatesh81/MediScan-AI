from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.services import (
    inference_service as inference_module,
)
from backend.app.services.inference_service import (
    InferenceService,
)


pytestmark = pytest.mark.unit


def valid_selection(
    v1_model_path: Path,
) -> dict[str, Any]:
    return {
        "selected_primary_model": {
            "name": "baseline_cnn_v1",
            "model_file": str(v1_model_path),
            "decision_threshold": 0.25,
        }
    }


def valid_addendum(
    v3_model_path: Path,
) -> dict[str, Any]:
    return {
        "deployment_policy": {
            "primary_prediction_source": (
                "baseline_cnn_v1"
            ),
            "automatic_override_allowed": False,
            "automatic_ensemble_allowed": False,
            "recommended_warning_condition": (
                "v1_predicts_normal_and_v3_predicts_pneumonia"
            ),
        },
        "advanced_v3": {
            "checkpoint": str(v3_model_path),
            "threshold": 0.75,
        },
    }


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.write_text(json.dumps(payload))


def configure_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, Any],
    dict[str, Any],
]:
    selection_file = tmp_path / "selection.json"
    addendum_file = tmp_path / "addendum.json"
    v1_model = tmp_path / "v1.keras"
    v3_model = tmp_path / "v3.keras"

    v1_model.touch()
    v3_model.touch()

    selection = valid_selection(v1_model)
    addendum = valid_addendum(v3_model)

    write_json(selection_file, selection)
    write_json(addendum_file, addendum)

    monkeypatch.setattr(
        inference_module,
        "SELECTION_FILE",
        selection_file,
    )
    monkeypatch.setattr(
        inference_module,
        "V3_ADDENDUM_FILE",
        addendum_file,
    )

    return (
        selection_file,
        addendum_file,
        v1_model,
        v3_model,
        selection,
        addendum,
    )


def test_rejects_missing_model_selection_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-selection.json"

    monkeypatch.setattr(
        inference_module,
        "SELECTION_FILE",
        missing,
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing model selection record",
    ):
        InferenceService()


def test_rejects_missing_v3_addendum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selection_file = tmp_path / "selection.json"
    selection_file.write_text("{}")

    monkeypatch.setattr(
        inference_module,
        "SELECTION_FILE",
        selection_file,
    )
    monkeypatch.setattr(
        inference_module,
        "V3_ADDENDUM_FILE",
        tmp_path / "missing-addendum.json",
    )

    with pytest.raises(
        FileNotFoundError,
        match="Missing V3 addendum",
    ):
        InferenceService()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda selection, addendum: selection[
                "selected_primary_model"
            ].update({"name": "unexpected_model"}),
            "Frozen primary model is not baseline_cnn_v1",
        ),
        (
            lambda selection, addendum: addendum[
                "deployment_policy"
            ].update(
                {
                    "primary_prediction_source": (
                        "unexpected_model"
                    )
                }
            ),
            "Unexpected primary prediction source",
        ),
        (
            lambda selection, addendum: addendum[
                "deployment_policy"
            ].update(
                {"automatic_override_allowed": True}
            ),
            "Automatic V3 override must remain disabled",
        ),
        (
            lambda selection, addendum: addendum[
                "deployment_policy"
            ].update(
                {"automatic_ensemble_allowed": True}
            ),
            "Automatic ensemble must remain disabled",
        ),
        (
            lambda selection, addendum: addendum[
                "deployment_policy"
            ].update(
                {
                    "recommended_warning_condition": (
                        "unexpected_condition"
                    )
                }
            ),
            "Unexpected V3 warning condition",
        ),
    ],
)
def test_rejects_invalid_frozen_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    (
        selection_file,
        addendum_file,
        _,
        _,
        selection,
        addendum,
    ) = configure_files(monkeypatch, tmp_path)

    mutation(selection, addendum)

    write_json(selection_file, selection)
    write_json(addendum_file, addendum)

    with pytest.raises(
        RuntimeError,
        match=message,
    ):
        InferenceService()


@pytest.mark.parametrize(
    ("missing_model", "message"),
    [
        ("v1", "Missing V1 model"),
        ("v3", "Missing V3 model"),
    ],
)
def test_rejects_missing_model_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing_model: str,
    message: str,
) -> None:
    (
        selection_file,
        addendum_file,
        v1_model,
        v3_model,
        selection,
        addendum,
    ) = configure_files(monkeypatch, tmp_path)

    if missing_model == "v1":
        v1_model.unlink()
    else:
        v3_model.unlink()

    write_json(selection_file, selection)
    write_json(addendum_file, addendum)

    with pytest.raises(
        FileNotFoundError,
        match=message,
    ):
        InferenceService()


def test_loads_valid_frozen_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (
        _,
        _,
        v1_model,
        v3_model,
        _,
        _,
    ) = configure_files(monkeypatch, tmp_path)

    service = InferenceService()

    assert service.v1_model_path == v1_model
    assert service.v3_model_path == v3_model
    assert service.v1_threshold == pytest.approx(0.25)
    assert service.v3_threshold == pytest.approx(0.75)
    assert service.warning_condition == (
        "v1_predicts_normal_and_v3_predicts_pneumonia"
    )
