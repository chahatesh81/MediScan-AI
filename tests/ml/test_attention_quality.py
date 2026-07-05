from __future__ import annotations

import numpy as np
import pytest

from backend.app.ml.attention_quality import (
    analyze_attention_quality,
    calculate_attention_regions,
    classify_attention_quality,
)


pytestmark = pytest.mark.unit


def test_rejects_non_2d_heatmap() -> None:
    heatmap = np.zeros((4, 4, 1), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="must be a 2D array",
    ):
        calculate_attention_regions(heatmap)


def test_rejects_empty_heatmap() -> None:
    heatmap = np.empty((0, 0), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        calculate_attention_regions(heatmap)


@pytest.mark.parametrize(
    "heatmap",
    [
        np.array(
            [[0.0, np.nan]],
            dtype=np.float32,
        ),
        np.array(
            [[0.0, np.inf]],
            dtype=np.float32,
        ),
    ],
)
def test_rejects_non_finite_heatmap(
    heatmap: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match="NaN or infinity",
    ):
        calculate_attention_regions(heatmap)


def test_rejects_negative_attention() -> None:
    heatmap = np.array(
        [[0.0, -0.1]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="negative values",
    ):
        calculate_attention_regions(heatmap)


def test_zero_energy_returns_zero_metrics() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float32)

    assert calculate_attention_regions(heatmap) == {
        "border_energy_ratio": 0.0,
        "thorax_energy_ratio": 0.0,
        "peak_in_border": 0.0,
    }


def test_center_attention_is_not_border_peak() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[5, 5] = 1.0

    metrics = calculate_attention_regions(heatmap)

    assert metrics["border_energy_ratio"] == 0.0
    assert metrics["thorax_energy_ratio"] == 1.0
    assert metrics["peak_in_border"] == 0.0


def test_border_attention_is_detected() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[0, 0] = 1.0

    metrics = calculate_attention_regions(heatmap)

    assert metrics["border_energy_ratio"] == 1.0
    assert metrics["thorax_energy_ratio"] == 0.0
    assert metrics["peak_in_border"] == 1.0


def test_high_shortcut_risk_has_priority() -> None:
    result = classify_attention_quality(
        border_energy_ratio=0.80,
        thorax_energy_ratio=0.20,
        peak_in_border=1.0,
    )

    assert result == {
        "quality_status": "HIGH_SHORTCUT_RISK",
        "display_warning": True,
        "warning_code": (
            "EXPLANATION_HIGH_SHORTCUT_RISK"
        ),
    }


@pytest.mark.parametrize(
    (
        "border_energy_ratio",
        "thorax_energy_ratio",
        "peak_in_border",
    ),
    [
        (0.70, 0.80, 0.0),
        (0.20, 0.80, 1.0),
    ],
)
def test_elevated_shortcut_risk(
    border_energy_ratio: float,
    thorax_energy_ratio: float,
    peak_in_border: float,
) -> None:
    result = classify_attention_quality(
        border_energy_ratio=border_energy_ratio,
        thorax_energy_ratio=thorax_energy_ratio,
        peak_in_border=peak_in_border,
    )

    assert result["quality_status"] == (
        "ELEVATED_SHORTCUT_RISK"
    )
    assert result["display_warning"] is True
    assert result["warning_code"] == (
        "EXPLANATION_ELEVATED_SHORTCUT_RISK"
    )


def test_limited_spatial_reliability_is_still_warned() -> None:
    result = classify_attention_quality(
        border_energy_ratio=0.20,
        thorax_energy_ratio=0.80,
        peak_in_border=0.0,
    )

    assert result == {
        "quality_status": (
            "LIMITED_SPATIAL_RELIABILITY"
        ),
        "display_warning": True,
        "warning_code": (
            "EXPLANATION_LIMITED_RELIABILITY"
        ),
    }


@pytest.mark.parametrize(
    "bad_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_quality_classifier_rejects_non_finite_metrics(
    bad_value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        classify_attention_quality(
            border_energy_ratio=bad_value,
            thorax_energy_ratio=0.5,
            peak_in_border=0.0,
        )


def test_absolute_attribution_adds_interpretation_note() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[5, 5] = 1.0

    result = analyze_attention_quality(
        heatmap,
        explanation_mode="absolute_attribution",
    )

    assert result["explanation_mode"] == (
        "absolute_attribution"
    )
    assert result["attribution_note"] is not None
    assert "attribution magnitude" in result[
        "attribution_note"
    ]


def test_positive_gradcam_has_no_attribution_note() -> None:
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[5, 5] = 1.0

    result = analyze_attention_quality(
        heatmap,
        explanation_mode="positive_gradcam",
    )

    assert result["explanation_mode"] == (
        "positive_gradcam"
    )
    assert result["attribution_note"] is None
    assert result["region_definition"] == (
        "geometric_proxy_not_anatomical_lung_mask"
    )
