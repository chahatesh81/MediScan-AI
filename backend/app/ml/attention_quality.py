from __future__ import annotations

from typing import Any

import numpy as np


HIGH_RISK_BORDER_THRESHOLD = 0.75
HIGH_RISK_THORAX_THRESHOLD = 0.30

ELEVATED_RISK_BORDER_THRESHOLD = 0.70


def calculate_attention_regions(
    heatmap: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the frozen geometric attention diagnostics.

    These definitions exactly reproduce the original V1
    Grad-CAM attention audit.

    The regions are engineering proxies only.
    They are not anatomical lung masks.
    """

    if heatmap.ndim != 2:
        raise ValueError(
            "Attention heatmap must be a 2D array."
        )

    if heatmap.size == 0:
        raise ValueError(
            "Attention heatmap must not be empty."
        )

    if not np.isfinite(heatmap).all():
        raise ValueError(
            "Attention heatmap contains NaN or infinity."
        )

    if np.any(heatmap < 0.0):
        raise ValueError(
            "Attention heatmap contains negative values."
        )

    height, width = heatmap.shape

    border_y = max(
        1,
        int(height * 0.20),
    )

    border_x = max(
        1,
        int(width * 0.20),
    )

    border_mask = np.ones(
        (height, width),
        dtype=bool,
    )

    border_mask[
        border_y : height - border_y,
        border_x : width - border_x,
    ] = False

    thorax_mask = np.zeros(
        (height, width),
        dtype=bool,
    )

    thorax_mask[
        int(height * 0.15) : int(height * 0.90),
        int(width * 0.15) : int(width * 0.85),
    ] = True

    total_energy = float(
        np.sum(heatmap)
    )

    if total_energy <= 0.0:
        return {
            "border_energy_ratio": 0.0,
            "thorax_energy_ratio": 0.0,
            "peak_in_border": 0.0,
        }

    border_energy = float(
        np.sum(
            heatmap[border_mask]
        )
    )

    thorax_energy = float(
        np.sum(
            heatmap[thorax_mask]
        )
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(heatmap),
        heatmap.shape,
    )

    peak_in_border = float(
        border_mask[
            peak_y,
            peak_x,
        ]
    )

    return {
        "border_energy_ratio": (
            border_energy / total_energy
        ),
        "thorax_energy_ratio": (
            thorax_energy / total_energy
        ),
        "peak_in_border": peak_in_border,
    }


def classify_attention_quality(
    border_energy_ratio: float,
    thorax_energy_ratio: float,
    peak_in_border: float,
) -> dict[str, Any]:
    """
    Classify explanation quality using thresholds frozen
    from the V1 Grad-CAM audit distribution.

    No status claims that an explanation is clinically
    reliable or anatomically valid.
    """

    if not all(
        np.isfinite(
            [
                border_energy_ratio,
                thorax_energy_ratio,
                peak_in_border,
            ]
        )
    ):
        raise ValueError(
            "Attention quality metrics must be finite."
        )

    high_shortcut_risk = (
        border_energy_ratio
        >= HIGH_RISK_BORDER_THRESHOLD
        and thorax_energy_ratio
        <= HIGH_RISK_THORAX_THRESHOLD
    )

    elevated_shortcut_risk = (
        peak_in_border >= 0.5
        or border_energy_ratio
        >= ELEVATED_RISK_BORDER_THRESHOLD
    )

    if high_shortcut_risk:
        quality_status = (
            "HIGH_SHORTCUT_RISK"
        )

        warning_code = (
            "EXPLANATION_HIGH_SHORTCUT_RISK"
        )

        display_warning = True

    elif elevated_shortcut_risk:
        quality_status = (
            "ELEVATED_SHORTCUT_RISK"
        )

        warning_code = (
            "EXPLANATION_ELEVATED_SHORTCUT_RISK"
        )

        display_warning = True

    else:
        quality_status = (
            "LIMITED_SPATIAL_RELIABILITY"
        )

        warning_code = (
            "EXPLANATION_LIMITED_RELIABILITY"
        )

        display_warning = True

    return {
        "quality_status": quality_status,
        "display_warning": display_warning,
        "warning_code": warning_code,
    }


def analyze_attention_quality(
    heatmap: np.ndarray,
    explanation_mode: str,
) -> dict[str, Any]:
    """
    Run the complete frozen explanation-quality analysis.
    """

    metrics = calculate_attention_regions(
        heatmap
    )

    classification = (
        classify_attention_quality(
            border_energy_ratio=metrics[
                "border_energy_ratio"
            ],
            thorax_energy_ratio=metrics[
                "thorax_energy_ratio"
            ],
            peak_in_border=metrics[
                "peak_in_border"
            ],
        )
    )

    attribution_note = None

    if explanation_mode == "absolute_attribution":
        attribution_note = (
            "Heatmap shows attribution magnitude, "
            "not exclusively positive evidence "
            "for pneumonia."
        )

    return {
        **metrics,
        **classification,
        "explanation_mode": explanation_mode,
        "attribution_note": attribution_note,
        "region_definition": (
            "geometric_proxy_not_anatomical_lung_mask"
        ),
    }
