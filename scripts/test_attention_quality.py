from __future__ import annotations

import numpy as np

from backend.app.ml.attention_quality import (
    analyze_attention_quality,
    calculate_attention_regions,
    classify_attention_quality,
)


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ATTENTION QUALITY COMPONENT TEST"
    )
    print("=" * 70)

    heatmap = np.ones(
        (28, 28),
        dtype=np.float32,
    )

    metrics = calculate_attention_regions(
        heatmap
    )

    print("\nUNIFORM HEATMAP")
    print(
        "Border energy:",
        f"{metrics['border_energy_ratio']:.6f}",
    )
    print(
        "Thorax energy:",
        f"{metrics['thorax_energy_ratio']:.6f}",
    )
    print(
        "Peak in border:",
        metrics["peak_in_border"],
    )

    high = classify_attention_quality(
        border_energy_ratio=0.90,
        thorax_energy_ratio=0.10,
        peak_in_border=1.0,
    )

    elevated = classify_attention_quality(
        border_energy_ratio=0.65,
        thorax_energy_ratio=0.45,
        peak_in_border=1.0,
    )

    limited = classify_attention_quality(
        border_energy_ratio=0.60,
        thorax_energy_ratio=0.50,
        peak_in_border=0.0,
    )

    require(
        high["quality_status"]
        == "HIGH_SHORTCUT_RISK",
        "High-risk classification failed.",
    )

    require(
        elevated["quality_status"]
        == "ELEVATED_SHORTCUT_RISK",
        "Elevated-risk classification failed.",
    )

    require(
        limited["quality_status"]
        == "LIMITED_SPATIAL_RELIABILITY",
        "Limited-reliability classification failed.",
    )

    analysis = analyze_attention_quality(
        heatmap=heatmap,
        explanation_mode=(
            "absolute_attribution"
        ),
    )

    require(
        analysis["attribution_note"]
        is not None,
        "Absolute-attribution note is missing.",
    )

    require(
        analysis["region_definition"]
        == (
            "geometric_proxy_not_anatomical_lung_mask"
        ),
        "Region definition is incorrect.",
    )

    print("\nCLASSIFICATION TESTS")
    print(
        "High risk:",
        high["quality_status"],
    )
    print(
        "Elevated:",
        elevated["quality_status"],
    )
    print(
        "Limited:",
        limited["quality_status"],
    )

    print("\nABSOLUTE ATTRIBUTION NOTE")
    print(
        analysis["attribution_note"]
    )

    print()
    print("=" * 70)
    print(
        "ATTENTION QUALITY COMPONENT "
        "STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
