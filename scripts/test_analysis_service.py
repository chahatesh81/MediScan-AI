from __future__ import annotations

from pathlib import Path

from backend.app.services.analysis_service import (
    analysis_service,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
    / "PNEUMONIA"
    / "person155_bacteria_730.jpeg"
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
        "MEDISCAN AI — COMBINED ANALYSIS "
        "SERVICE TEST"
    )
    print("=" * 70)

    image_bytes = TEST_IMAGE.read_bytes()

    result = analysis_service.analyze_bytes(
        image_bytes
    )

    primary = result["primary_prediction"]
    secondary = result["secondary_signal"]
    decision = result["decision"]
    explanation = result["explanation"]
    quality = result[
        "explanation_quality"
    ]
    endpoints = result[
        "visualization_endpoints"
    ]

    print()
    print(
        f"Primary model:     "
        f"{primary['model']}"
    )
    print(
        f"Primary label:     "
        f"{primary['label']}"
    )
    print(
        f"Primary prob:      "
        f"{primary['probability']:.12f}"
    )
    print(
        f"Secondary model:   "
        f"{secondary['model']}"
    )
    print(
        f"Final source:      "
        f"{decision['source']}"
    )
    print(
        f"Explanation mode: "
        f"{explanation['mode']}"
    )
    print(
        f"Quality status:    "
        f"{quality['quality_status']}"
    )
    print(
        f"Display warning:   "
        f"{quality['display_warning']}"
    )

    require(
        primary["model"]
        == "baseline_cnn_v1",
        "Unexpected primary model.",
    )

    require(
        decision["source"]
        == "baseline_cnn_v1",
        "Authoritative source changed.",
    )

    require(
        secondary[
            "automatic_override_allowed"
        ]
        is False,
        "V3 override policy changed.",
    )

    require(
        explanation["method"]
        == "gradcam",
        "Unexpected explanation method.",
    )

    require(
        quality["explanation_mode"]
        == explanation["mode"],
        (
            "Explanation/quality mode "
            "mismatch."
        ),
    )

    require(
        endpoints["heatmap"]
        == "/api/v1/explain",
        "Unexpected heatmap endpoint.",
    )

    require(
        endpoints["overlay"]
        == "/api/v1/explain/overlay",
        "Unexpected overlay endpoint.",
    )

    print()
    print("Primary policy: PASS")
    print("V3 safety policy: PASS")
    print("Explanation metadata: PASS")
    print("Quality propagation: PASS")
    print("Visualization contract: PASS")

    print()
    print("=" * 70)
    print(
        "COMBINED ANALYSIS SERVICE "
        "STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
