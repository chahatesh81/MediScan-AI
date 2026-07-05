from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.services.analysis_service import (
    analysis_service,
)
from backend.app.services.inference_service import (
    inference_service,
)


TEST_IMAGE = Path(
    "data/raw/chest_xray/test/"
    "PNEUMONIA/person155_bacteria_730.jpeg"
)


class CountingModel:
    """
    Transparent model proxy that counts direct calls.
    """

    def __init__(
        self,
        model: Any,
        name: str,
    ) -> None:
        self._model = model
        self.name = name
        self.call_count = 0

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        self.call_count += 1

        return self._model(
            *args,
            **kwargs,
        )

    def __getattr__(
        self,
        name: str,
    ) -> Any:
        return getattr(
            self._model,
            name,
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
        "MEDISCAN AI — ANALYSIS FORWARD-PASS TEST"
    )
    print("=" * 70)

    require(
        TEST_IMAGE.is_file(),
        f"Missing test image: {TEST_IMAGE}",
    )

    image_bytes = TEST_IMAGE.read_bytes()

    inference_service.load_models()

    original_v1 = inference_service._v1_model
    original_v3 = inference_service._v3_model

    require(
        original_v1 is not None,
        "V1 model is not loaded.",
    )

    require(
        original_v3 is not None,
        "V3 model is not loaded.",
    )

    counting_v1 = CountingModel(
        original_v1,
        "baseline_cnn_v1",
    )

    counting_v3 = CountingModel(
        original_v3,
        "advanced_v3",
    )

    try:
        inference_service._v1_model = counting_v1
        inference_service._v3_model = counting_v3

        result = analysis_service.analyze_bytes(
            image_bytes
        )

        print()
        print(
            "Primary model:  "
            f"{result['primary_prediction']['model']}"
        )
        print(
            "Primary label:  "
            f"{result['primary_prediction']['label']}"
        )
        print(
            "V1 direct calls: "
            f"{counting_v1.call_count}"
        )
        print(
            "V3 direct calls: "
            f"{counting_v3.call_count}"
        )

        require(
            counting_v1.call_count == 1,
            (
                "Expected exactly one direct V1 "
                "forward pass."
            ),
        )

        require(
            counting_v3.call_count == 1,
            (
                "Expected exactly one direct V3 "
                "forward pass."
            ),
        )

        require(
            result[
                "decision"
            ]["source"]
            == "baseline_cnn_v1",
            "Primary decision source changed.",
        )

        require(
            result[
                "secondary_signal"
            ]["automatic_override_allowed"]
            is False,
            "V3 override policy changed.",
        )

        print()
        print("V1 single forward pass: PASS")
        print("V3 single forward pass: PASS")
        print("Deployment policy preserved: PASS")

    finally:
        inference_service._v1_model = original_v1
        inference_service._v3_model = original_v3

    print()
    print("=" * 70)
    print(
        "ANALYSIS FORWARD-PASS TEST STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
