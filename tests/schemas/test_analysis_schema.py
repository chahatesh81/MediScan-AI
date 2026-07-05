from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas.analysis import AttentionQuality


def build_attention_quality(
    peak_in_border: object,
) -> AttentionQuality:
    return AttentionQuality(
        border_energy_ratio=0.25,
        thorax_energy_ratio=0.75,
        peak_in_border=peak_in_border,
        quality_status="LIMITED_SPATIAL_RELIABILITY",
        display_warning=False,
        warning_code=None,
        explanation_mode="positive_gradcam",
        attribution_note=None,
        region_definition="central thorax region",
    )


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (False, False),
        (True, True),
        (0.0, False),
        (1.0, True),
    ],
)
def test_peak_in_border_normalizes_to_boolean(
    input_value: object,
    expected: bool,
) -> None:
    result = build_attention_quality(input_value)

    assert result.peak_in_border is expected
    assert isinstance(result.peak_in_border, bool)


def test_peak_in_border_serializes_as_json_boolean() -> None:
    result = build_attention_quality(1.0)

    payload = result.model_dump(mode="json")

    assert payload["peak_in_border"] is True
    assert isinstance(payload["peak_in_border"], bool)


def test_peak_in_border_rejects_non_boolean_numeric_value() -> None:
    with pytest.raises(ValidationError):
        build_attention_quality(0.5)
