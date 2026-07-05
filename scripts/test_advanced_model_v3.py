from __future__ import annotations

import numpy as np
import tensorflow as tf

from backend.app.ml.advanced_model_v3 import (
    build_advanced_model_v3,
    count_non_trainable_parameters,
    count_trainable_parameters,
    enable_partial_fine_tuning,
    get_backbone,
)
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
)


BATCH_SIZE = 2
UNFREEZE_LAST_LAYERS = 40


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ADVANCED MODEL V3 COMPONENT TEST"
    )
    print("=" * 70)

    configure_training_runtime()

    print("\nBuilding frozen-backbone model...")

    model = build_advanced_model_v3(
        backbone_trainable=False,
    )

    backbone = get_backbone(model)

    print(f"\nModel name:          {model.name}")
    print(f"Backbone name:       {backbone.name}")
    print(f"Input shape:         {model.input_shape}")
    print(f"Output shape:        {model.output_shape}")
    print(
        f"Total parameters:    "
        f"{model.count_params():,}"
    )
    print(
        f"Trainable params:    "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"Non-trainable:       "
        f"{count_non_trainable_parameters(model):,}"
    )
    print(
        f"Backbone trainable:  "
        f"{backbone.trainable}"
    )

    if backbone.trainable:
        raise RuntimeError(
            "Backbone must be frozen during Stage 1."
        )

    print("\nLoading cached V2 batch...")

    dataset = build_cached_dataset_v2(
        split_name="train",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    images, labels = next(iter(dataset))

    print(f"Image shape:         {images.shape}")
    print(f"Label shape:         {labels.shape}")
    print(f"Image dtype:         {images.dtype}")
    print(
        f"Image range:         "
        f"{tf.reduce_min(images).numpy():.2f} "
        f"to "
        f"{tf.reduce_max(images).numpy():.2f}"
    )

    print("\nRunning frozen forward pass...")

    predictions = model(
        images,
        training=False,
    )

    prediction_values = (
        predictions.numpy().reshape(-1)
    )

    print(
        f"Prediction shape:    "
        f"{predictions.shape}"
    )
    print(
        f"Prediction dtype:    "
        f"{predictions.dtype}"
    )
    print(
        "Predictions:         "
        f"{prediction_values.tolist()}"
    )

    if predictions.shape != (BATCH_SIZE, 1):
        raise RuntimeError(
            "Unexpected prediction shape."
        )

    if not np.all(
        np.isfinite(prediction_values)
    ):
        raise RuntimeError(
            "Predictions contain non-finite values."
        )

    if not np.all(
        (
            prediction_values >= 0.0
        )
        & (
            prediction_values <= 1.0
        )
    ):
        raise RuntimeError(
            "Predictions are outside [0, 1]."
        )

    print("\nTesting partial fine-tuning...")

    fine_tune_info = (
        enable_partial_fine_tuning(
            model=model,
            unfreeze_last_layers=(
                UNFREEZE_LAST_LAYERS
            ),
        )
    )

    backbone = get_backbone(model)

    print(
        f"Total backbone layers:     "
        f"{fine_tune_info['total_backbone_layers']}"
    )
    print(
        f"Requested final layers:    "
        f"{fine_tune_info['requested_unfreeze_layers']}"
    )
    print(
        f"Trainable backbone layers: "
        f"{fine_tune_info['trainable_backbone_layers']}"
    )
    print(
        f"Frozen backbone layers:    "
        f"{fine_tune_info['frozen_backbone_layers']}"
    )
    print(
        f"Trainable params after FT: "
        f"{count_trainable_parameters(model):,}"
    )

    if (
        fine_tune_info[
            "trainable_backbone_layers"
        ]
        <= 0
    ):
        raise RuntimeError(
            "No backbone layers were unfrozen."
        )

    batch_norm_trainable = [
        layer.name
        for layer in backbone.layers
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        )
        and layer.trainable
    ]

    print(
        f"Trainable backbone BN:     "
        f"{len(batch_norm_trainable)}"
    )

    if batch_norm_trainable:
        raise RuntimeError(
            "Backbone BatchNormalization "
            "layers must remain frozen."
        )

    print(
        "\nRunning fine-tuned forward pass..."
    )

    predictions_ft = model(
        images,
        training=False,
    )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(predictions_ft)
        )
    ):
        raise RuntimeError(
            "Fine-tuned model produced "
            "non-finite predictions."
        )

    print("\n" + "=" * 70)
    print(
        "ADVANCED MODEL V3 COMPONENT STATUS: READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()