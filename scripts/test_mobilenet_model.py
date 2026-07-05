import tensorflow as tf

from backend.app.ml.mobilenet_v2 import (
    build_mobilenet_v2,
)
from backend.app.ml.runtime import (
    configure_training_runtime,
)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — MOBILENETV2 MODEL TEST")
    print("=" * 70)

    runtime = configure_training_runtime()

    model = build_mobilenet_v2(
        trainable_backbone=False,
    )

    backbone = model.get_layer("mobilenetv2_1.00_224")

    dummy_batch = tf.random.uniform(
        shape=(2, 224, 224, 3),
        minval=0,
        maxval=255,
        dtype=tf.float32,
    )

    predictions = model(
        dummy_batch,
        training=False,
    )

    trainable_parameters = sum(
        tf.size(weight).numpy()
        for weight in model.trainable_weights
    )

    non_trainable_parameters = sum(
        tf.size(weight).numpy()
        for weight in model.non_trainable_weights
    )

    print(f"GPU count:                {runtime['gpu_count']}")
    print(f"Precision policy:         {runtime['policy']}")
    print(f"Model name:               {model.name}")
    print(f"Total parameters:         {model.count_params():,}")
    print(f"Trainable parameters:     {trainable_parameters:,}")
    print(f"Non-trainable parameters: {non_trainable_parameters:,}")
    print(f"Backbone trainable:       {backbone.trainable}")
    print(f"Prediction shape:         {predictions.shape}")
    print(f"Prediction dtype:         {predictions.dtype}")

    assert runtime["policy"] == "mixed_float16"
    assert backbone.trainable is False
    assert predictions.shape == (2, 1)
    assert predictions.dtype == tf.float32
    assert trainable_parameters < non_trainable_parameters

    print("\nMOBILENETV2 MODEL STATUS: READY")


if __name__ == "__main__":
    main()