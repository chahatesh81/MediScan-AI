import tensorflow as tf

from backend.app.ml.baseline_cnn import (
    build_baseline_cnn,
)
from backend.app.ml.runtime import (
    configure_training_runtime,
)


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — TRAINING RUNTIME TEST")
    print("=" * 60)

    runtime = configure_training_runtime()

    print(f"GPU count:       {runtime['gpu_count']}")
    print(f"Policy:          {runtime['policy']}")
    print(f"Compute dtype:   {runtime['compute_dtype']}")
    print(f"Variable dtype:  {runtime['variable_dtype']}")

    model = build_baseline_cnn()

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

    print(f"Model output dtype: {predictions.dtype}")

    assert runtime["gpu_count"] >= 1
    assert runtime["policy"] == "mixed_float16"
    assert runtime["compute_dtype"] == "float16"
    assert runtime["variable_dtype"] == "float32"

    # The final model layer was explicitly configured as
    # float32 for numerically stable loss and metrics.
    assert predictions.dtype == tf.float32

    print("\nTRAINING RUNTIME STATUS: READY")


if __name__ == "__main__":
    main()