import math

import tensorflow as tf

from backend.app.ml.baseline_cnn import build_baseline_cnn
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.class_weights import calculate_class_weights
from backend.app.services.data_pipeline import build_dataset


TRAIN_BATCH_SIZE = 8
TRAIN_STEPS = 3
VALIDATION_STEPS = 2


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — TRAINING SMOKE TEST")
    print("=" * 60)

    runtime = configure_training_runtime()

    print(f"GPU count: {runtime['gpu_count']}")
    print(f"Policy:    {runtime['policy']}")

    train_dataset = build_dataset(
        split="train",
        batch_size=TRAIN_BATCH_SIZE,
    )

    val_dataset = build_dataset(
        split="val",
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
    )

    model = build_baseline_cnn()
    class_weights = calculate_class_weights()

    initial_weights = [
        weight.numpy().copy()
        for weight in model.trainable_weights
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        steps_per_epoch=TRAIN_STEPS,
        validation_steps=VALIDATION_STEPS,
        class_weight=class_weights,
        verbose=1,
    )

    weights_changed = any(
        not tf.reduce_all(
            tf.equal(before, after)
        ).numpy()
        for before, after in zip(
            initial_weights,
            model.trainable_weights,
        )
    )

    loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    print("\nSmoke-test results:")
    print(f"Training loss:       {loss:.6f}")
    print(f"Validation loss:     {val_loss:.6f}")
    print(f"Trainable weights changed: {weights_changed}")

    assert math.isfinite(loss)
    assert math.isfinite(val_loss)
    assert weights_changed

    print("\nTRAINING SMOKE TEST STATUS: READY")


if __name__ == "__main__":
    main()