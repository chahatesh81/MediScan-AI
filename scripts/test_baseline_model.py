import tensorflow as tf

from backend.app.ml.baseline_cnn import build_baseline_cnn


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — BASELINE CNN TEST")
    print("=" * 60)

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

    print(f"Model name:       {model.name}")
    print(f"Input shape:      {model.input_shape}")
    print(f"Output shape:     {model.output_shape}")
    print(f"Parameters:       {model.count_params():,}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction dtype: {predictions.dtype}")
    print(
        "Prediction range: "
        f"{tf.reduce_min(predictions).numpy():.4f} "
        f"to {tf.reduce_max(predictions).numpy():.4f}"
    )

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1)
    assert predictions.shape == (2, 1)
    assert predictions.dtype == tf.float32
    assert tf.reduce_all(predictions >= 0.0)
    assert tf.reduce_all(predictions <= 1.0)

    print("\nBASELINE CNN STATUS: READY")


if __name__ == "__main__":
    main()