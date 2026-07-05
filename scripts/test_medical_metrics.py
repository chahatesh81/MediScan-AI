import tensorflow as tf

from backend.app.ml.metrics import build_binary_metrics


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — MEDICAL METRICS TEST")
    print("=" * 60)

    y_true = tf.constant(
        [0, 0, 1, 1, 1, 0],
        dtype=tf.float32,
    )

    y_pred = tf.constant(
        [0.1, 0.7, 0.9, 0.8, 0.2, 0.1],
        dtype=tf.float32,
    )

    metrics = build_binary_metrics()

    results = {}

    for metric in metrics:
        metric.update_state(y_true, y_pred)
        results[metric.name] = float(metric.result().numpy())

    for name, value in results.items():
        print(f"{name:20s}: {value:.4f}")

    assert abs(results["accuracy"] - (4 / 6)) < 1e-6
    assert abs(results["precision"] - (2 / 3)) < 1e-6
    assert abs(results["recall"] - (2 / 3)) < 1e-6
    assert results["true_positives"] == 2
    assert results["true_negatives"] == 2
    assert results["false_positives"] == 1
    assert results["false_negatives"] == 1

    print("\nMEDICAL METRICS STATUS: READY")


if __name__ == "__main__":
    main()