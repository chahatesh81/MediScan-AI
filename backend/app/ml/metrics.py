import tensorflow as tf


def build_binary_metrics() -> list[tf.keras.metrics.Metric]:
    return [
        tf.keras.metrics.BinaryAccuracy(
            name="accuracy",
            threshold=0.5,
        ),
        tf.keras.metrics.Precision(
            name="precision",
            thresholds=0.5,
        ),
        tf.keras.metrics.Recall(
            name="recall",
            thresholds=0.5,
        ),
        tf.keras.metrics.AUC(
            name="roc_auc",
            curve="ROC",
        ),
        tf.keras.metrics.AUC(
            name="pr_auc",
            curve="PR",
        ),
        tf.keras.metrics.TruePositives(
            name="true_positives",
            thresholds=0.5,
        ),
        tf.keras.metrics.TrueNegatives(
            name="true_negatives",
            thresholds=0.5,
        ),
        tf.keras.metrics.FalsePositives(
            name="false_positives",
            thresholds=0.5,
        ),
        tf.keras.metrics.FalseNegatives(
            name="false_negatives",
            thresholds=0.5,
        ),
    ]