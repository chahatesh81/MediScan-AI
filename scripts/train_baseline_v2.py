import json
import time
from datetime import datetime

import tensorflow as tf

from backend.app.core.config import (
    PROJECT_ROOT,
    RANDOM_SEED,
)
from backend.app.core.reproducibility import set_global_seed
from backend.app.ml.baseline_cnn_v2 import build_baseline_cnn_v2
from backend.app.ml.callbacks import build_training_callbacks
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.class_weights import calculate_class_weights
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
)


EXPERIMENT_NAME = "baseline_cnn_v2"

BATCH_SIZE = 16
MAX_EPOCHS = 25

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

METADATA_FILE = (
    RESULTS_DIR
    / "baseline_v2_training_metadata.json"
)


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE CNN V2 TRAINING"
    )
    print("=" * 70)

    set_global_seed(
        RANDOM_SEED
    )

    runtime = (
        configure_training_runtime()
    )

    print(
        f"GPU count:        "
        f"{runtime['gpu_count']}"
    )

    print(
        f"Precision policy: "
        f"{runtime['policy']}"
    )

    print(
        f"Batch size:       "
        f"{BATCH_SIZE}"
    )

    print(
        f"Maximum epochs:   "
        f"{MAX_EPOCHS}"
    )

    print(
        "Pipeline:         "
        "cached artifact-aware v2"
    )

    print(
        "Architecture:     "
        "controlled match to v1"
    )

    train_dataset = (
        build_cached_dataset_v2(
            split_name="train",
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    )

    val_dataset = (
        build_cached_dataset_v2(
            split_name="val",
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    )

    model = (
        build_baseline_cnn_v2()
    )

    class_weights = (
        calculate_class_weights()
    )

    (
        callbacks,
        experiment_dir,
    ) = build_training_callbacks(
        EXPERIMENT_NAME
    )

    print(
        f"Parameters:       "
        f"{model.count_params():,}"
    )

    print(
        f"Class weights:    "
        f"{class_weights}"
    )

    print(
        f"Experiment log:   "
        f"{experiment_dir}"
    )

    print(
        "\nStarting controlled "
        "v2 training...\n"
    )

    start_time = (
        time.perf_counter()
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    epochs_completed = len(
        history.history["loss"]
    )

    best_epoch = (
        int(
            tf.argmax(
                history.history[
                    "val_roc_auc"
                ]
            ).numpy()
        )
        + 1
    )

    best_val_roc_auc = max(
        history.history[
            "val_roc_auc"
        ]
    )

    best_val_pr_auc = max(
        history.history[
            "val_pr_auc"
        ]
    )

    checkpoint_path = (
        PROJECT_ROOT
        / "models"
        / "baseline_cnn_v2_best.keras"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Expected V2 checkpoint "
            f"was not created: "
            f"{checkpoint_path}"
        )

    metadata = {
        "experiment": (
            EXPERIMENT_NAME
        ),
        "experiment_type": (
            "shortcut_learning_repair"
        ),
        "timestamp": (
            datetime.now().isoformat()
        ),
        "random_seed": (
            RANDOM_SEED
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "maximum_epochs": (
            MAX_EPOCHS
        ),
        "epochs_completed": (
            epochs_completed
        ),
        "best_epoch_by_val_roc_auc": (
            best_epoch
        ),
        "best_val_roc_auc": float(
            best_val_roc_auc
        ),
        "best_val_pr_auc": float(
            best_val_pr_auc
        ),
        "training_seconds": float(
            elapsed_seconds
        ),
        "parameters": int(
            model.count_params()
        ),
        "class_weights": {
            str(key): float(value)
            for key, value
            in class_weights.items()
        },
        "precision_policy": (
            runtime["policy"]
        ),
        "gpu_count": (
            runtime["gpu_count"]
        ),
        "training_pipeline": (
            "baseline_v2_cached"
        ),
        "preprocessing": {
            "artifact_aware_foreground_crop": True,
            "aspect_ratio_preserved": True,
            "target_size": [
                224,
                224,
            ],
            "cache_format": (
                "JPEG"
            ),
            "cache_jpeg_quality": (
                95
            ),
        },
        "controlled_comparison": {
            "reference_model": (
                "baseline_cnn"
            ),
            "architecture_unchanged": (
                True
            ),
            "parameter_count_unchanged": (
                True
            ),
            "optimizer_unchanged": (
                True
            ),
            "loss_unchanged": (
                True
            ),
            "metrics_unchanged": (
                True
            ),
            "class_weights_unchanged": (
                True
            ),
            "batch_size_unchanged": (
                True
            ),
            "maximum_epochs_unchanged": (
                True
            ),
            "preprocessing_changed": (
                True
            ),
            "augmentation_changed": (
                True
            ),
        },
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "BASELINE V2 TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Epochs completed:    "
        f"{epochs_completed}"
    )

    print(
        f"Best epoch:          "
        f"{best_epoch}"
    )

    print(
        f"Best val ROC-AUC:    "
        f"{best_val_roc_auc:.4f}"
    )

    print(
        f"Best val PR-AUC:     "
        f"{best_val_pr_auc:.4f}"
    )

    print(
        f"Training time:       "
        f"{elapsed_seconds / 60:.2f} "
        f"minutes"
    )

    print(
        f"Metadata:            "
        f"{METADATA_FILE}"
    )

    print(
        f"Best checkpoint:     "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()