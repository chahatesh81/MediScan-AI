from __future__ import annotations

import json
import time
from datetime import datetime

import tensorflow as tf

from backend.app.core.config import (
    PROJECT_ROOT,
    RANDOM_SEED,
)
from backend.app.core.reproducibility import (
    set_global_seed,
)
from backend.app.ml.advanced_model_v3 import (
    build_advanced_model_v3,
    count_non_trainable_parameters,
    count_trainable_parameters,
    enable_partial_fine_tuning,
)
from backend.app.ml.runtime import (
    configure_training_runtime,
)
from backend.app.services.class_weights import (
    calculate_class_weights,
)
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
)


EXPERIMENT_NAME = "advanced_v3"

BATCH_SIZE = 16

STAGE_1_MAX_EPOCHS = 15
STAGE_2_MAX_EPOCHS = 15

UNFREEZE_LAST_LAYERS = 40

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

LOGS_DIR = (
    PROJECT_ROOT
    / "logs"
    / "training"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

BEST_MODEL_FILE = (
    MODELS_DIR
    / "advanced_v3_best.keras"
)

STAGE_1_MODEL_FILE = (
    MODELS_DIR
    / "advanced_v3_stage1_best.keras"
)

METADATA_FILE = (
    RESULTS_DIR
    / "advanced_v3_training_metadata.json"
)


def build_stage_callbacks(
    stage_name: str,
    checkpoint_path,
    log_directory,
) -> list[tf.keras.callbacks.Callback]:

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_roc_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            log_directory
            / f"{stage_name}_training_log.csv"
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=(
                log_directory
                / f"{stage_name}_tensorboard"
            ),
            histogram_freq=0,
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]


def history_to_serializable(
    history: tf.keras.callbacks.History,
) -> dict[str, list[float]]:
    return {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }


def best_epoch_by_metric(
    history: tf.keras.callbacks.History,
    metric_name: str,
) -> int:
    values = history.history[
        metric_name
    ]

    return int(
        tf.argmax(values).numpy()
    ) + 1


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "ADVANCED MODEL V3 TWO-STAGE TRAINING"
    )
    print("=" * 70)

    set_global_seed(
        RANDOM_SEED
    )

    runtime = configure_training_runtime()

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    experiment_dir = (
        LOGS_DIR
        / f"{EXPERIMENT_NAME}_{timestamp}"
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"GPU count:             {runtime['gpu_count']}")
    print(f"Precision policy:      {runtime['policy']}")
    print(f"Batch size:            {BATCH_SIZE}")
    print(
        f"Stage 1 max epochs:    "
        f"{STAGE_1_MAX_EPOCHS}"
    )
    print(
        f"Stage 2 max epochs:    "
        f"{STAGE_2_MAX_EPOCHS}"
    )
    print(
        f"Fine-tune final layers:"
        f" {UNFREEZE_LAST_LAYERS}"
    )

    print("\nBuilding cached datasets...")

    train_dataset = build_cached_dataset_v2(
        split_name="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_dataset = build_cached_dataset_v2(
        split_name="val",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    class_weights = calculate_class_weights()

    print(f"Class weights:         {class_weights}")

    print("\nBuilding V3 model...")

    model = build_advanced_model_v3(
        backbone_trainable=False,
    )

    print(
        f"Total parameters:      "
        f"{model.count_params():,}"
    )
    print(
        f"Stage 1 trainable:     "
        f"{count_trainable_parameters(model):,}"
    )
    print(
        f"Stage 1 non-trainable: "
        f"{count_non_trainable_parameters(model):,}"
    )

    stage_1_callbacks = build_stage_callbacks(
        stage_name="stage1",
        checkpoint_path=STAGE_1_MODEL_FILE,
        log_directory=experiment_dir,
    )

    print("\n" + "=" * 70)
    print("STAGE 1 — FROZEN BACKBONE")
    print("=" * 70)

    stage_1_start = time.perf_counter()

    stage_1_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=STAGE_1_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=stage_1_callbacks,
        verbose=1,
    )

    stage_1_seconds = (
        time.perf_counter()
        - stage_1_start
    )

    stage_1_best_epoch = (
        best_epoch_by_metric(
            stage_1_history,
            "val_roc_auc",
        )
    )

    stage_1_best_auc = max(
        stage_1_history.history[
            "val_roc_auc"
        ]
    )

    print("\nStage 1 complete.")
    print(
        f"Best epoch:            "
        f"{stage_1_best_epoch}"
    )
    print(
        f"Best validation AUC:   "
        f"{stage_1_best_auc:.6f}"
    )
    print(
        f"Training time:         "
        f"{stage_1_seconds / 60:.2f} minutes"
    )

    print(
        "\nReloading best Stage 1 checkpoint..."
    )

    model = tf.keras.models.load_model(
        STAGE_1_MODEL_FILE,
    )

    print("\nEnabling partial fine-tuning...")

    fine_tune_info = (
        enable_partial_fine_tuning(
            model=model,
            unfreeze_last_layers=(
                UNFREEZE_LAST_LAYERS
            ),
        )
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
        f"Stage 2 trainable params:  "
        f"{count_trainable_parameters(model):,}"
    )

    stage_2_callbacks = build_stage_callbacks(
        stage_name="stage2",
        checkpoint_path=BEST_MODEL_FILE,
        log_directory=experiment_dir,
    )

    print("\n" + "=" * 70)
    print("STAGE 2 — PARTIAL FINE-TUNING")
    print("=" * 70)

    stage_2_start = time.perf_counter()

    stage_2_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=STAGE_2_MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=stage_2_callbacks,
        verbose=1,
    )

    stage_2_seconds = (
        time.perf_counter()
        - stage_2_start
    )

    stage_2_best_epoch = (
        best_epoch_by_metric(
            stage_2_history,
            "val_roc_auc",
        )
    )

    stage_2_best_auc = max(
        stage_2_history.history[
            "val_roc_auc"
        ]
    )

    total_seconds = (
        stage_1_seconds
        + stage_2_seconds
    )

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "architecture": (
            "EfficientNetV2B0"
        ),
        "pretrained_weights": "imagenet",
        "data_pipeline": (
            "baseline_v2_cached"
        ),
        "batch_size": BATCH_SIZE,
        "class_weights": {
            str(key): float(value)
            for key, value
            in class_weights.items()
        },
        "stage_1": {
            "maximum_epochs": (
                STAGE_1_MAX_EPOCHS
            ),
            "epochs_completed": len(
                stage_1_history.history[
                    "loss"
                ]
            ),
            "best_epoch_by_val_roc_auc": (
                stage_1_best_epoch
            ),
            "best_val_roc_auc": float(
                stage_1_best_auc
            ),
            "training_seconds": float(
                stage_1_seconds
            ),
            "history": (
                history_to_serializable(
                    stage_1_history
                )
            ),
        },
        "stage_2": {
            "maximum_epochs": (
                STAGE_2_MAX_EPOCHS
            ),
            "epochs_completed": len(
                stage_2_history.history[
                    "loss"
                ]
            ),
            "best_epoch_by_val_roc_auc": (
                stage_2_best_epoch
            ),
            "best_val_roc_auc": float(
                stage_2_best_auc
            ),
            "training_seconds": float(
                stage_2_seconds
            ),
            "fine_tuning": (
                fine_tune_info
            ),
            "history": (
                history_to_serializable(
                    stage_2_history
                )
            ),
        },
        "total_training_seconds": float(
            total_seconds
        ),
        "total_parameters": int(
            model.count_params()
        ),
        "final_trainable_parameters": (
            count_trainable_parameters(
                model
            )
        ),
        "precision_policy": runtime[
            "policy"
        ],
        "gpu_count": runtime[
            "gpu_count"
        ],
        "stage_1_checkpoint": str(
            STAGE_1_MODEL_FILE
        ),
        "best_model_checkpoint": str(
            BEST_MODEL_FILE
        ),
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("\n" + "=" * 70)
    print(
        "ADVANCED MODEL V3 TRAINING COMPLETE"
    )
    print("=" * 70)

    print(
        f"Stage 1 best epoch:    "
        f"{stage_1_best_epoch}"
    )
    print(
        f"Stage 1 best ROC-AUC:  "
        f"{stage_1_best_auc:.4f}"
    )
    print(
        f"Stage 2 best epoch:    "
        f"{stage_2_best_epoch}"
    )
    print(
        f"Stage 2 best ROC-AUC:  "
        f"{stage_2_best_auc:.4f}"
    )
    print(
        f"Total training time:   "
        f"{total_seconds / 60:.2f} minutes"
    )
    print(
        f"Best checkpoint:       "
        f"{BEST_MODEL_FILE}"
    )
    print(
        f"Metadata:              "
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()