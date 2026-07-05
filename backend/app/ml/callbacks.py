from datetime import datetime
from pathlib import Path

import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def build_training_callbacks(
    experiment_name: str,
) -> tuple[list[tf.keras.callbacks.Callback], Path]:

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiment_dir = (
        LOGS_DIR
        / "training"
        / f"{experiment_name}_{timestamp}"
    )

    checkpoint_path = (
        MODELS_DIR
        / f"{experiment_name}_best.keras"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [
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
            min_lr=1e-6,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            experiment_dir / "training_log.csv",
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=experiment_dir / "tensorboard",
            histogram_freq=0,
        ),

        tf.keras.callbacks.TerminateOnNaN(),
    ]

    return callbacks, experiment_dir