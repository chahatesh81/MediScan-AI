import json
import time
from datetime import datetime

import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT, RANDOM_SEED
from backend.app.core.reproducibility import set_global_seed
from backend.app.ml.callbacks import build_training_callbacks
from backend.app.ml.mobilenet_v2 import build_mobilenet_v2
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.class_weights import calculate_class_weights
from backend.app.services.data_pipeline import build_dataset


EXPERIMENT_NAME = "mobilenet_v2_stage_a"
BATCH_SIZE = 16
MAX_EPOCHS = 15

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
METADATA_FILE = (
    RESULTS_DIR / "mobilenet_stage_a_training_metadata.json"
)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — MOBILENETV2 STAGE A TRAINING")
    print("=" * 70)

    set_global_seed(RANDOM_SEED)
    runtime = configure_training_runtime()

    train_dataset = build_dataset(
        split="train",
        batch_size=BATCH_SIZE,
    )

    val_dataset = build_dataset(
        split="val",
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = build_mobilenet_v2(
        trainable_backbone=False,
        learning_rate=1e-3,
    )

    class_weights = calculate_class_weights()

    callbacks, experiment_dir = build_training_callbacks(
        EXPERIMENT_NAME
    )

    backbone = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    print(f"GPU count:          {runtime['gpu_count']}")
    print(f"Precision policy:   {runtime['policy']}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Maximum epochs:     {MAX_EPOCHS}")
    print(f"Total parameters:   {model.count_params():,}")
    print(f"Backbone trainable: {backbone.trainable}")
    print(f"Class weights:      {class_weights}")
    print(f"Experiment log:     {experiment_dir}")

    print("\nStarting Stage A training...\n")

    start_time = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_seconds = time.perf_counter() - start_time

    best_epoch = (
        int(tf.argmax(
            history.history["val_roc_auc"]
        ).numpy())
        + 1
    )

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "stage": "frozen_backbone",
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "epochs_completed": len(
            history.history["loss"]
        ),
        "best_epoch_by_val_roc_auc": best_epoch,
        "best_val_roc_auc": float(
            max(history.history["val_roc_auc"])
        ),
        "best_val_pr_auc": float(
            max(history.history["val_pr_auc"])
        ),
        "training_seconds": float(elapsed_seconds),
        "parameters": int(model.count_params()),
        "precision_policy": runtime["policy"],
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2)
    )

    print("\n" + "=" * 70)
    print("MOBILENETV2 STAGE A TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Epochs completed:    "
        f"{metadata['epochs_completed']}"
    )
    print(f"Best epoch:          {best_epoch}")
    print(
        f"Best val ROC-AUC:    "
        f"{metadata['best_val_roc_auc']:.4f}"
    )
    print(
        f"Best val PR-AUC:     "
        f"{metadata['best_val_pr_auc']:.4f}"
    )
    print(
        f"Training time:       "
        f"{elapsed_seconds / 60:.2f} minutes"
    )
    print(f"Metadata:            {METADATA_FILE}")
    print(
        "Best checkpoint:     "
        f"{PROJECT_ROOT / 'models' / (EXPERIMENT_NAME + '_best.keras')}"
    )


if __name__ == "__main__":
    main()