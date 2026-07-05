import json
import time
from datetime import datetime

import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT, RANDOM_SEED
from backend.app.core.reproducibility import set_global_seed
from backend.app.ml.callbacks import build_training_callbacks
from backend.app.ml.metrics import build_binary_metrics
from backend.app.ml.runtime import configure_training_runtime
from backend.app.services.class_weights import calculate_class_weights
from backend.app.services.data_pipeline import build_dataset


EXPERIMENT_NAME = "mobilenet_v2_stage_b"
BATCH_SIZE = 16
MAX_EPOCHS = 15
UNFREEZE_LAST_LAYERS = 30
LEARNING_RATE = 1e-5

STAGE_A_MODEL = (
    PROJECT_ROOT
    / "models"
    / "mobilenet_v2_stage_a_best.keras"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
METADATA_FILE = (
    RESULTS_DIR
    / "mobilenet_stage_b_training_metadata.json"
)


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — MOBILENETV2 STAGE B FINE-TUNING")
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

    print(f"Loading Stage A checkpoint: {STAGE_A_MODEL}")

    model = tf.keras.models.load_model(
        STAGE_A_MODEL
    )

    backbone = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    for layer in backbone.layers[
        -UNFREEZE_LAST_LAYERS:
    ]:
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=build_binary_metrics(),
    )

    trainable_parameters = sum(
        int(tf.size(weight).numpy())
        for weight in model.trainable_weights
    )

    class_weights = calculate_class_weights()

    callbacks, experiment_dir = build_training_callbacks(
        EXPERIMENT_NAME
    )

    print(f"GPU count:             {runtime['gpu_count']}")
    print(f"Precision policy:      {runtime['policy']}")
    print(f"Batch size:            {BATCH_SIZE}")
    print(f"Maximum epochs:        {MAX_EPOCHS}")
    print(f"Learning rate:         {LEARNING_RATE}")
    print(f"Last layers considered:{UNFREEZE_LAST_LAYERS}")
    print(f"Trainable parameters:  {trainable_parameters:,}")
    print(f"Experiment log:        {experiment_dir}")

    print("\nStarting Stage B fine-tuning...\n")

    start_time = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    best_epoch = (
        int(
            tf.argmax(
                history.history["val_roc_auc"]
            ).numpy()
        )
        + 1
    )

    metadata = {
        "experiment": EXPERIMENT_NAME,
        "timestamp": datetime.now().isoformat(),
        "stage": "fine_tuning",
        "source_checkpoint": str(STAGE_A_MODEL),
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
        "learning_rate": LEARNING_RATE,
        "unfreeze_last_layers": UNFREEZE_LAST_LAYERS,
        "trainable_parameters": trainable_parameters,
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
    print("MOBILENETV2 STAGE B FINE-TUNING COMPLETE")
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