from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterator

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from backend.app.ml.brain_mri.dataset_loader import (
    ArchiveBackedImageLoader,
    ArchiveBackedRecord,
    DatasetSplit,
    class_index,
    join_canonical_records_with_splits,
)
from backend.app.ml.brain_mri.model_training import (
    BrainMRIArtifactPaths,
    BrainMRIModelConfig,
    BrainMRITrainingConfig,
    build_compiled_brain_mri_model,
    build_training_callbacks,
    calculate_class_weights,
    class_labels_by_index,
    ensure_artifact_directory,
    preprocessing_config_for_model,
    training_contract_metadata,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_AUDIT_PATH = (
    REPOSITORY_ROOT
    / ".local"
    / "audits"
    / "update_7h"
    / "canonical_audit.json"
)

SPLIT_AUDIT_PATH = (
    REPOSITORY_ROOT
    / ".local"
    / "audits"
    / "update_7i"
    / "split_audit.json"
)

ARCHIVE_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "external"
    / "brain_mri"
    / "archives"
)

MODEL_CONFIG = BrainMRIModelConfig()

TRAINING_CONFIG = BrainMRITrainingConfig(
    batch_size=32,
    epochs=30,
    early_stopping_patience=6,
    reduce_lr_patience=3,
    reduce_lr_factor=0.5,
    minimum_learning_rate=1e-7,
    seed=42,
)

ARTIFACT_PATHS = BrainMRIArtifactPaths(
    model_directory=REPOSITORY_ROOT / "models" / "brain_mri",
    best_model_path=(
        REPOSITORY_ROOT
        / "models"
        / "brain_mri"
        / "brain_mri_best.keras"
    ),
    final_model_path=(
        REPOSITORY_ROOT
        / "models"
        / "brain_mri"
        / "brain_mri_final.keras"
    ),
    metadata_path=(
        REPOSITORY_ROOT
        / "models"
        / "brain_mri"
        / "brain_mri_training_metadata.json"
    ),
)


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required audit file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected JSON object in {path}, "
            f"got {type(payload).__name__}."
        )

    return payload


def load_training_records() -> tuple[ArchiveBackedRecord, ...]:
    canonical_audit = load_json(CANONICAL_AUDIT_PATH)
    split_audit = load_json(SPLIT_AUDIT_PATH)

    canonical_records = canonical_audit.get("canonical_records")
    assignments = split_audit.get("assignments")

    if not isinstance(canonical_records, list):
        raise TypeError(
            "canonical_records must be a list."
        )

    if not isinstance(assignments, list):
        raise TypeError(
            "assignments must be a list."
        )

    records = join_canonical_records_with_splits(
        canonical_records=canonical_records,
        assignments=assignments,
    )

    if len(records) != 17_578:
        raise RuntimeError(
            "Unexpected joined record count: "
            f"{len(records):,}; expected 17,578."
        )

    return records


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError(
            "No TensorFlow GPU is visible. "
            "Training has been stopped."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu,
                True,
            )
        except RuntimeError:
            pass

    print(f"Visible GPUs: {len(gpus)}")

    for gpu in gpus:
        print(f"  {gpu}")


def split_records(
    records: tuple[ArchiveBackedRecord, ...],
    split: DatasetSplit,
) -> tuple[ArchiveBackedRecord, ...]:
    return tuple(
        record
        for record in records
        if record.split is split
    )


def image_generator(
    loader: ArchiveBackedImageLoader,
    split: DatasetSplit,
) -> Iterator[tuple[np.ndarray, np.int32]]:
    for loaded in loader.iter_split(split):
        yield (
            loaded.image.astype(
                np.float32,
                copy=False,
            ),
            np.int32(
                class_index(
                    loaded.record.normalized_class
                )
            ),
        )


def build_tf_dataset(
    *,
    loader: ArchiveBackedImageLoader,
    split: DatasetSplit,
    record_count: int,
    training: bool,
) -> tf.data.Dataset:
    output_signature = (
        tf.TensorSpec(
            shape=(
                MODEL_CONFIG.image_height,
                MODEL_CONFIG.image_width,
                MODEL_CONFIG.channels,
            ),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(),
            dtype=tf.int32,
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(loader, split),
        output_signature=output_signature,
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(record_count, 2048),
            seed=TRAINING_CONFIG.seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        TRAINING_CONFIG.batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def history_to_json(
    history: tf.keras.callbacks.History,
) -> dict[str, list[float]]:
    return {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }


def main() -> None:
    print("=" * 70)
    print("MEDISCAN AI — BRAIN MRI FOUR-CLASS TRAINING")
    print("=" * 70)

    configure_runtime(TRAINING_CONFIG.seed)

    print()
    print("=== LOAD CANONICAL SPLIT RECORDS ===")

    records = load_training_records()

    train_records = split_records(
        records,
        DatasetSplit.TRAIN,
    )
    validation_records = split_records(
        records,
        DatasetSplit.VALIDATION,
    )
    test_records = split_records(
        records,
        DatasetSplit.TEST,
    )

    print(f"Total:      {len(records):,}")
    print(f"TRAIN:      {len(train_records):,}")
    print(f"VALIDATION: {len(validation_records):,}")
    print(f"TEST:       {len(test_records):,}")

    if len(train_records) != 12_305:
        raise RuntimeError("Unexpected TRAIN count.")

    if len(validation_records) != 2_637:
        raise RuntimeError(
            "Unexpected VALIDATION count."
        )

    if len(test_records) != 2_636:
        raise RuntimeError("Unexpected TEST count.")

    print("PASS: Dataset split counts verified.")

    print()
    print("=== CLASS DISTRIBUTION ===")

    for split_name, split_group in (
        ("TRAIN", train_records),
        ("VALIDATION", validation_records),
        ("TEST", test_records),
    ):
        counts = Counter(
            record.normalized_class.value
            for record in split_group
        )

        print(split_name)

        for label in sorted(counts):
            print(
                f"  {label:<24} "
                f"{counts[label]:>6,}"
            )

    print()
    print("=== BUILD ARCHIVE-BACKED LOADER ===")

    preprocessing_config = (
        preprocessing_config_for_model(
            MODEL_CONFIG
        )
    )

    loader = ArchiveBackedImageLoader(
        archive_root=ARCHIVE_ROOT,
        records=records,
        config=preprocessing_config,
    )

    print(f"Archive root: {ARCHIVE_ROOT}")
    print(
        "Preprocessing: "
        f"{preprocessing_config.height}x"
        f"{preprocessing_config.width}x"
        f"{preprocessing_config.channels}"
    )

    print()
    print("=== BUILD TENSORFLOW DATASETS ===")

    train_dataset = build_tf_dataset(
        loader=loader,
        split=DatasetSplit.TRAIN,
        record_count=len(train_records),
        training=True,
    )

    validation_dataset = build_tf_dataset(
        loader=loader,
        split=DatasetSplit.VALIDATION,
        record_count=len(validation_records),
        training=False,
    )

    test_dataset = build_tf_dataset(
        loader=loader,
        split=DatasetSplit.TEST,
        record_count=len(test_records),
        training=False,
    )

    print("PASS: TensorFlow datasets created.")

    print()
    print("=== CALCULATE CLASS WEIGHTS ===")

    class_weights = calculate_class_weights(
        train_records
    )

    for index, weight in sorted(
        class_weights.items()
    ):
        print(
            f"  {index}: "
            f"{class_labels_by_index()[index]:<20} "
            f"{weight:.6f}"
        )

    print()
    print("=== BUILD MODEL ===")

    ensure_artifact_directory(ARTIFACT_PATHS)

    model = build_compiled_brain_mri_model(
        MODEL_CONFIG
    )

    print(f"Model:      {model.name}")
    print(f"Input:      {model.input_shape}")
    print(f"Output:     {model.output_shape}")
    print(f"Parameters: {model.count_params():,}")

    print()
    print("=== TRAINING CONFIGURATION ===")
    print(
        f"Batch size:     "
        f"{TRAINING_CONFIG.batch_size}"
    )
    print(
        f"Maximum epochs: "
        f"{TRAINING_CONFIG.epochs}"
    )
    print(
        f"Seed:           "
        f"{TRAINING_CONFIG.seed}"
    )
    print(
        f"Best model:     "
        f"{ARTIFACT_PATHS.best_model_path}"
    )
    print(
        f"Final model:    "
        f"{ARTIFACT_PATHS.final_model_path}"
    )

    callbacks = build_training_callbacks(
        paths=ARTIFACT_PATHS,
        config=TRAINING_CONFIG,
    )

    print()
    print("=" * 70)
    print("STARTING GPU TRAINING")
    print("=" * 70)

    started = time.time()

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=TRAINING_CONFIG.epochs,
        callbacks=list(callbacks),
        class_weight=class_weights,
        verbose=1,
    )

    training_seconds = time.time() - started

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print()
    print("=== FINAL TEST EVALUATION ===")

    test_metrics = model.evaluate(
        test_dataset,
        return_dict=True,
        verbose=1,
    )

    for name, value in test_metrics.items():
        print(f"{name:<24} {float(value):.6f}")

    print()
    print("=== SAVE FINAL MODEL ===")

    ARTIFACT_PATHS.final_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        ARTIFACT_PATHS.final_model_path
    )

    print(
        f"Saved: "
        f"{ARTIFACT_PATHS.final_model_path}"
    )

    metadata = training_contract_metadata(
        model_config=MODEL_CONFIG,
        training_config=TRAINING_CONFIG,
    )

    metadata.update(
        {
            "dataset": {
                "total_records": len(records),
                "train_records": len(
                    train_records
                ),
                "validation_records": len(
                    validation_records
                ),
                "test_records": len(
                    test_records
                ),
            },
            "training": {
                "completed_epochs": len(
                    history.epoch
                ),
                "training_seconds": float(
                    training_seconds
                ),
                "history": history_to_json(
                    history
                ),
            },
            "test_metrics": {
                name: float(value)
                for name, value
                in test_metrics.items()
            },
            "artifacts": {
                "best_model": str(
                    ARTIFACT_PATHS.best_model_path
                ),
                "final_model": str(
                    ARTIFACT_PATHS.final_model_path
                ),
            },
        }
    )

    with ARTIFACT_PATHS.metadata_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print(
        f"Metadata: "
        f"{ARTIFACT_PATHS.metadata_path}"
    )

    print()
    print("=" * 70)
    print("BRAIN MRI TRAINING RUN COMPLETE")
    print("=" * 70)
    print(
        f"Epochs completed: "
        f"{len(history.epoch)}"
    )
    print(
        f"Training time:    "
        f"{training_seconds / 60:.2f} minutes"
    )

    for name, value in test_metrics.items():
        print(
            f"Test {name:<18} "
            f"{float(value):.6f}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
