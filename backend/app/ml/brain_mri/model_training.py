from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import tensorflow as tf

from backend.app.ml.brain_mri.contract import BrainMRIClass
from backend.app.ml.brain_mri.dataset_loader import (
    ArchiveBackedRecord,
    ImagePreprocessingConfig,
    class_index_mapping,
)
from backend.app.ml.brain_mri.dataset_split import DatasetSplit


MODEL_NAME = "brain_mri_classifier"

DEFAULT_MODEL_DIRECTORY = Path("models") / "brain_mri"
DEFAULT_BEST_MODEL_PATH = (
    DEFAULT_MODEL_DIRECTORY / "brain_mri_best.keras"
)
DEFAULT_FINAL_MODEL_PATH = (
    DEFAULT_MODEL_DIRECTORY / "brain_mri_final.keras"
)
DEFAULT_METADATA_PATH = (
    DEFAULT_MODEL_DIRECTORY / "brain_mri_training_metadata.json"
)


@dataclass(frozen=True, slots=True)
class BrainMRIModelConfig:
    image_height: int = 224
    image_width: int = 224
    channels: int = 3
    num_classes: int = 4
    learning_rate: float = 1e-3
    dropout_rate: float = 0.35
    dense_units: int = 256


@dataclass(frozen=True, slots=True)
class BrainMRITrainingConfig:
    batch_size: int = 32
    epochs: int = 30
    early_stopping_patience: int = 6
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    minimum_learning_rate: float = 1e-7
    seed: int = 42


@dataclass(frozen=True, slots=True)
class BrainMRIArtifactPaths:
    model_directory: Path = DEFAULT_MODEL_DIRECTORY
    best_model_path: Path = DEFAULT_BEST_MODEL_PATH
    final_model_path: Path = DEFAULT_FINAL_MODEL_PATH
    metadata_path: Path = DEFAULT_METADATA_PATH


def validate_model_config(
    config: BrainMRIModelConfig,
) -> None:
    if config.image_height <= 0:
        raise ValueError("image_height must be positive.")

    if config.image_width <= 0:
        raise ValueError("image_width must be positive.")

    if config.channels != 3:
        raise ValueError(
            "Brain MRI classifier requires exactly 3 channels."
        )

    canonical_class_count = len(tuple(BrainMRIClass))

    if config.num_classes != canonical_class_count:
        raise ValueError(
            "num_classes must match the canonical Brain MRI "
            f"class count ({canonical_class_count})."
        )

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if not 0.0 <= config.dropout_rate < 1.0:
        raise ValueError(
            "dropout_rate must be in the interval [0, 1)."
        )

    if config.dense_units <= 0:
        raise ValueError("dense_units must be positive.")


def validate_training_config(
    config: BrainMRITrainingConfig,
) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")

    if config.early_stopping_patience < 0:
        raise ValueError(
            "early_stopping_patience cannot be negative."
        )

    if config.reduce_lr_patience < 0:
        raise ValueError(
            "reduce_lr_patience cannot be negative."
        )

    if not 0.0 < config.reduce_lr_factor < 1.0:
        raise ValueError(
            "reduce_lr_factor must be in the interval (0, 1)."
        )

    if config.minimum_learning_rate <= 0:
        raise ValueError(
            "minimum_learning_rate must be positive."
        )


def preprocessing_config_for_model(
    config: BrainMRIModelConfig,
) -> ImagePreprocessingConfig:
    validate_model_config(config)

    return ImagePreprocessingConfig(
        height=config.image_height,
        width=config.image_width,
        channels=config.channels,
    )


def build_brain_mri_model(
    config: BrainMRIModelConfig = BrainMRIModelConfig(),
) -> tf.keras.Model:
    validate_model_config(config)

    inputs = tf.keras.Input(
        shape=(
            config.image_height,
            config.image_width,
            config.channels,
        ),
        name="image",
    )

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="block1_conv",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(
        name="block1_batch_norm",
    )(x)
    x = tf.keras.layers.Activation(
        "relu",
        name="block1_relu",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="block1_pool",
    )(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="block2_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block2_batch_norm",
    )(x)
    x = tf.keras.layers.Activation(
        "relu",
        name="block2_relu",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="block2_pool",
    )(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="block3_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block3_batch_norm",
    )(x)
    x = tf.keras.layers.Activation(
        "relu",
        name="block3_relu",
    )(x)
    x = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        name="block3_pool",
    )(x)

    x = tf.keras.layers.Conv2D(
        256,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="block4_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        name="block4_batch_norm",
    )(x)
    x = tf.keras.layers.Activation(
        "relu",
        name="block4_relu",
    )(x)

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pool",
    )(x)

    x = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        name="classifier_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        config.dropout_rate,
        name="classifier_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        config.num_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )


def compile_brain_mri_model(
    model: tf.keras.Model,
    config: BrainMRIModelConfig = BrainMRIModelConfig(),
) -> tf.keras.Model:
    validate_model_config(config)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate,
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy",
            ),
        ],
    )

    return model


def build_compiled_brain_mri_model(
    config: BrainMRIModelConfig = BrainMRIModelConfig(),
) -> tf.keras.Model:
    model = build_brain_mri_model(config)

    return compile_brain_mri_model(
        model,
        config,
    )


def calculate_class_weights(
    records: Iterable[ArchiveBackedRecord],
) -> dict[int, float]:
    training_records = tuple(
        record
        for record in records
        if record.split is DatasetSplit.TRAIN
    )

    if not training_records:
        raise ValueError(
            "At least one TRAIN record is required."
        )

    class_mapping = class_index_mapping()

    counts = Counter(
        record.normalized_class
        for record in training_records
    )

    missing_classes = tuple(
        brain_class
        for brain_class in BrainMRIClass
        if counts[brain_class] == 0
    )

    if missing_classes:
        missing = ", ".join(
            brain_class.value
            for brain_class in missing_classes
        )

        raise ValueError(
            "TRAIN records must contain every canonical class. "
            f"Missing: {missing}."
        )

    total = len(training_records)
    class_count = len(class_mapping)

    return {
        class_mapping[brain_class]:
            total / (class_count * counts[brain_class])
        for brain_class in BrainMRIClass
    }


def class_labels_by_index() -> tuple[str, ...]:
    mapping = class_index_mapping()

    return tuple(
        brain_class.value
        for brain_class, _ in sorted(
            mapping.items(),
            key=lambda item: item[1],
        )
    )


def build_training_callbacks(
    *,
    paths: BrainMRIArtifactPaths = BrainMRIArtifactPaths(),
    config: BrainMRITrainingConfig = BrainMRITrainingConfig(),
) -> tuple[tf.keras.callbacks.Callback, ...]:
    validate_training_config(config)

    return (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(paths.best_model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.minimum_learning_rate,
            verbose=1,
        ),
    )


def ensure_artifact_directory(
    paths: BrainMRIArtifactPaths = BrainMRIArtifactPaths(),
) -> Path:
    paths.model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return paths.model_directory


def training_contract_metadata(
    *,
    model_config: BrainMRIModelConfig = BrainMRIModelConfig(),
    training_config: BrainMRITrainingConfig = (
        BrainMRITrainingConfig()
    ),
) -> dict[str, object]:
    validate_model_config(model_config)
    validate_training_config(training_config)

    return {
        "model_name": MODEL_NAME,
        "input_shape": [
            model_config.image_height,
            model_config.image_width,
            model_config.channels,
        ],
        "num_classes": model_config.num_classes,
        "class_labels": list(class_labels_by_index()),
        "learning_rate": model_config.learning_rate,
        "dropout_rate": model_config.dropout_rate,
        "dense_units": model_config.dense_units,
        "batch_size": training_config.batch_size,
        "epochs": training_config.epochs,
        "seed": training_config.seed,
    }
