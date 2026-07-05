from __future__ import annotations

import tensorflow as tf

from backend.app.core.config import IMAGE_SIZE
from backend.app.ml.metrics import build_binary_metrics
from backend.app.services.augmentation import build_augmentation


MODEL_NAME = "mediscan_advanced_v3"
BACKBONE_NAME = "efficientnetv2_b0"

DENSE_UNITS = 256
DROPOUT_RATE = 0.40
LABEL_SMOOTHING = 0.05

INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


def build_advanced_model_v3(
    backbone_trainable: bool = False,
) -> tf.keras.Model:
    """
    Build MediScan AI Advanced Model V3.

    Architecture:
        Input
        -> Conservative augmentation
        -> EfficientNetV2-B0 ImageNet backbone
        -> Global Average Pooling
        -> Batch Normalization
        -> Dense
        -> Dropout
        -> Sigmoid probability

    The model is initially built with a frozen backbone.
    Partial fine-tuning is enabled separately.
    """

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="xray_image",
    )

    x = build_augmentation()(inputs)

    backbone = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        include_preprocessing=True,
    )

    backbone.trainable = backbone_trainable

    x = backbone(
        x,
        training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = tf.keras.layers.BatchNormalization(
        name="head_batch_norm",
    )(x)

    x = tf.keras.layers.Dense(
        DENSE_UNITS,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(
            1e-4
        ),
        name="classification_dense",
    )(x)

    x = tf.keras.layers.Dropout(
        DROPOUT_RATE,
        name="classification_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="pneumonia_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=MODEL_NAME,
    )

    compile_advanced_model_v3(
        model=model,
        learning_rate=INITIAL_LEARNING_RATE,
    )

    return model


def compile_advanced_model_v3(
    model: tf.keras.Model,
    learning_rate: float,
) -> None:
    """
    Compile V3 with AdamW and binary metrics.
    """

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        global_clipnorm=1.0,
    )

    loss = tf.keras.losses.BinaryCrossentropy(
        label_smoothing=LABEL_SMOOTHING,
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=build_binary_metrics(),
    )


def get_backbone(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Return the EfficientNetV2-B0 backbone.
    """

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "efficientnetv2" in layer.name.lower():
                return layer

    raise RuntimeError(
        "EfficientNetV2 backbone was not found."
    )


def enable_partial_fine_tuning(
    model: tf.keras.Model,
    unfreeze_last_layers: int = 40,
) -> dict[str, int]:
    """
    Unfreeze only the final backbone layers.

    BatchNormalization layers remain frozen because
    medical-image datasets are relatively small and
    unstable BN statistics can damage pretrained
    representations.
    """

    backbone = get_backbone(model)

    backbone.trainable = True

    total_layers = len(backbone.layers)

    freeze_until = max(
        0,
        total_layers - unfreeze_last_layers,
    )

    trainable_layers = 0
    frozen_layers = 0

    for index, layer in enumerate(backbone.layers):
        if index < freeze_until:
            layer.trainable = False
            frozen_layers += 1
            continue

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False
            frozen_layers += 1
        else:
            layer.trainable = True
            trainable_layers += 1

    compile_advanced_model_v3(
        model=model,
        learning_rate=FINE_TUNE_LEARNING_RATE,
    )

    return {
        "total_backbone_layers": total_layers,
        "requested_unfreeze_layers": (
            unfreeze_last_layers
        ),
        "trainable_backbone_layers": (
            trainable_layers
        ),
        "frozen_backbone_layers": frozen_layers,
    }


def count_trainable_parameters(
    model: tf.keras.Model,
) -> int:
    return int(
        sum(
            tf.keras.backend.count_params(weight)
            for weight in model.trainable_weights
        )
    )


def count_non_trainable_parameters(
    model: tf.keras.Model,
) -> int:
    return int(
        sum(
            tf.keras.backend.count_params(weight)
            for weight in model.non_trainable_weights
        )
    )