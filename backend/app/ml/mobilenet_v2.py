import tensorflow as tf

from backend.app.core.config import IMAGE_SIZE
from backend.app.ml.metrics import build_binary_metrics
from backend.app.services.augmentation import build_augmentation


def build_mobilenet_v2(
    trainable_backbone: bool = False,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="xray_image",
    )

    x = build_augmentation()(inputs)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        x
    )

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = trainable_backbone

    x = backbone(
        x,
        training=False,
    )

    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = tf.keras.layers.Dropout(
        0.30,
        name="classifier_dropout",
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
        name="mediscan_mobilenet_v2",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=build_binary_metrics(),
    )

    return model