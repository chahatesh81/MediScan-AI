import tensorflow as tf

from backend.app.core.config import IMAGE_SIZE
from backend.app.ml.metrics import build_binary_metrics
from backend.app.services.augmentation import build_augmentation


def build_baseline_cnn() -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="xray_image",
    )

    x = build_augmentation()(inputs)

    x = tf.keras.layers.Rescaling(
        scale=1.0 / 255.0,
        name="normalize",
    )(x)

    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
        )(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        dtype="float32",
        name="pneumonia_probability",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="mediscan_baseline_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=build_binary_metrics(),
    )

    return model