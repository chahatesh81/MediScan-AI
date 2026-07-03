import tensorflow as tf

from backend.app.core.config import RANDOM_SEED


def build_augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.03,
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.05, 0.05),
                width_factor=(-0.05, 0.05),
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
            tf.keras.layers.RandomContrast(
                factor=0.10,
                seed=RANDOM_SEED,
            ),
        ],
        name="medical_augmentation",
    )