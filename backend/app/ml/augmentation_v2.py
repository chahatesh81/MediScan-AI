from __future__ import annotations

import tensorflow as tf


RANDOM_SEED = 42


def build_augmentation_v2() -> tf.keras.Sequential:
    """
    Conservative chest X-ray augmentation.

    Design goals:
    - reduce exact framing dependence;
    - preserve diagnostic anatomy;
    - avoid unrealistic medical-image transformations.

    Intentionally excluded:
    - vertical flips;
    - large rotations;
    - aggressive crops;
    - strong contrast distortion;
    - elastic deformation.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomTranslation(
                height_factor=0.03,
                width_factor=0.03,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED,
                name="v2_random_translation",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.03, 0.03),
                width_factor=(-0.03, 0.03),
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED + 1,
                name="v2_random_zoom",
            ),
            tf.keras.layers.RandomRotation(
                factor=0.015,
                fill_mode="constant",
                fill_value=0.0,
                seed=RANDOM_SEED + 2,
                name="v2_random_rotation",
            ),
            tf.keras.layers.RandomContrast(
                factor=0.05,
                seed=RANDOM_SEED + 3,
                name="v2_random_contrast",
            ),
        ],
        name="mediscan_augmentation_v2",
    )