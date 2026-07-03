from pathlib import Path

import pandas as pd
import tensorflow as tf

from backend.app.core.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
)


DATASET_DIR = RAW_DATA_DIR / "chest_xray"
MANIFEST_FILE = PROCESSED_DATA_DIR / "final_manifest.csv"

AUTOTUNE = tf.data.AUTOTUNE

CLASS_TO_LABEL = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_FILE)


def decode_image(
    path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:

    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    return image, label


def build_dataset(
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
) -> tf.data.Dataset:

    df = load_manifest()

    split_df = df[
        df["final_split"] == split
    ].copy()

    paths = [
        str(DATASET_DIR / path)
        for path in split_df["path"]
    ]

    labels = [
        CLASS_TO_LABEL[class_name]
        for class_name in split_df["class_name"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        decode_image,
        num_parallel_calls=AUTOTUNE,
    )

    if shuffle is None:
        shuffle = split == "train"

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(split_df),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset