from __future__ import annotations

import pandas as pd
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT


CACHE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "baseline_v2_cache_manifest.csv"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def load_cache_split(
    split_name: str,
) -> pd.DataFrame:
    valid_splits = {
        "train",
        "val",
        "test",
    }

    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split: {split_name}"
        )

    manifest = pd.read_csv(
        CACHE_MANIFEST_PATH
    )

    split = (
        manifest[
            manifest["final_split"]
            == split_name
        ]
        .reset_index(drop=True)
        .copy()
    )

    if split.empty:
        raise RuntimeError(
            f"No cached images for "
            f"split '{split_name}'."
        )

    split["absolute_path"] = (
        split["cache_path"]
        .apply(
            lambda path: str(
                PROJECT_ROOT / path
            )
        )
    )

    return split


def _load_cached_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[
    tf.Tensor,
    tf.Tensor,
]:
    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.ensure_shape(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


def build_cached_dataset_v2(
    split_name: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
) -> tf.data.Dataset:
    split = load_cache_split(
        split_name
    )

    paths = split[
        "absolute_path"
    ].to_numpy(
        dtype=str
    )

    labels = split[
        "label"
    ].to_numpy(
        dtype="float32"
    )

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                paths,
                labels,
            )
        )
    )

    if shuffle is None:
        shuffle = (
            split_name == "train"
        )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(split),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _load_cached_image,
        num_parallel_calls=AUTOTUNE,
        deterministic=(
            split_name != "train"
        ),
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


def get_cached_dataset_v2_size(
    split_name: str,
) -> int:
    return len(
        load_cache_split(
            split_name
        )
    )