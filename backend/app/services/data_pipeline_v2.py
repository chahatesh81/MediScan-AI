from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from backend.app.core.config import PROJECT_ROOT
from backend.app.ml.xray_preprocessing import preprocess_xray


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_manifest.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chest_xray"
)

IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def load_manifest_split(
    split_name: str,
) -> pd.DataFrame:
    """
    Load one final patient-safe dataset split.
    """

    valid_splits = {
        "train",
        "val",
        "test",
    }

    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split: {split_name}. "
            f"Expected one of: "
            f"{sorted(valid_splits)}"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH
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
            f"No images found for "
            f"split '{split_name}'."
        )

    split["label"] = (
        split["class_name"]
        .map(
            {
                "NORMAL": 0,
                "PNEUMONIA": 1,
            }
        )
    )

    if split["label"].isna().any():
        unknown_classes = (
            split.loc[
                split["label"].isna(),
                "class_name",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown class names: "
            f"{unknown_classes}"
        )

    split["label"] = split[
        "label"
    ].astype(np.float32)

    split["absolute_path"] = (
        split["path"]
        .apply(
            lambda relative_path: str(
                IMAGE_ROOT
                / relative_path
            )
        )
    )

    missing_paths = [
        path
        for path in split["absolute_path"]
        if not Path(path).is_file()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Dataset image not found: "
            f"{missing_paths[0]}"
        )

    return split


def _opencv_preprocess(
    image_path: bytes | np.ndarray,
) -> np.ndarray:
    """
    Python/OpenCV preprocessing bridge.

    Called through tf.numpy_function.
    """

    if isinstance(
        image_path,
        np.ndarray,
    ):
        image_path = (
            image_path.item()
        )

    if isinstance(
        image_path,
        bytes,
    ):
        image_path = (
            image_path.decode("utf-8")
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: "
            f"{image_path}"
        )

    processed = preprocess_xray(
        image,
        target_size=IMAGE_SIZE,
        return_metadata=False,
    )

    return processed.astype(
        np.float32
    )


def _load_and_preprocess(
    image_path: tf.Tensor,
    label: tf.Tensor,
) -> tuple[
    tf.Tensor,
    tf.Tensor,
]:
    """
    TensorFlow wrapper around deterministic
    artifact-aware preprocessing.
    """

    image = tf.numpy_function(
        func=_opencv_preprocess,
        inp=[image_path],
        Tout=tf.float32,
    )

    image.set_shape(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        )
    )

    label = tf.cast(
        label,
        tf.float32,
    )

    label.set_shape(
        ()
    )

    return image, label


def build_dataset_v2(
    split_name: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool | None = None,
) -> tf.data.Dataset:
    """
    Build the deterministic baseline_v2 dataset.

    Important:
    - train is shuffled;
    - val and test preserve manifest row order;
    - augmentation is NOT performed here;
    - preprocessing is identical across splits.
    """

    split = load_manifest_split(
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
        dtype=np.float32
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
        _load_and_preprocess,
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


def get_dataset_v2_size(
    split_name: str,
) -> int:
    """
    Return exact image count for a split.
    """

    return len(
        load_manifest_split(
            split_name
        )
    )