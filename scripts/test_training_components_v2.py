import numpy as np
import tensorflow as tf

from backend.app.ml.augmentation_v2 import (
    build_augmentation_v2,
)
from backend.app.services.data_pipeline_v2_cached import (
    build_cached_dataset_v2,
    load_cache_split,
)


def calculate_class_weights_v2() -> dict[int, float]:
    train_split = load_cache_split(
        "train"
    )

    labels = train_split[
        "label"
    ].to_numpy(
        dtype=np.int32
    )

    classes, counts = np.unique(
        labels,
        return_counts=True,
    )

    total_samples = len(labels)
    number_of_classes = len(classes)

    weights = {
        int(class_id): float(
            total_samples
            / (
                number_of_classes
                * class_count
            )
        )
        for class_id, class_count
        in zip(classes, counts)
    }

    return weights


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 TRAINING COMPONENT TEST"
    )
    print("=" * 70)

    dataset = build_cached_dataset_v2(
        split_name="train",
        batch_size=8,
        shuffle=False,
    )

    images, labels = next(
        iter(dataset)
    )

    augmentation = (
        build_augmentation_v2()
    )

    augmented = augmentation(
        images,
        training=True,
    )

    class_weights = (
        calculate_class_weights_v2()
    )

    images_np = images.numpy()
    augmented_np = augmented.numpy()

    mean_absolute_change = float(
        np.mean(
            np.abs(
                augmented_np
                - images_np
            )
        )
    )

    print()
    print(
        f"Input shape:          "
        f"{images_np.shape}"
    )
    print(
        f"Augmented shape:      "
        f"{augmented_np.shape}"
    )
    print(
        f"Input dtype:          "
        f"{images_np.dtype}"
    )
    print(
        f"Augmented dtype:      "
        f"{augmented_np.dtype}"
    )
    print(
        f"Input range:          "
        f"{images_np.min():.2f} "
        f"to {images_np.max():.2f}"
    )
    print(
        f"Augmented range:      "
        f"{augmented_np.min():.2f} "
        f"to {augmented_np.max():.2f}"
    )
    print(
        f"Finite augmentation:  "
        f"{np.isfinite(augmented_np).all()}"
    )
    print(
        f"Mean absolute change: "
        f"{mean_absolute_change:.4f}"
    )
    print(
        f"Class weights:        "
        f"{class_weights}"
    )

    if augmented_np.shape != images_np.shape:
        raise RuntimeError(
            "Augmentation changed tensor shape."
        )

    if not np.isfinite(
        augmented_np
    ).all():
        raise RuntimeError(
            "Augmentation produced "
            "non-finite values."
        )

    if mean_absolute_change <= 0.0:
        raise RuntimeError(
            "Augmentation did not modify images."
        )

    expected_weights = {
        0: 1.7664618086040387,
        1: 0.6974003466204506,
    }

    for class_id in expected_weights:
        if not np.isclose(
            class_weights[class_id],
            expected_weights[class_id],
            rtol=1e-5,
        ):
            raise RuntimeError(
                "Class weights differ from v1. "
                "The training population may "
                "have changed."
            )

    print()
    print("=" * 70)
    print(
        "BASELINE V2 TRAINING COMPONENT "
        "STATUS: READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()