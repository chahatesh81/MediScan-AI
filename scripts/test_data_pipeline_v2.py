import numpy as np

from backend.app.services.data_pipeline_v2 import (
    build_dataset_v2,
    get_dataset_v2_size,
)


EXPECTED_COUNTS = {
    "train": 4024,
    "val": 713,
    "test": 618,
}


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 DATA PIPELINE TEST"
    )
    print("=" * 70)

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        expected_count = (
            EXPECTED_COUNTS[
                split_name
            ]
        )

        actual_count = (
            get_dataset_v2_size(
                split_name
            )
        )

        if actual_count != expected_count:
            raise RuntimeError(
                f"{split_name}: "
                f"expected "
                f"{expected_count}, "
                f"found "
                f"{actual_count}."
            )

        dataset = build_dataset_v2(
            split_name=split_name,
            batch_size=8,
            shuffle=False,
        )

        images, labels = next(
            iter(dataset)
        )

        images_np = images.numpy()
        labels_np = labels.numpy()

        print()
        print(
            f"Split:            "
            f"{split_name}"
        )
        print(
            f"Expected images:  "
            f"{expected_count:,}"
        )
        print(
            f"Batch shape:      "
            f"{images_np.shape}"
        )
        print(
            f"Label shape:      "
            f"{labels_np.shape}"
        )
        print(
            f"Image dtype:      "
            f"{images_np.dtype}"
        )
        print(
            f"Pixel minimum:    "
            f"{images_np.min():.2f}"
        )
        print(
            f"Pixel maximum:    "
            f"{images_np.max():.2f}"
        )
        print(
            f"Finite pixels:    "
            f"{np.isfinite(images_np).all()}"
        )
        print(
            f"Labels:           "
            f"{labels_np.tolist()}"
        )

        if images_np.shape != (
            8,
            224,
            224,
            3,
        ):
            raise RuntimeError(
                "Unexpected image shape."
            )

        if labels_np.shape != (8,):
            raise RuntimeError(
                "Unexpected label shape."
            )

        if images_np.dtype != np.float32:
            raise RuntimeError(
                "Images must be float32."
            )

        if not np.isfinite(
            images_np
        ).all():
            raise RuntimeError(
                "Non-finite image values found."
            )

        if (
            images_np.min() < 0.0
            or images_np.max() > 255.0
        ):
            raise RuntimeError(
                "Pixel range is invalid."
            )

    print()
    print("=" * 70)
    print(
        "BASELINE V2 DATA PIPELINE "
        "STATUS: READY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()