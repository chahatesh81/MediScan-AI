import tensorflow as tf

from backend.app.services.data_pipeline import (
    build_dataset,
    load_manifest,
)


def main() -> None:
    manifest = load_manifest()

    print("=" * 60)
    print("MEDISCAN AI — DATA PIPELINE TEST")
    print("=" * 60)

    for split in ("train", "val", "test"):
        dataset = build_dataset(
            split=split,
            batch_size=16,
        )

        images, labels = next(iter(dataset))

        expected_count = (
            manifest["final_split"] == split
        ).sum()

        print(f"\nSplit: {split}")
        print(f"Expected images: {expected_count:,}")
        print(f"Batch shape:     {images.shape}")
        print(f"Label shape:     {labels.shape}")
        print(f"Image dtype:     {images.dtype}")
        print(f"Pixel minimum:   {tf.reduce_min(images).numpy():.2f}")
        print(f"Pixel maximum:   {tf.reduce_max(images).numpy():.2f}")

        assert images.shape[1:] == (224, 224, 3)
        assert images.dtype == tf.float32
        assert labels.shape[0] == images.shape[0]

    print("\n" + "=" * 60)
    print("DATA PIPELINE STATUS: READY")
    print("=" * 60)


if __name__ == "__main__":
    main()