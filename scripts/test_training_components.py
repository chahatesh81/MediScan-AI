import tensorflow as tf

from backend.app.services.augmentation import build_augmentation
from backend.app.services.class_weights import calculate_class_weights
from backend.app.services.data_pipeline import build_dataset


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — TRAINING COMPONENT TEST")
    print("=" * 60)

    train_dataset = build_dataset(
        split="train",
        batch_size=8,
    )

    images, labels = next(iter(train_dataset))

    augmentation = build_augmentation()

    augmented_images = augmentation(
        images,
        training=True,
    )

    class_weights = calculate_class_weights()

    print(f"Input shape:       {images.shape}")
    print(f"Augmented shape:   {augmented_images.shape}")
    print(f"Input dtype:       {images.dtype}")
    print(f"Augmented dtype:   {augmented_images.dtype}")
    print(f"Class weights:     {class_weights}")

    assert augmented_images.shape == images.shape
    assert tf.reduce_all(
        tf.math.is_finite(augmented_images)
    )
    assert set(class_weights.keys()) == {0, 1}
    assert class_weights[0] > class_weights[1]

    print("\nTRAINING COMPONENT STATUS: READY")


if __name__ == "__main__":
    main()