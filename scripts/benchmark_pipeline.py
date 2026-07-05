import time

import tensorflow as tf

from backend.app.services.data_pipeline import build_dataset


WARMUP_BATCHES = 5
BENCHMARK_BATCHES = 50
BATCH_SIZE = 16


def main() -> None:
    print("=" * 60)
    print("MEDISCAN AI — DATA PIPELINE BENCHMARK")
    print("=" * 60)

    dataset = build_dataset(
        split="train",
        batch_size=BATCH_SIZE,
    )

    iterator = iter(dataset.repeat())

    print("\nWarming up pipeline...")

    for _ in range(WARMUP_BATCHES):
        next(iterator)

    print("Benchmarking...")

    start_time = time.perf_counter()

    total_images = 0

    for _ in range(BENCHMARK_BATCHES):
        images, _ = next(iterator)

        # Force TensorFlow execution to finish.
        _ = tf.reduce_sum(images).numpy()

        total_images += int(images.shape[0])

    elapsed_time = time.perf_counter() - start_time

    images_per_second = total_images / elapsed_time
    milliseconds_per_batch = (
        elapsed_time / BENCHMARK_BATCHES
    ) * 1000

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Batches measured:    {BENCHMARK_BATCHES}")
    print(f"Images processed:    {total_images}")
    print(f"Elapsed time:        {elapsed_time:.2f} seconds")
    print(f"Images/second:       {images_per_second:.2f}")
    print(f"Milliseconds/batch:  {milliseconds_per_batch:.2f}")

    print("\nPIPELINE BENCHMARK STATUS: COMPLETE")r


if __name__ == "__main__":
    main()r