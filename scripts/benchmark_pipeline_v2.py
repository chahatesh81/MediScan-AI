import time

from backend.app.services.data_pipeline_v2 import (
    build_dataset_v2,
)


WARMUP_BATCHES = 10
BENCHMARK_BATCHES = 50
BATCH_SIZE = 16


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — "
        "BASELINE V2 PIPELINE BENCHMARK"
    )
    print("=" * 70)

    dataset = build_dataset_v2(
        split_name="train",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("\nWarming up pipeline...")

    iterator = iter(
        dataset.repeat()
    )

    for _ in range(WARMUP_BATCHES):
        next(iterator)

    print("Benchmarking...")

    start_time = time.perf_counter()

    images_processed = 0

    for _ in range(BENCHMARK_BATCHES):
        images, _ = next(iterator)

        images_processed += int(
            images.shape[0]
        )

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    images_per_second = (
        images_processed
        / elapsed_time
    )

    milliseconds_per_batch = (
        elapsed_time
        / BENCHMARK_BATCHES
        * 1000.0
    )

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Batches measured:    "
        f"{BENCHMARK_BATCHES}"
    )

    print(
        f"Images processed:    "
        f"{images_processed}"
    )

    print(
        f"Elapsed time:        "
        f"{elapsed_time:.2f} seconds"
    )

    print(
        f"Images/second:       "
        f"{images_per_second:.2f}"
    )

    print(
        f"Milliseconds/batch:  "
        f"{milliseconds_per_batch:.2f}"
    )

    print()

    if images_per_second >= 500:
        status = "FAST — KEEP LIVE PREPROCESSING"

    elif images_per_second >= 150:
        status = "ACCEPTABLE — TRAINING SHOULD BE FINE"

    else:
        status = "SLOW — BUILD PREPROCESSED CACHE"

    print(
        f"Pipeline decision:   {status}"
    )

    print(
        "\nBASELINE V2 PIPELINE "
        "BENCHMARK STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()