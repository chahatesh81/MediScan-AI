from __future__ import annotations

import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app


TEST_IMAGE = Path(
    "data/raw/chest_xray/test/"
    "PNEUMONIA/person155_bacteria_730.jpeg"
)

WARMUP_RUNS = 2
TIMED_RUNS = 10

ENDPOINTS = [
    "/api/v1/predict",
    "/api/v1/explain",
    "/api/v1/explain/overlay",
    "/api/v1/analyze",
]


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(message)


def percentile(
    values: list[float],
    fraction: float,
) -> float:
    ordered = sorted(values)

    index = int(
        round(
            (len(ordered) - 1)
            * fraction
        )
    )

    return ordered[index]


def run_request(
    client: TestClient,
    endpoint: str,
    image_bytes: bytes,
) -> float:
    start = time.perf_counter()

    response = client.post(
        endpoint,
        files={
            "file": (
                TEST_IMAGE.name,
                image_bytes,
                "image/jpeg",
            )
        },
    )

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    require(
        response.status_code == 200,
        (
            f"{endpoint} failed with "
            f"status {response.status_code}: "
            f"{response.text}"
        ),
    )

    return elapsed_ms


def main() -> None:
    print("=" * 70)
    print(
        "MEDISCAN AI — API LATENCY BENCHMARK"
    )
    print("=" * 70)

    require(
        TEST_IMAGE.is_file(),
        f"Missing test image: {TEST_IMAGE}",
    )

    image_bytes = TEST_IMAGE.read_bytes()

    results: dict[str, list[float]] = {}

    with TestClient(app) as client:
        for endpoint in ENDPOINTS:
            print()
            print(f"Endpoint: {endpoint}")

            print(
                f"Warm-up runs: {WARMUP_RUNS}"
            )

            for _ in range(WARMUP_RUNS):
                run_request(
                    client,
                    endpoint,
                    image_bytes,
                )

            print(
                f"Timed runs:   {TIMED_RUNS}"
            )

            timings = []

            for run_number in range(
                1,
                TIMED_RUNS + 1,
            ):
                elapsed_ms = run_request(
                    client,
                    endpoint,
                    image_bytes,
                )

                timings.append(elapsed_ms)

                print(
                    f"  Run {run_number:02d}: "
                    f"{elapsed_ms:.2f} ms"
                )

            results[endpoint] = timings

    print()
    print("=" * 70)
    print("LATENCY SUMMARY")
    print("=" * 70)

    for endpoint, timings in results.items():
        mean_ms = statistics.mean(timings)
        median_ms = statistics.median(timings)
        minimum_ms = min(timings)
        maximum_ms = max(timings)
        p90_ms = percentile(
            timings,
            0.90,
        )

        print()
        print(endpoint)
        print(
            f"  Mean:   {mean_ms:.2f} ms"
        )
        print(
            f"  Median: {median_ms:.2f} ms"
        )
        print(
            f"  P90:    {p90_ms:.2f} ms"
        )
        print(
            f"  Min:    {minimum_ms:.2f} ms"
        )
        print(
            f"  Max:    {maximum_ms:.2f} ms"
        )

    predict_mean = statistics.mean(
        results["/api/v1/predict"]
    )

    analyze_mean = statistics.mean(
        results["/api/v1/analyze"]
    )

    overhead_ms = (
        analyze_mean - predict_mean
    )

    overhead_ratio = (
        analyze_mean / predict_mean
    )

    print()
    print("=" * 70)
    print("COMBINED ANALYSIS OVERHEAD")
    print("=" * 70)

    print(
        f"Predict mean:  "
        f"{predict_mean:.2f} ms"
    )

    print(
        f"Analyze mean:  "
        f"{analyze_mean:.2f} ms"
    )

    print(
        f"Added latency: "
        f"{overhead_ms:.2f} ms"
    )

    print(
        f"Latency ratio: "
        f"{overhead_ratio:.2f}x"
    )

    print()
    print("=" * 70)
    print(
        "API LATENCY BENCHMARK STATUS: PASS"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
