from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BACKEND_ROOT = (
    PROJECT_ROOT
    / "backend"
    / "app"
)

CRITICAL_TESTS = [
    "scripts.test_attention_quality",
    "scripts.test_explanation_service",
    "scripts.test_inference_service_parity",
    "scripts.test_analysis_service",
    "scripts.test_analysis_forward_passes",
    "scripts.test_api",
]


@dataclass
class ValidationResult:
    name: str
    status: str
    duration_seconds: float


def print_separator() -> None:
    print("=" * 70)


def require_file(
    path: Path,
) -> None:
    if not path.is_file():
        raise RuntimeError(
            f"Required file is missing: {path}"
        )


def collect_backend_files() -> list[Path]:
    files = sorted(
        path
        for path in BACKEND_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )

    if not files:
        raise RuntimeError(
            "No backend Python files were found."
        )

    return files


def run_command(
    name: str,
    command: list[str],
) -> ValidationResult:
    print()
    print_separator()
    print(name)
    print_separator()

    started = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
    )

    duration_seconds = (
        time.perf_counter() - started
    )

    if completed.returncode != 0:
        print()
        print(
            f"{name}: FAIL "
            f"({duration_seconds:.2f} s)"
        )

        raise RuntimeError(
            f"Release validation failed: {name}"
        )

    print()
    print(
        f"{name}: PASS "
        f"({duration_seconds:.2f} s)"
    )

    return ValidationResult(
        name=name,
        status="PASS",
        duration_seconds=duration_seconds,
    )


def main() -> None:
    print_separator()
    print(
        "MEDISCAN AI — BACKEND RELEASE VALIDATION"
    )
    print_separator()

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Python:       {sys.executable}")

    backend_files = collect_backend_files()

    print(
        f"Backend files: {len(backend_files)}"
    )

    for module_name in CRITICAL_TESTS:
        module_path = (
            PROJECT_ROOT
            / Path(
                *module_name.split(".")
            )
        ).with_suffix(".py")

        require_file(module_path)

    results: list[ValidationResult] = []

    compile_command = [
        sys.executable,
        "-m",
        "py_compile",
        *[
            str(path)
            for path in backend_files
        ],
    ]

    results.append(
        run_command(
            "BACKEND SYNTAX COMPILATION",
            compile_command,
        )
    )

    for module_name in CRITICAL_TESTS:
        display_name = (
            module_name
            .removeprefix("scripts.")
            .replace("_", " ")
            .upper()
        )

        results.append(
            run_command(
                display_name,
                [
                    sys.executable,
                    "-m",
                    module_name,
                ],
            )
        )

    total_duration = sum(
        result.duration_seconds
        for result in results
    )

    print()
    print_separator()
    print("RELEASE VALIDATION SUMMARY")
    print_separator()

    print()

    for result in results:
        print(
            f"{result.status:<5}  "
            f"{result.duration_seconds:>8.2f} s  "
            f"{result.name}"
        )

    print()
    print(f"Checks passed: {len(results)}")
    print(f"Checks failed: 0")
    print(
        f"Total time:    "
        f"{total_duration:.2f} s"
    )

    print()
    print_separator()
    print(
        "MEDISCAN AI BACKEND RELEASE STATUS: PASS"
    )
    print_separator()


if __name__ == "__main__":
    main()
