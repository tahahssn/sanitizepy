"""
Benchmarks for the Cleaner public API and configuration.
"""

from __future__ import annotations

from time import perf_counter

from sanitizepy import Cleaner, CleanerConfig


def benchmark_sanitizepy_initialization(
    iterations: int = 10_000,
) -> float:
    """
    Benchmark Cleaner initialization with default configuration.

    Parameters
    ----------
    iterations:
        Number of Cleaner instances to create.

    Returns
    -------
    float
        Average initialization time in seconds.
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    start = perf_counter()

    for _ in range(iterations):
        Cleaner()

    elapsed = perf_counter() - start

    return elapsed / iterations


def benchmark_sanitizepy_with_custom_config(
    iterations: int = 10_000,
) -> float:
    """
    Benchmark Cleaner initialization with explicit custom configuration.
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    config = CleanerConfig(
        preview_rows=20,
        top_values=15,
        float_precision=4,
        enable_logging=False,
    )

    start = perf_counter()

    for _ in range(iterations):
        Cleaner(config=config)

    elapsed = perf_counter() - start

    return elapsed / iterations


def benchmark_config_validation(
    iterations: int = 10_000,
) -> float:
    """
    Benchmark CleanerConfig creation and post-init validation.
    """
    if iterations <= 0:
        raise ValueError("iterations must be greater than zero")

    start = perf_counter()

    for _ in range(iterations):
        CleanerConfig(
            preview_rows=10,
            top_values=5,
            float_precision=2,
        )

    elapsed = perf_counter() - start

    return elapsed / iterations


def main() -> None:
    """Run the Cleaner core benchmark suite."""
    init_avg = benchmark_sanitizepy_initialization()
    custom_avg = benchmark_sanitizepy_with_custom_config()
    config_avg = benchmark_config_validation()

    print("=== Cleaner Core Benchmarks ===")
    print(f"Cleaner default initialization avg: {init_avg:.9f} s")
    print(f"Cleaner custom config init avg:     {custom_avg:.9f} s")
    print(f"CleanerConfig validation avg:       {config_avg:.9f} s")


if __name__ == "__main__":
    main()