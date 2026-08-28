"""
Benchmarks for the Cleaner inspection engine.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from sanitizepy.inspection import (
    DatatypeInspector,
    DuplicateInspector,
    MemoryInspector,
    MissingValueInspector,
    StatisticsInspector,
)


def _generate_inspection_data(
    rows: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a deterministic representative DataFrame for inspection benchmarks.
    """
    rng = np.random.default_rng(seed)

    int_col = rng.integers(0, 100, size=rows)
    float_col = rng.normal(50.0, 15.0, size=rows)

    # Insert missing values deterministically
    float_with_nan = float_col.copy()
    nan_mask = rng.random(size=rows) < 0.1
    float_with_nan[nan_mask] = np.nan

    str_choices = np.array(["Alpha", "Beta", "Gamma", "Delta", None], dtype=object)
    str_col = rng.choice(str_choices, size=rows)

    dates = pd.date_range("2025-01-01", periods=rows, freq="h")

    df = pd.DataFrame(
        {
            "id": int_col,
            "value_a": float_col,
            "value_b": float_with_nan,
            "category": str_col,
            "timestamp": dates,
        }
    )

    # Duplicate top 100 rows to test duplicate detection
    df = pd.concat([df, df.iloc[:100]], ignore_index=True)
    return df


def benchmark_missing_value_inspection(
    runs: int = 50,
) -> float:
    """Benchmark MissingValueInspector.inspect()."""
    df = _generate_inspection_data()
    inspector = MissingValueInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.inspect(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_duplicate_inspection(
    runs: int = 50,
) -> float:
    """Benchmark DuplicateInspector.inspect()."""
    df = _generate_inspection_data()
    inspector = DuplicateInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.inspect(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_datatype_inspection(
    runs: int = 50,
) -> float:
    """Benchmark DatatypeInspector.inspect()."""
    df = _generate_inspection_data()
    inspector = DatatypeInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.inspect(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_memory_inspection(
    runs: int = 50,
) -> float:
    """Benchmark MemoryInspector.inspect()."""
    df = _generate_inspection_data()
    inspector = MemoryInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.inspect(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_statistics_inspection(
    runs: int = 50,
) -> float:
    """Benchmark StatisticsInspector.inspect()."""
    df = _generate_inspection_data()
    inspector = StatisticsInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.inspect(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_correlation_matrix(
    runs: int = 50,
) -> float:
    """Benchmark StatisticsInspector.correlation()."""
    df = _generate_inspection_data()
    inspector = StatisticsInspector()

    start = perf_counter()
    for _ in range(runs):
        inspector.correlation(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def main() -> None:
    """Run inspection benchmarks."""
    missing_time = benchmark_missing_value_inspection()
    duplicate_time = benchmark_duplicate_inspection()
    datatype_time = benchmark_datatype_inspection()
    memory_time = benchmark_memory_inspection()
    statistics_time = benchmark_statistics_inspection()
    correlation_time = benchmark_correlation_matrix()

    print("=== Inspection Engine Benchmarks ===")
    print(f"MissingValueInspector avg: {missing_time:.6f} s")
    print(f"DuplicateInspector avg:    {duplicate_time:.6f} s")
    print(f"DatatypeInspector avg:     {datatype_time:.6f} s")
    print(f"MemoryInspector avg:       {memory_time:.6f} s")
    print(f"StatisticsInspector avg:   {statistics_time:.6f} s")
    print(f"Correlation Matrix avg:    {correlation_time:.6f} s")


if __name__ == "__main__":
    main()
