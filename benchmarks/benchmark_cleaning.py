"""
Benchmarks for the Cleaner cleaning engine and operations.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from sanitizepy.cleaning import (
    CleaningEngine,
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)


def _generate_cleaning_data(
    rows: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic DataFrame for cleaning benchmarks.
    """
    rng = np.random.default_rng(seed)

    val_a = rng.normal(100.0, 25.0, size=rows)
    val_b = rng.normal(0.0, 1.0, size=rows)

    nan_mask = rng.random(size=rows) < 0.15
    val_b[nan_mask] = np.nan

    unwanted_col = rng.integers(0, 10, size=rows)

    df = pd.DataFrame(
        {
            "feature_a": val_a,
            "feature_b": val_b,
            "extra_col": unwanted_col,
        }
    )

    # Add duplicate rows
    df = pd.concat([df, df.iloc[:500]], ignore_index=True)
    return df


def benchmark_drop_missing_rows(
    runs: int = 50,
) -> float:
    """Benchmark DropMissingRows.apply()."""
    df = _generate_cleaning_data()
    op = DropMissingRows()

    start = perf_counter()
    for _ in range(runs):
        op.apply(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_drop_missing_columns(
    runs: int = 50,
) -> float:
    """Benchmark DropMissingColumns.apply()."""
    df = _generate_cleaning_data()
    op = DropMissingColumns()

    start = perf_counter()
    for _ in range(runs):
        op.apply(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_fill_missing(
    runs: int = 50,
) -> float:
    """Benchmark FillMissing.apply()."""
    df = _generate_cleaning_data()
    op = FillMissing(value=0.0)

    start = perf_counter()
    for _ in range(runs):
        op.apply(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_drop_duplicates(
    runs: int = 50,
) -> float:
    """Benchmark DropDuplicates.apply()."""
    df = _generate_cleaning_data()
    op = DropDuplicates()

    start = perf_counter()
    for _ in range(runs):
        op.apply(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_drop_columns(
    runs: int = 50,
) -> float:
    """Benchmark DropColumns.apply()."""
    df = _generate_cleaning_data()
    op = DropColumns(columns=["extra_col"])

    start = perf_counter()
    for _ in range(runs):
        op.apply(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_cleaning_engine_pipeline(
    runs: int = 50,
) -> float:
    """Benchmark CleaningEngine orchestrating multiple operations."""
    df = _generate_cleaning_data()
    engine = CleaningEngine(
        [
            DropDuplicates(),
            FillMissing(value=0.0, subset=["feature_b"]),
            DropColumns(columns=["extra_col"]),
        ]
    )

    start = perf_counter()
    for _ in range(runs):
        engine.run(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def main() -> None:
    """Run cleaning benchmarks."""
    drop_rows_time = benchmark_drop_missing_rows()
    drop_cols_time = benchmark_drop_missing_columns()
    fill_time = benchmark_fill_missing()
    drop_dup_time = benchmark_drop_duplicates()
    drop_col_time = benchmark_drop_columns()
    engine_time = benchmark_cleaning_engine_pipeline()

    print("=== Cleaning Engine Benchmarks ===")
    print(f"DropMissingRows avg:       {drop_rows_time:.6f} s")
    print(f"DropMissingColumns avg:    {drop_cols_time:.6f} s")
    print(f"FillMissing avg:           {fill_time:.6f} s")
    print(f"DropDuplicates avg:        {drop_dup_time:.6f} s")
    print(f"DropColumns avg:           {drop_col_time:.6f} s")
    print(f"CleaningEngine pipeline:   {engine_time:.6f} s")


if __name__ == "__main__":
    main()
