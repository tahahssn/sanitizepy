"""
Benchmarks for the Cleaner preprocessing / feature engineering engine.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from sanitizepy.preprocessing import (
    ColumnInteraction,
    DatetimeFeatures,
    FeatureEngineeringEngine,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)


def _generate_preprocessing_data(
    rows: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic DataFrame for preprocessing benchmarks.
    """
    rng = np.random.default_rng(seed)

    col_a = rng.uniform(1.0, 100.0, size=rows)
    col_b = rng.uniform(1.0, 50.0, size=rows)
    dates = pd.date_range("2025-01-01", periods=rows, freq="min")

    return pd.DataFrame(
        {
            "feature_a": col_a,
            "feature_b": col_b,
            "created_at": dates,
        }
    )


def benchmark_column_interaction(
    runs: int = 50,
) -> float:
    """Benchmark ColumnInteraction fit_transform()."""
    df = _generate_preprocessing_data()
    op = ColumnInteraction(column_a="feature_a", column_b="feature_b")

    start = perf_counter()
    for _ in range(runs):
        op.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_ratio_feature(
    runs: int = 50,
) -> float:
    """Benchmark RatioFeature fit_transform()."""
    df = _generate_preprocessing_data()
    op = RatioFeature(numerator="feature_a", denominator="feature_b")

    start = perf_counter()
    for _ in range(runs):
        op.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_polynomial_feature(
    runs: int = 50,
) -> float:
    """Benchmark PolynomialFeature fit_transform()."""
    df = _generate_preprocessing_data()
    op = PolynomialFeature(column="feature_a", degree=3)

    start = perf_counter()
    for _ in range(runs):
        op.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_log_feature(
    runs: int = 50,
) -> float:
    """Benchmark LogFeature fit_transform()."""
    df = _generate_preprocessing_data()
    op = LogFeature(column="feature_a")

    start = perf_counter()
    for _ in range(runs):
        op.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_datetime_features(
    runs: int = 50,
) -> float:
    """Benchmark DatetimeFeatures fit_transform()."""
    df = _generate_preprocessing_data()
    op = DatetimeFeatures(
        column="created_at",
        features=["year", "month", "day", "day_of_week", "hour"],
    )

    start = perf_counter()
    for _ in range(runs):
        op.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_feature_engineering_engine(
    runs: int = 50,
) -> float:
    """Benchmark FeatureEngineeringEngine fit_transform()."""
    df = _generate_preprocessing_data()
    engine = FeatureEngineeringEngine(
        [
            ColumnInteraction(column_a="feature_a", column_b="feature_b"),
            RatioFeature(numerator="feature_a", denominator="feature_b"),
            LogFeature(column="feature_a"),
            DatetimeFeatures(
                column="created_at",
                features=["year", "month", "day", "day_of_week", "hour"],
            ),
        ]
    )

    start = perf_counter()
    for _ in range(runs):
        engine.fit_transform(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def main() -> None:
    """Run preprocessing benchmarks."""
    interaction_time = benchmark_column_interaction()
    ratio_time = benchmark_ratio_feature()
    poly_time = benchmark_polynomial_feature()
    log_time = benchmark_log_feature()
    datetime_time = benchmark_datetime_features()
    engine_time = benchmark_feature_engineering_engine()

    print("=== Preprocessing Engine Benchmarks ===")
    print(f"ColumnInteraction avg:     {interaction_time:.6f} s")
    print(f"RatioFeature avg:          {ratio_time:.6f} s")
    print(f"PolynomialFeature avg:     {poly_time:.6f} s")
    print(f"LogFeature avg:            {log_time:.6f} s")
    print(f"DatetimeFeatures avg:      {datetime_time:.6f} s")
    print(f"FeatureEngineeringEngine:  {engine_time:.6f} s")


if __name__ == "__main__":
    main()
