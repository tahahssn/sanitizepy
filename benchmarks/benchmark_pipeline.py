"""
Benchmarks for the Cleaner pipeline execution engine and step adapters.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from sanitizepy.cleaning import DropDuplicates, FillMissing
from sanitizepy.pipeline import CallableStep, PipelineEngine, TransformStep
from sanitizepy.preprocessing import (
    ColumnInteraction,
    FeatureEngineeringEngine,
    LogFeature,
)


def _generate_pipeline_data(
    rows: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate deterministic DataFrame for pipeline benchmarks.
    """
    rng = np.random.default_rng(seed)

    col_a = rng.uniform(1.0, 100.0, size=rows)
    col_b = rng.uniform(1.0, 50.0, size=rows)

    nan_mask = rng.random(size=rows) < 0.1
    col_b[nan_mask] = np.nan

    df = pd.DataFrame(
        {
            "feature_a": col_a,
            "feature_b": col_b,
        }
    )

    df = pd.concat([df, df.iloc[:200]], ignore_index=True)
    return df


def benchmark_pipeline_with_callable_steps(
    runs: int = 50,
) -> float:
    """Benchmark PipelineEngine using CallableStep adapters."""
    df = _generate_pipeline_data()
    dup_op = DropDuplicates()
    fill_op = FillMissing(value=0.0)

    engine = PipelineEngine(
        [
            CallableStep("drop_duplicates", dup_op.apply),
            CallableStep("fill_missing", fill_op.apply),
        ]
    )

    start = perf_counter()
    for _ in range(runs):
        engine.run(df)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_pipeline_with_transform_steps(
    runs: int = 50,
) -> float:
    """Benchmark PipelineEngine using TransformStep wrapping FeatureEngineeringEngine."""
    df = _generate_pipeline_data()

    fe_engine = FeatureEngineeringEngine(
        [
            ColumnInteraction(column_a="feature_a", column_b="feature_b"),
            LogFeature(column="feature_a"),
        ]
    )
    # Fill missing values first so fe_engine can transform cleanly
    fill_op = FillMissing(value=1.0)
    clean_df = fill_op.apply(df)
    fe_engine.fit(clean_df)

    pipeline = PipelineEngine(
        [
            TransformStep("feature_engineering", fe_engine),
        ]
    )

    start = perf_counter()
    for _ in range(runs):
        pipeline.run(clean_df)
    elapsed = perf_counter() - start

    return elapsed / runs


def main() -> None:
    """Run pipeline benchmarks."""
    callable_time = benchmark_pipeline_with_callable_steps()
    transform_time = benchmark_pipeline_with_transform_steps()

    print("=== Pipeline Engine Benchmarks ===")
    print(f"Pipeline with CallableSteps avg:  {callable_time:.6f} s")
    print(f"Pipeline with TransformSteps avg: {transform_time:.6f} s")


if __name__ == "__main__":
    main()
