"""
End-to-end scalability benchmark for the sanitizepy high-level pipeline.

Exercises the full ``inspect -> plan -> clean`` data flow over a large, messy
synthetic dataset (at least 30,000 rows by default), recording the total
wall-clock duration and peak memory usage via :mod:`tracemalloc`, along with
the row/column counts, the sequence of operations the plan executed, and the
resulting DataFrame shape.

The benchmark is both runnable as a script and importable: :func:`run_benchmark`
returns a :class:`BenchmarkMetrics` instance so callers (for example a smoke
test) can drive the harness on a tiny frame and assert the recorded metrics.

Validates: Requirements 6.6, 6.7
"""

from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import pandas as pd

from sanitizepy.core import clean, inspect, plan

# Minimum row count mandated by the end-to-end scalability requirement.
MIN_BENCHMARK_ROWS = 30_000


@dataclass
class BenchmarkMetrics:
    """
    Recorded metrics from a single end-to-end benchmark run.

    Attributes
    ----------
    row_count:
        Number of rows in the generated input DataFrame.
    column_count:
        Number of columns in the generated input DataFrame.
    operation_sequence:
        Ordered names of the operations the plan executed against the data.
    result_shape:
        ``(rows, cols)`` shape of the cleaned DataFrame.
    duration_seconds:
        Total wall-clock duration of the ``inspect -> plan -> clean`` flow.
    peak_memory_bytes:
        Peak memory (in bytes) allocated during the flow, per ``tracemalloc``.
    """

    row_count: int
    column_count: int
    operation_sequence: list[str] = field(default_factory=list)
    result_shape: tuple[int, int] = (0, 0)
    duration_seconds: float = 0.0
    peak_memory_bytes: int = 0

    @property
    def peak_memory_mib(self) -> float:
        """Peak memory usage expressed in mebibytes."""
        return self.peak_memory_bytes / (1024 * 1024)


def generate_messy_dataframe(
    rows: int = MIN_BENCHMARK_ROWS,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a deterministic, messy DataFrame with a mix of column types.

    The frame deliberately contains real cleaning work so the generated plan
    is non-trivial:

    * numeric columns with injected missing values,
    * object/text columns with surrounding/collapsible whitespace and tokens,
    * an all-but-constant column that is a strong drop candidate,
    * duplicated rows appended to the tail.

    Parameters
    ----------
    rows:
        Number of base rows to generate before duplicates are appended.
    seed:
        Seed for the random generator so runs are reproducible.

    Returns
    -------
    pandas.DataFrame
        The generated messy DataFrame.
    """
    rng = np.random.default_rng(seed)

    # Numeric columns with ~10-15% missing values injected.
    revenue = rng.uniform(10.0, 10_000.0, size=rows)
    revenue[rng.random(size=rows) < 0.12] = np.nan

    quantity = rng.integers(1, 500, size=rows).astype("float64")
    quantity[rng.random(size=rows) < 0.10] = np.nan

    # Text/object column with whitespace noise and a small token vocabulary.
    categories = np.array(["  alpha ", "beta", "GAMMA  ", " delta", "epsilon "])
    category = rng.choice(categories, size=rows)

    labels = np.array(["  yes", "no ", " maybe ", "YES", "No"])
    status = rng.choice(labels, size=rows)

    # Low-cardinality object column that is a strong drop/constant candidate.
    constant_ish = np.where(rng.random(size=rows) < 0.999, "active", "inactive")

    frame = pd.DataFrame(
        {
            "revenue": revenue,
            "quantity": quantity,
            "category": category,
            "status": status,
            "account_state": constant_ish,
        }
    )

    # Append duplicate rows so duplicate detection/removal has real work.
    duplicate_count = max(1, rows // 20)
    frame = pd.concat(
        [frame, frame.iloc[:duplicate_count]],
        ignore_index=True,
    )

    return frame


def run_benchmark(
    rows: int = MIN_BENCHMARK_ROWS,
    seed: int = 42,
) -> BenchmarkMetrics:
    """
    Run the full ``inspect -> plan -> clean`` flow and record metrics.

    The wall-clock timer and :mod:`tracemalloc` peak-memory tracking wrap the
    entire pipeline so the reported figures cover inspection, planning, and
    cleaning together.

    Parameters
    ----------
    rows:
        Number of base rows to generate. Kept configurable so a smoke test can
        drive the harness on a tiny frame; defaults to the 30,000-row minimum.
    seed:
        Seed forwarded to :func:`generate_messy_dataframe` for reproducibility.

    Returns
    -------
    BenchmarkMetrics
        The recorded metrics for the run.
    """
    dataframe = generate_messy_dataframe(rows=rows, seed=seed)
    row_count, column_count = dataframe.shape

    tracemalloc.start()
    start = perf_counter()

    report = inspect(dataframe)
    cleaning_plan = plan(report)
    result = clean(dataframe, cleaning_plan=cleaning_plan)

    duration_seconds = perf_counter() - start
    _, peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    operation_sequence = [op.operation_name for op in result.operations]

    return BenchmarkMetrics(
        row_count=int(row_count),
        column_count=int(column_count),
        operation_sequence=operation_sequence,
        result_shape=result.data.shape,
        duration_seconds=duration_seconds,
        peak_memory_bytes=int(peak_memory_bytes),
    )


def main() -> None:
    """Run the end-to-end benchmark and print the recorded metrics."""
    metrics = run_benchmark()

    print("=== sanitizepy End-to-End Scalability Benchmark ===")
    print(f"Rows (input):      {metrics.row_count}")
    print(f"Columns (input):   {metrics.column_count}")
    print(f"Result shape:      {metrics.result_shape}")
    print(f"Total duration:    {metrics.duration_seconds:.4f} s")
    print(f"Peak memory:       {metrics.peak_memory_mib:.2f} MiB")
    if metrics.operation_sequence:
        print("Operation sequence:")
        for index, name in enumerate(metrics.operation_sequence, start=1):
            print(f"  {index}. {name}")
    else:
        print("Operation sequence: (none)")


if __name__ == "__main__":
    main()
