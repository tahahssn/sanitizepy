"""
Smoke test for the end-to-end scalability benchmark harness.

Runs :func:`benchmarks.benchmark_end_to_end.run_benchmark` on a tiny frame
(for speed) and asserts the harness records a duration and a peak-memory
figure without error, along with the other recorded metrics.

Requirements: 6.7
"""

from __future__ import annotations

from benchmarks.benchmark_end_to_end import BenchmarkMetrics, run_benchmark

# A tiny frame keeps the smoke test fast while still exercising the full
# inspect -> plan -> clean flow.
SMOKE_ROWS = 200


def test_run_benchmark_records_duration_and_peak_memory() -> None:
    """The harness runs on a tiny frame and records duration + peak memory."""
    metrics = run_benchmark(rows=SMOKE_ROWS)

    assert isinstance(metrics, BenchmarkMetrics)

    # Duration and peak memory must be recorded as positive figures.
    assert metrics.duration_seconds > 0
    assert metrics.peak_memory_bytes > 0

    # Row count accounts for the appended duplicate rows, so it is >= the
    # requested base row count.
    assert metrics.row_count >= SMOKE_ROWS

    # Result shape is a 2-tuple (rows, cols).
    assert isinstance(metrics.result_shape, tuple)
    assert len(metrics.result_shape) == 2

    # The operation sequence is recorded as a list.
    assert isinstance(metrics.operation_sequence, list)
