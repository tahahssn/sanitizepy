"""
Benchmarks for the Cleaner reporting engine, renderers, and exporters.
"""

from __future__ import annotations

from time import perf_counter

from sanitizepy.reports import (
    JSONRenderer,
    ReportEngine,
    StringExporter,
    TextRenderer,
)


def _generate_report_input() -> dict[str, object]:
    """
    Generate representative structured processing results for report engine.
    """
    return {
        "dataset_summary": {
            "total_rows": 10_000,
            "total_columns": 15,
            "memory_mb": 4.5,
        },
        "missing_values": {
            "missing_cells": 120,
            "missing_percentage": 0.08,
            "affected_columns": ["col_b", "col_f"],
        },
        "duplicates": {
            "duplicate_rows": 50,
            "duplicate_percentage": 0.005,
        },
        "datatype_distribution": {
            "integer": 5,
            "float": 5,
            "string": 3,
            "datetime": 2,
        },
        "recommendations": [
            "Fill missing values in col_b using median imputer.",
            "Remove 50 duplicated rows.",
        ],
    }


def benchmark_report_engine(
    runs: int = 1_000,
) -> float:
    """Benchmark ReportEngine.run()."""
    results = _generate_report_input()
    engine = ReportEngine()

    start = perf_counter()
    for _ in range(runs):
        engine.run(results, title="Benchmark Inspection Report")
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_text_rendering(
    runs: int = 1_000,
) -> float:
    """Benchmark TextRenderer.render()."""
    results = _generate_report_input()
    report = ReportEngine().run(results, title="Benchmark Report")
    renderer = TextRenderer()

    start = perf_counter()
    for _ in range(runs):
        renderer.render(report)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_json_rendering(
    runs: int = 1_000,
) -> float:
    """Benchmark JSONRenderer.render()."""
    results = _generate_report_input()
    report = ReportEngine().run(results, title="Benchmark Report")
    renderer = JSONRenderer()

    start = perf_counter()
    for _ in range(runs):
        renderer.render(report)
    elapsed = perf_counter() - start

    return elapsed / runs


def benchmark_string_exporter(
    runs: int = 1_000,
) -> float:
    """Benchmark StringExporter.export()."""
    results = _generate_report_input()
    report = ReportEngine().run(results, title="Benchmark Report")
    renderer = JSONRenderer()
    exporter = StringExporter()

    start = perf_counter()
    for _ in range(runs):
        exporter.export(report, renderer)
    elapsed = perf_counter() - start

    return elapsed / runs


def main() -> None:
    """Run reports benchmarks."""
    report_engine_time = benchmark_report_engine()
    text_render_time = benchmark_text_rendering()
    json_render_time = benchmark_json_rendering()
    exporter_time = benchmark_string_exporter()

    print("=== Reporting Engine Benchmarks ===")
    print(f"ReportEngine.run avg:      {report_engine_time:.8f} s")
    print(f"TextRenderer.render avg:   {text_render_time:.8f} s")
    print(f"JSONRenderer.render avg:   {json_render_time:.8f} s")
    print(f"StringExporter.export avg: {exporter_time:.8f} s")


if __name__ == "__main__":
    main()
