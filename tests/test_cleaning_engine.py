"""
Tests for sanitizepy.cleaning.engine.CleaningEngine execution modes.

This module contains:

1. Property tests
   * Task 7.2 — **Property 6: Dry-run never mutates input**
     **Validates: Requirements 6.2, 7.4, 17.4**
   * Task 7.3 — **Property 22: Chunk-safe execution matches whole-dataset
     execution**
     **Validates: Requirements 6.4**
   * Task 7.4 — **Property 23: Non-chunk-safe operations never execute
     incorrectly per chunk**
     **Validates: Requirements 6.5, 6.6**

2. Unit tests (task 7.5) — Requirements: 6.1, 6.3, 6.5
   Covers: the in-place-safe path avoids the defensive copy, chunked-mode
   correctness, chunk_size handling/validation, and whole-dataset fallback,
   plus edge cases (empty, single-row, all-null, mixed-type, infinite, wide,
   tall).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sanitizepy.cleaning.base import CleaningOperation, OperationResult
from sanitizepy.cleaning.engine import CleaningEngine
from sanitizepy.cleaning.missing_tokens import MissingTokenOperation
from sanitizepy.cleaning.operations import (
    DropDuplicates,
    DropMissingRows,
    FillMissing,
)
from sanitizepy.cleaning.text_normalization import TextNormalizationOperation
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation

# ===========================================================================
# Shared helpers and probe operations
# ===========================================================================


class _RecordingChunkOp(CleaningOperation):
    """
    Chunk-safe, deterministic probe operation.

    Records the shape of every DataFrame it receives so tests can assert
    how many partitions the engine handed to it (one call per chunk in
    chunked mode, a single whole-dataset call otherwise). The transform
    itself is per-cell (add a constant to a numeric column) so results are
    independent of how rows are partitioned.
    """

    name = "recording_chunk_op"
    is_chunk_safe = True

    def __init__(self, column: str = "num", delta: int = 1) -> None:
        self.column = column
        self.delta = delta
        self.received_shapes: list[tuple[int, int]] = []

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        self.received_shapes.append(dataframe.shape)
        result = dataframe.copy()
        result[self.column] = result[self.column] + self.delta
        return result

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "column": self.column, "delta": self.delta}


class _RecordingWholeOp(CleaningOperation):
    """
    Non-chunk-safe probe operation.

    Records every received shape so tests can prove the engine never
    partitioned the frame. The transform (drop duplicate rows) is genuinely
    dataset-wide, so per-chunk execution would produce a different result.
    """

    name = "recording_whole_op"
    is_chunk_safe = False

    def __init__(self) -> None:
        self.received_shapes: list[tuple[int, int]] = []

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        self.received_shapes.append(dataframe.shape)
        return dataframe.drop_duplicates().copy()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


class _InplaceSafeOp(CleaningOperation):
    """
    In-place-safe probe operation.

    Records the ``id()`` of the object it receives so the test can prove
    the engine passed the caller's DataFrame straight through (no defensive
    copy) in the non-dry-run path.
    """

    name = "inplace_safe_op"
    is_inplace_safe = True

    def __init__(self) -> None:
        self.received_id: int | None = None

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        self.received_id = id(dataframe)
        # Return the same object unchanged; this operation is a no-op that
        # only observes identity.
        return dataframe

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        self.received_id = id(dataframe)
        result = OperationResult(
            operation_name=self.name,
            before_shape=dataframe.shape,
            after_shape=dataframe.shape,
            dry_run=dry_run,
        )
        final_df = dataframe.copy() if dry_run else dataframe
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def _messy_frame() -> pd.DataFrame:
    """A representative mixed-content frame used by several unit tests."""
    return pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, 6],
            "token": ["n/a", "ok", "NULL", "value", "  na ", "keep"],
            "text": ["  Hi ", "world", "FOO", "  ", "b\u00a0ar", "baz"],
        }
    )


# ===========================================================================
# Hypothesis strategies
# ===========================================================================

_finite_float = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)

_num_with_missing = st.lists(
    st.one_of(_finite_float, st.none()),
    min_size=1,
    max_size=40,
)


@st.composite
def _token_frame(draw: st.DrawFn) -> pd.DataFrame:
    """Draw a frame with a numeric column and an object token column."""
    size = draw(st.integers(min_value=1, max_value=40))
    tokens = st.sampled_from(
        ["n/a", "NULL", "value", "keep", "  na ", "ok", "None", "-"]
    )
    return pd.DataFrame(
        {
            "num": draw(
                st.lists(
                    st.integers(min_value=-1000, max_value=1000),
                    min_size=size,
                    max_size=size,
                )
            ),
            "token": draw(st.lists(tokens, min_size=size, max_size=size)),
        }
    )


# ===========================================================================
# Property tests — Task 7.2
# **Property 6: Dry-run never mutates input**
# **Validates: Requirements 6.2, 7.4, 17.4**
# ===========================================================================


class TestDryRunNonMutation:
    """
    **Property 6: Dry-run never mutates input**
    **Validates: Requirements 6.2, 7.4, 17.4**
    """

    @given(values=_num_with_missing)
    @settings(max_examples=80)
    def test_dry_run_leaves_caller_frame_unchanged(
        self, values: list[float | None]
    ) -> None:
        df = pd.DataFrame({"num": values})
        snapshot = df.copy(deep=True)

        engine = CleaningEngine(
            [
                MissingTokenOperation(),
                FillMissing(strategy="constant", value=0),
                DropDuplicates(),
            ]
        )
        engine.run(df, dry_run=True)

        pd.testing.assert_frame_equal(df, snapshot)

    @given(frame=_token_frame())
    @settings(max_examples=60)
    def test_dry_run_mixed_operations_do_not_mutate(self, frame: pd.DataFrame) -> None:
        snapshot = frame.copy(deep=True)

        engine = CleaningEngine(
            [
                MissingTokenOperation(),
                TextNormalizationOperation(subset=["token"], case="lower"),
                DropDuplicates(),
            ]
        )
        result = engine.run_with_result(frame, dry_run=True)

        # Caller frame untouched.
        pd.testing.assert_frame_equal(frame, snapshot)
        # And the returned data equals the (unmodified) caller frame.
        pd.testing.assert_frame_equal(result.data, snapshot)

    @given(frame=_token_frame())
    @settings(max_examples=60, deadline=None)
    def test_dry_run_non_mutation_holds_with_chunking(
        self, frame: pd.DataFrame
    ) -> None:
        snapshot = frame.copy(deep=True)

        engine = CleaningEngine([MissingTokenOperation()])
        engine.run(frame, dry_run=True, chunk_size=3)

        pd.testing.assert_frame_equal(frame, snapshot)


# ===========================================================================
# Property tests — Task 7.3
# **Property 22: Chunk-safe execution matches whole-dataset execution**
# **Validates: Requirements 6.4**
# ===========================================================================


class TestChunkSafeEquivalence:
    """
    **Property 22: Chunk-safe execution matches whole-dataset execution**
    **Validates: Requirements 6.4**
    """

    @given(
        frame=_token_frame(),
        chunk_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=80, deadline=None)
    def test_missing_token_chunked_matches_whole(
        self, frame: pd.DataFrame, chunk_size: int
    ) -> None:
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, chunked)

    @given(
        frame=_token_frame(),
        chunk_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=80, deadline=None)
    def test_text_normalization_chunked_matches_whole(
        self, frame: pd.DataFrame, chunk_size: int
    ) -> None:
        engine = CleaningEngine(
            [TextNormalizationOperation(subset=["token"], case="upper")]
        )
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, chunked)

    @given(
        values=st.lists(
            st.integers(min_value=-1000, max_value=1000).map(str),
            min_size=1,
            max_size=40,
        ),
        chunk_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=80, deadline=None)
    def test_type_coercion_chunked_matches_whole(
        self, values: list[str], chunk_size: int
    ) -> None:
        frame = pd.DataFrame({"val": values})
        engine = CleaningEngine(
            [TypeCoercionOperation({"val": "int64"}, error_policy="coerce")]
        )
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, chunked)

    @given(
        frame=_token_frame(),
        chunk_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=60, deadline=None)
    def test_chunked_aggregated_metrics_match_whole(
        self, frame: pd.DataFrame, chunk_size: int
    ) -> None:
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run_with_result(frame, chunk_size=None)
        chunked = engine.run_with_result(frame, chunk_size=chunk_size)

        assert len(whole.operations) == len(chunked.operations) == 1
        assert whole.operations[0].rows_affected == chunked.operations[0].rows_affected
        assert whole.operations[0].after_shape == chunked.operations[0].after_shape


# ===========================================================================
# Property tests — Task 7.4
# **Property 23: Non-chunk-safe operations never execute incorrectly per
# chunk**
# **Validates: Requirements 6.5, 6.6**
# ===========================================================================


class TestNonChunkSafeWholeDatasetFallback:
    """
    **Property 23: Non-chunk-safe operations never execute incorrectly per
    chunk**
    **Validates: Requirements 6.5, 6.6**
    """

    @given(
        n_repeats=st.integers(min_value=2, max_value=6),
        chunk_size=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=60)
    def test_non_chunk_safe_op_receives_whole_dataset(
        self, n_repeats: int, chunk_size: int
    ) -> None:
        # A frame where every row is identical, so a correct whole-dataset
        # DropDuplicates collapses to exactly one row. A naive per-chunk
        # application would instead leave one row per chunk.
        frame = pd.DataFrame({"a": [7] * n_repeats, "b": ["x"] * n_repeats})
        probe = _RecordingWholeOp()
        engine = CleaningEngine([probe])

        result = engine.run(frame, chunk_size=chunk_size)

        # The operation was invoked exactly once, on the entire frame.
        assert probe.received_shapes == [frame.shape]
        # Correct whole-dataset semantics: all identical rows collapse to one.
        assert len(result) == 1

    @given(
        frame=_token_frame(),
        chunk_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=60, deadline=None)
    def test_non_chunk_safe_result_equals_non_chunked(
        self, frame: pd.DataFrame, chunk_size: int
    ) -> None:
        # DropDuplicates is not chunk-safe; passing chunk_size must not
        # change its output relative to whole-dataset execution.
        engine = CleaningEngine([DropDuplicates()])
        whole = engine.run(frame, chunk_size=None)
        with_chunk = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, with_chunk)

    @given(chunk_size=st.integers(min_value=1, max_value=4))
    @settings(max_examples=40)
    def test_fill_missing_statistical_is_not_chunked(self, chunk_size: int) -> None:
        # Median over the whole column differs from per-chunk medians, so
        # this proves the engine did not chunk a non-chunk-safe FillMissing.
        frame = pd.DataFrame({"num": [1.0, np.nan, 100.0, np.nan, 3.0, 2.0]})
        assert FillMissing(strategy="median").is_chunk_safe is False

        engine = CleaningEngine([FillMissing(strategy="median")])
        whole = engine.run(frame, chunk_size=None)
        with_chunk = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, with_chunk)


# ===========================================================================
# Unit tests — Task 7.5
# Requirements: 6.1, 6.3, 6.5
# ===========================================================================


class TestInplaceSafePathAvoidsCopy:
    """The engine skips the defensive copy for in-place-safe operations."""

    def test_non_dry_run_passes_caller_frame_by_identity(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        probe = _InplaceSafeOp()
        engine = CleaningEngine([probe])

        result = engine.run(df, dry_run=False)

        # No defensive copy: the operation received the exact caller object.
        assert probe.received_id == id(df)
        assert result is df

    def test_mixed_flags_force_defensive_copy(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        probe = _InplaceSafeOp()
        # DropDuplicates is not in-place-safe, so the engine must copy.
        engine = CleaningEngine([probe, DropDuplicates()])

        engine.run(df, dry_run=False)

        assert probe.received_id != id(df)

    def test_dry_run_always_copies_even_when_inplace_safe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        probe = _InplaceSafeOp()
        engine = CleaningEngine([probe])

        engine.run(df, dry_run=True)

        assert probe.received_id != id(df)


class TestChunkedModeCorrectness:
    """Chunked execution partitions only chunk-safe operations."""

    def test_chunk_safe_op_called_once_per_chunk(self) -> None:
        frame = pd.DataFrame({"num": list(range(10))})
        probe = _RecordingChunkOp(column="num", delta=5)
        engine = CleaningEngine([probe])

        result = engine.run(frame, chunk_size=4)

        # 10 rows / chunk_size 4 -> chunks of 4, 4, 2.
        assert probe.received_shapes == [(4, 1), (4, 1), (2, 1)]
        expected = pd.DataFrame({"num": [v + 5 for v in range(10)]})
        pd.testing.assert_frame_equal(result, expected)

    def test_chunk_safe_op_single_call_without_chunk_size(self) -> None:
        frame = pd.DataFrame({"num": list(range(10))})
        probe = _RecordingChunkOp(column="num", delta=1)
        engine = CleaningEngine([probe])

        engine.run(frame, chunk_size=None)

        assert probe.received_shapes == [(10, 1)]

    def test_chunk_size_larger_than_frame_uses_single_chunk(self) -> None:
        frame = pd.DataFrame({"num": [1, 2, 3]})
        probe = _RecordingChunkOp(column="num")
        engine = CleaningEngine([probe])

        engine.run(frame, chunk_size=100)

        assert probe.received_shapes == [(3, 1)]


class TestChunkSizeHandling:
    """chunk_size validation and boundary handling."""

    def test_zero_chunk_size_raises_value_error(self) -> None:
        engine = CleaningEngine([MissingTokenOperation()])
        with pytest.raises(ValueError, match="positive integer"):
            engine.run(pd.DataFrame({"a": ["x"]}), chunk_size=0)

    def test_negative_chunk_size_raises_value_error(self) -> None:
        engine = CleaningEngine([MissingTokenOperation()])
        with pytest.raises(ValueError, match="positive integer"):
            engine.run_with_result(pd.DataFrame({"a": ["x"]}), chunk_size=-3)

    def test_chunk_size_of_one_row_per_chunk(self) -> None:
        frame = pd.DataFrame({"num": [10, 20, 30]})
        probe = _RecordingChunkOp(column="num", delta=0)
        engine = CleaningEngine([probe])

        engine.run(frame, chunk_size=1)

        assert probe.received_shapes == [(1, 1), (1, 1), (1, 1)]

    def test_empty_frame_skips_chunking(self) -> None:
        frame = pd.DataFrame({"num": pd.Series([], dtype="int64")})
        probe = _RecordingChunkOp(column="num")
        engine = CleaningEngine([probe])

        engine.run(frame, chunk_size=5)

        # Non-empty guard: an empty frame runs whole-dataset (one call).
        assert probe.received_shapes == [(0, 1)]


class TestWholeDatasetFallback:
    """Non-chunk-safe operations always run over the whole dataset."""

    def test_non_chunk_safe_ignores_chunk_size(self) -> None:
        frame = pd.DataFrame({"a": [1, 1, 2, 2, 3]})
        probe = _RecordingWholeOp()
        engine = CleaningEngine([probe])

        engine.run(frame, chunk_size=2)

        assert probe.received_shapes == [(5, 1)]

    def test_mixed_pipeline_chunks_only_safe_ops(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1, 2, 3, 4],
                "dup": ["a", "a", "b", "b"],
            }
        )
        chunk_probe = _RecordingChunkOp(column="num", delta=0)
        whole_probe = _RecordingWholeOp()
        engine = CleaningEngine([chunk_probe, whole_probe])

        engine.run(frame, chunk_size=2)

        # chunk-safe op saw two chunks of 2 rows...
        assert chunk_probe.received_shapes == [(2, 2), (2, 2)]
        # ...while the non-chunk-safe op saw the whole (chunk-processed) frame.
        assert whole_probe.received_shapes == [(4, 2)]


# ===========================================================================
# Edge cases (task 7.5): empty, single-row, all-null, mixed-type, infinite,
# wide, tall
# ===========================================================================


class TestEngineEdgeCases:
    """Execution modes on structurally unusual frames."""

    def test_empty_frame_dry_run_and_run(self) -> None:
        frame = pd.DataFrame({"num": pd.Series([], dtype="float64")})
        engine = CleaningEngine([MissingTokenOperation(), DropDuplicates()])

        dry = engine.run(frame, dry_run=True, chunk_size=3)
        real = engine.run(frame, dry_run=False, chunk_size=3)

        assert dry.empty
        assert real.empty

    def test_single_row_chunked_matches_whole(self) -> None:
        frame = pd.DataFrame({"num": [1], "token": ["n/a"]})
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=1)
        pd.testing.assert_frame_equal(whole, chunked)
        assert bool(chunked["token"].isna().iloc[0]) is True

    def test_all_null_column_chunked_matches_whole(self) -> None:
        frame = pd.DataFrame({"token": [None, None, None, None]})
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=2)
        pd.testing.assert_frame_equal(whole, chunked)

    def test_mixed_type_frame_chunked_matches_whole(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1, 2, 3, 4],
                "flt": [1.5, np.nan, 3.5, 4.5],
                "token": ["n/a", "keep", "NULL", "ok"],
            }
        )
        engine = CleaningEngine(
            [
                MissingTokenOperation(),
                TextNormalizationOperation(subset=["token"], case="lower"),
            ]
        )
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=3)
        pd.testing.assert_frame_equal(whole, chunked)

    def test_infinite_values_preserved_across_modes(self) -> None:
        frame = pd.DataFrame({"num": [np.inf, -np.inf, 1.0, np.nan]})
        engine = CleaningEngine([FillMissing(strategy="constant", value=0.0)])
        whole = engine.run(frame, chunk_size=None)
        # FillMissing is not chunk-safe, so chunk_size must not alter output.
        with_chunk = engine.run(frame, chunk_size=2)
        pd.testing.assert_frame_equal(whole, with_chunk)
        assert np.isinf(whole["num"].iloc[0])
        assert np.isinf(whole["num"].iloc[1])

    def test_wide_frame_chunked_matches_whole(self) -> None:
        data = {f"c{i}": ["n/a", "keep", "ok"] for i in range(60)}
        frame = pd.DataFrame(data)
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=2)
        pd.testing.assert_frame_equal(whole, chunked)

    def test_tall_frame_chunked_matches_whole(self) -> None:
        frame = pd.DataFrame({"token": (["n/a", "keep", "NULL", "ok"] * 500)})
        engine = CleaningEngine([MissingTokenOperation()])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=128)
        pd.testing.assert_frame_equal(whole, chunked)

    def test_messy_frame_full_pipeline_dry_run_non_mutation(self) -> None:
        frame = _messy_frame()
        snapshot = frame.copy(deep=True)
        engine = CleaningEngine(
            [
                MissingTokenOperation(),
                TextNormalizationOperation(subset=["text"], case="lower"),
                DropMissingRows(),
            ]
        )
        engine.run(frame, dry_run=True, chunk_size=2)
        pd.testing.assert_frame_equal(frame, snapshot)


# ===========================================================================
# Property tests — Task 8.2
# **Property 7: Audit entries correspond to executed operations**
# **Validates: Requirements 7.2, 7.3**
# ===========================================================================


@st.composite
def _pipeline_and_frame(
    draw: st.DrawFn,
) -> tuple[list[CleaningOperation], pd.DataFrame]:
    """
    Draw a frame plus a pipeline built from real built-in operations.

    The frame always carries a numeric column (``num``), an object token
    column (``token``), a text column (``text``) and a string-of-int column
    (``code``) so any subset of the sampled operations can execute
    meaningfully. Each drawn operation is one of the production operations
    named in the task.
    """
    size = draw(st.integers(min_value=1, max_value=30))
    tokens = st.sampled_from(["n/a", "NULL", "value", "keep", "  na ", "ok"])
    texts = st.sampled_from(["  Hi ", "world", "FOO", "  ", "b\u00a0ar", "baz"])
    frame = pd.DataFrame(
        {
            "num": draw(
                st.lists(
                    st.one_of(st.integers(min_value=-100, max_value=100), st.none()),
                    min_size=size,
                    max_size=size,
                )
            ),
            "token": draw(st.lists(tokens, min_size=size, max_size=size)),
            "text": draw(st.lists(texts, min_size=size, max_size=size)),
            "code": draw(
                st.lists(
                    st.integers(min_value=0, max_value=50).map(str),
                    min_size=size,
                    max_size=size,
                )
            ),
        }
    )

    # Fresh instances per selection so repeated picks never share state.
    factories: list[Callable[[], CleaningOperation]] = [
        MissingTokenOperation,
        lambda: TextNormalizationOperation(subset=["text"], case="lower"),
        lambda: FillMissing(strategy="constant", value=0, subset=["num"]),
        DropDuplicates,
        lambda: TypeCoercionOperation({"code": "int64"}, error_policy="coerce"),
    ]

    # Draw an ordered, non-empty selection (with possible repeats) of ops.
    n_ops = draw(st.integers(min_value=1, max_value=len(factories)))
    chosen_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(factories) - 1),
            min_size=n_ops,
            max_size=n_ops,
        )
    )
    pipeline = [factories[i]() for i in chosen_indices]
    return pipeline, frame


class TestAuditEntryCorrespondence:
    """
    **Property 7: Audit entries correspond to executed operations**
    **Validates: Requirements 7.2, 7.3**

    For any pipeline of executed operations, the audit log contains exactly
    one entry per operation, in order, with sequential 1..N ``order`` values
    whose ``operation`` name matches the operation that produced it, and the
    entire audit log is JSON-serializable.
    """

    @given(pipeline_and_frame=_pipeline_and_frame(), dry_run=st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_audit_log_matches_operations_one_to_one(
        self,
        pipeline_and_frame: tuple[list[CleaningOperation], pd.DataFrame],
        dry_run: bool,
    ) -> None:
        pipeline, frame = pipeline_and_frame
        engine = CleaningEngine(pipeline)

        result = engine.run_with_result(frame, dry_run=dry_run)

        # One audit entry per executed operation.
        assert len(result.audit_log) == len(pipeline)
        assert len(result.audit_log) == len(result.operations)

        expected_names = [op.name for op in pipeline]
        for index, entry in enumerate(result.audit_log):
            # order is sequential 1..N.
            assert entry["order"] == index + 1
            # operation name matches the op that produced it, in order.
            assert entry["operation"] == expected_names[index]
            # dry_run flag is faithfully recorded.
            assert entry["dry_run"] is dry_run

        # order values are exactly the sequence 1..N with no gaps/repeats.
        orders = [entry["order"] for entry in result.audit_log]
        assert orders == list(range(1, len(pipeline) + 1))

        # The whole audit log is JSON-serializable (Requirement 7.3).
        serialized = json.dumps(result.audit_log)
        assert isinstance(serialized, str)

    @given(pipeline_and_frame=_pipeline_and_frame())
    @settings(max_examples=60, deadline=None)
    def test_audit_entry_shape_metadata_present(
        self,
        pipeline_and_frame: tuple[list[CleaningOperation], pd.DataFrame],
    ) -> None:
        pipeline, frame = pipeline_and_frame
        engine = CleaningEngine(pipeline)

        result = engine.run_with_result(frame, dry_run=False)

        for entry in result.audit_log:
            # Every entry carries the documented, correspondence-critical keys.
            for key in (
                "order",
                "timestamp",
                "operation",
                "parameters",
                "before_shape",
                "after_shape",
            ):
                assert key in entry
            # parameters mirror the operation's describe() name.
            assert entry["parameters"]["name"] == entry["operation"]
            # before/after shapes are 2-element [rows, cols] lists.
            assert len(entry["before_shape"]) == 2
            assert len(entry["after_shape"]) == 2


# ===========================================================================
# Property tests — Task 8.3
# **Property 21: Operation impact is always explainable**
# **Validates: Requirements 7.2, 17.3**
# ===========================================================================


class TestExplainableOperationImpact:
    """
    **Property 21: Operation impact is always explainable**
    **Validates: Requirements 7.2, 17.3**

    Every executed operation produces an OperationResult whose impact is
    fully explainable: before/after shapes are present, rows/columns
    affected are non-negative integers, and describe()/parameters are
    present so the impact can always be explained.
    """

    @given(pipeline_and_frame=_pipeline_and_frame(), dry_run=st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_every_operation_result_is_explainable(
        self,
        pipeline_and_frame: tuple[list[CleaningOperation], pd.DataFrame],
        dry_run: bool,
    ) -> None:
        pipeline, frame = pipeline_and_frame
        engine = CleaningEngine(pipeline)

        result = engine.run_with_result(frame, dry_run=dry_run)

        assert len(result.operations) == len(pipeline)

        for op_result in result.operations:
            # before/after shape present as (rows, cols) with 2 dims each.
            assert len(op_result.before_shape) == 2
            assert len(op_result.after_shape) == 2

            # rows/columns affected are non-negative ints.
            assert isinstance(op_result.rows_affected, int)
            assert isinstance(op_result.columns_affected, int)
            assert op_result.rows_affected >= 0
            assert op_result.columns_affected >= 0

            # Impact is explainable: a human-readable strategy description and
            # operation-specific details (parameters) are present.
            assert isinstance(op_result.strategy_description, str)
            assert op_result.strategy_description != ""
            assert isinstance(op_result.details, dict)
            assert op_result.details.get("name") == op_result.operation_name

    @given(pipeline_and_frame=_pipeline_and_frame(), dry_run=st.booleans())
    @settings(max_examples=80, deadline=None)
    def test_audit_parameters_explain_impact(
        self,
        pipeline_and_frame: tuple[list[CleaningOperation], pd.DataFrame],
        dry_run: bool,
    ) -> None:
        pipeline, frame = pipeline_and_frame
        engine = CleaningEngine(pipeline)

        result = engine.run_with_result(frame, dry_run=dry_run)

        for entry in result.audit_log:
            # parameters (describe() output) present and non-empty so the
            # operation's configuration is always explainable.
            assert isinstance(entry["parameters"], dict)
            assert entry["parameters"]
            # affected counts recorded in the audit entry are non-negative.
            assert entry["rows_affected"] >= 0
            assert entry["columns_affected"] >= 0


# ===========================================================================
# Property tests — Task 18.2
# **Property 17: Custom operations behave like built-ins**
# **Validates: Requirements 14.1, 14.4**
# ===========================================================================


class _CustomAddConstant(CleaningOperation):
    """
    A user-defined cleaning operation, defined outside the library.

    It subclasses the existing :class:`CleaningOperation` ABC only, adds a
    constant to a numeric column, and relies entirely on the inherited
    ``apply_with_result``. It carries no library-internal wiring: registering
    it in an :class:`OperationRegistry` and running it through
    :class:`CleaningEngine` must work without any engine modification
    (Requirement 14.1) and must produce audit logging and results identical
    in shape to a built-in (Requirement 14.4).
    """

    name = "custom_add_constant"
    is_chunk_safe = True
    is_inplace_safe = False

    def __init__(self, column: str = "num", delta: int = 1) -> None:
        self.column = column
        self.delta = delta

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)
        result = dataframe.copy()
        result[self.column] = result[self.column] + self.delta
        return result

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "column": self.column, "delta": self.delta}


class TestCustomOperationsBehaveLikeBuiltins:
    """
    **Property 17: Custom operations behave like built-ins**
    **Validates: Requirements 14.1, 14.4**

    A custom ``CleaningOperation`` subclass registers through a fresh
    ``OperationRegistry`` and runs through ``CleaningEngine`` producing an
    ``OperationResult`` plus an audit entry indistinguishable in shape from a
    built-in, and behaves identically to built-ins for engine concerns
    (dry-run non-mutation, audit ordering, and ``describe()`` appearing in
    the audit ``parameters``).
    """

    # The exact set of audit-entry keys the engine records for a built-in,
    # captured from a built-in run so the assertion can never drift.
    def _builtin_audit_keys(self) -> set[str]:
        from sanitizepy.cleaning.registry import OperationRegistry

        # Confirm a built-in is retrievable from a registry the same way a
        # custom op will be, then run it to capture the audit-entry shape.
        registry = OperationRegistry()
        registry.register(DropDuplicates.name, DropDuplicates)
        built_cls = registry.get(DropDuplicates.name)
        engine = CleaningEngine([built_cls()])
        entry = engine.run_with_result(pd.DataFrame({"x": [1, 1, 2]})).audit_log[0]
        return set(entry.keys())

    def test_custom_op_registers_and_runs_without_engine_change(self) -> None:
        from sanitizepy.cleaning.registry import OperationRegistry

        # Register the custom op in a fresh, local registry (Requirement
        # 14.1 / 14.3): a name -> class lookup, same as built-ins.
        registry = OperationRegistry()
        registry.register(_CustomAddConstant.name, _CustomAddConstant)
        assert registry.get(_CustomAddConstant.name) is _CustomAddConstant

        # Reconstruct from the registry (name -> class) exactly like a plan
        # would, and run it through the unmodified engine.
        op_cls = registry.get(_CustomAddConstant.name)
        engine = CleaningEngine([op_cls(column="num", delta=10)])
        frame = pd.DataFrame({"num": [1, 2, 3]})

        result = engine.run_with_result(frame, dry_run=False)

        # Produces exactly one OperationResult, like any built-in.
        assert len(result.operations) == 1
        op_result = result.operations[0]
        assert isinstance(op_result, OperationResult)
        assert op_result.operation_name == _CustomAddConstant.name
        # The transform actually ran.
        pd.testing.assert_frame_equal(result.data, pd.DataFrame({"num": [11, 12, 13]}))

    @given(
        values=st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=1,
            max_size=40,
        ),
        delta=st.integers(min_value=-100, max_value=100),
        dry_run=st.booleans(),
    )
    @settings(max_examples=80, deadline=None)
    def test_custom_op_result_and_audit_shape_match_builtin(
        self, values: list[int], delta: int, dry_run: bool
    ) -> None:
        from sanitizepy.cleaning.registry import OperationRegistry

        registry = OperationRegistry()
        registry.register(_CustomAddConstant.name, _CustomAddConstant)
        op_cls = registry.get(_CustomAddConstant.name)

        frame = pd.DataFrame({"num": values})
        engine = CleaningEngine([op_cls(column="num", delta=delta)])

        result = engine.run_with_result(frame, dry_run=dry_run)

        # Exactly one audit entry per executed operation, like a built-in.
        assert len(result.audit_log) == len(result.operations) == 1
        entry = result.audit_log[0]

        # Audit-entry keys are indistinguishable from a built-in's.
        assert set(entry.keys()) == self._builtin_audit_keys()

        # Correspondence: order, operation name, and dry_run flag recorded.
        assert entry["order"] == 1
        assert entry["operation"] == _CustomAddConstant.name
        assert entry["dry_run"] is dry_run

        # describe() surfaces in the audit parameters exactly like a built-in.
        assert entry["parameters"]["name"] == _CustomAddConstant.name
        assert entry["parameters"]["column"] == "num"
        assert entry["parameters"]["delta"] == delta

        # OperationResult shape parity: 2-tuple shapes, non-negative counts,
        # non-empty strategy description, dict details carrying the name.
        op_result = result.operations[0]
        assert len(op_result.before_shape) == 2
        assert len(op_result.after_shape) == 2
        assert op_result.rows_affected >= 0
        assert op_result.columns_affected >= 0
        assert isinstance(op_result.strategy_description, str)
        assert op_result.strategy_description != ""
        assert op_result.details["name"] == _CustomAddConstant.name

        # The audit log is JSON-serializable, like any built-in run.
        assert isinstance(json.dumps(result.audit_log), str)

    @given(
        values=st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=1,
            max_size=40,
        ),
        delta=st.integers(min_value=-100, max_value=100),
    )
    @settings(max_examples=80, deadline=None)
    def test_custom_op_dry_run_never_mutates_caller(
        self, values: list[int], delta: int
    ) -> None:
        frame = pd.DataFrame({"num": values})
        snapshot = frame.copy(deep=True)

        engine = CleaningEngine([_CustomAddConstant(column="num", delta=delta)])
        result = engine.run_with_result(frame, dry_run=True)

        # Dry-run leaves the caller's frame untouched, exactly like a
        # built-in, and returns the unmodified frame.
        pd.testing.assert_frame_equal(frame, snapshot)
        pd.testing.assert_frame_equal(result.data, snapshot)

    @given(
        values=st.lists(
            st.integers(min_value=-1000, max_value=1000),
            min_size=1,
            max_size=40,
        ),
        delta=st.integers(min_value=-100, max_value=100),
        chunk_size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=60, deadline=None)
    def test_custom_chunk_safe_op_chunked_matches_whole(
        self, values: list[int], delta: int, chunk_size: int
    ) -> None:
        # A chunk-safe custom op participates in chunked execution exactly
        # like a chunk-safe built-in: chunked output equals whole-dataset.
        frame = pd.DataFrame({"num": values})
        engine = CleaningEngine([_CustomAddConstant(column="num", delta=delta)])
        whole = engine.run(frame, chunk_size=None)
        chunked = engine.run(frame, chunk_size=chunk_size)
        pd.testing.assert_frame_equal(whole, chunked)
