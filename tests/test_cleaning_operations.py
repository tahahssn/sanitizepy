"""
Tests for cleaner.cleaning.operations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaner.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)


def _make_df(**kwargs):
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# DropMissingRows
# ---------------------------------------------------------------------------

class TestDropMissingRows:
    def test_drops_rows_with_any_missing(self):
        df = _make_df(a=[1, None, 3], b=["x", "y", None])
        op = DropMissingRows()
        result = op.apply(df)
        assert len(result) == 1
        assert result["a"].iloc[0] == 1

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3])
        _ = DropMissingRows().apply(df)
        assert df["a"].isna().sum() == 1

    def test_subset_only_drops_based_on_subset(self):
        df = _make_df(a=[1, None, 3], b=[None, None, None])
        op = DropMissingRows(subset=["a"])
        result = op.apply(df)
        assert len(result) == 2

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropMissingRows(subset=["nonexistent"]).apply(df)

    def test_describe_returns_correct_keys(self):
        op = DropMissingRows(subset=["a"])
        desc = op.describe()
        assert desc["name"] == "drop_missing_rows"
        assert desc["subset"] == ["a"]

    def test_describe_subset_none(self):
        op = DropMissingRows()
        assert op.describe()["subset"] is None

    def test_raises_on_non_dataframe(self):
        with pytest.raises(TypeError):
            DropMissingRows().apply([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DropMissingColumns
# ---------------------------------------------------------------------------

class TestDropMissingColumns:
    def test_drops_columns_with_missing_values(self):
        df = _make_df(a=[1, None, 3], b=[1, 2, 3])
        op = DropMissingColumns()
        result = op.apply(df)
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3], b=[1, 2, 3])
        _ = DropMissingColumns().apply(df)
        assert "a" in df.columns

    def test_subset_drops_specified_columns(self):
        df = _make_df(a=[1, 2, 3], b=[4, 5, 6])
        op = DropMissingColumns(subset=["a"])
        result = op.apply(df)
        assert "a" not in result.columns

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropMissingColumns(subset=["nonexistent"]).apply(df)

    def test_describe_name(self):
        assert DropMissingColumns().describe()["name"] == "drop_missing_columns"


# ---------------------------------------------------------------------------
# FillMissing
# ---------------------------------------------------------------------------

class TestFillMissing:
    def test_fills_all_missing_with_scalar(self):
        df = _make_df(a=[1, None, 3], b=[None, "y", None])
        result = FillMissing(value=0).apply(df)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0

    def test_fill_string_value(self):
        df = _make_df(name=["Alice", None, "Bob"])
        result = FillMissing(value="UNKNOWN").apply(df)
        assert result["name"].iloc[1] == "UNKNOWN"

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, None, 3])
        _ = FillMissing(value=99).apply(df)
        assert df["a"].isna().sum() == 1

    def test_subset_fills_only_subset(self):
        df = _make_df(a=[1, None, 3], b=[None, None, None])
        result = FillMissing(value=0, subset=["a"]).apply(df)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 3

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            FillMissing(value=0, subset=["nonexistent"]).apply(df)

    def test_describe_includes_value_and_subset(self):
        op = FillMissing(value=42, subset=["a"])
        desc = op.describe()
        assert desc["value"] == 42
        assert desc["subset"] == ["a"]


# ---------------------------------------------------------------------------
# DropDuplicates
# ---------------------------------------------------------------------------

class TestDropDuplicates:
    def test_drops_duplicate_rows(self):
        df = _make_df(a=[1, 1, 2], b=["x", "x", "y"])
        result = DropDuplicates().apply(df)
        assert len(result) == 2

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, 1, 2])
        _ = DropDuplicates().apply(df)
        assert len(df) == 3

    def test_keep_first_default(self):
        df = _make_df(a=[1, 1, 2], b=["x", "x", "y"])
        result = DropDuplicates(keep="first").apply(df)
        assert result.index.tolist()[0] == 0

    def test_keep_last(self):
        df = _make_df(a=[1, 1, 2])
        result = DropDuplicates(keep="last").apply(df)
        assert 1 in result.index.tolist()
        assert 0 not in result.index.tolist()

    def test_keep_false_drops_all(self):
        df = _make_df(a=[1, 1, 2])
        result = DropDuplicates(keep=False).apply(df)
        assert len(result) == 1  # only unique row 2

    def test_invalid_keep_raises_value_error(self):
        with pytest.raises(ValueError):
            DropDuplicates(keep="invalid")  # type: ignore[arg-type]

    def test_subset_duplicates(self):
        df = _make_df(a=[1, 1, 2], b=["x", "y", "z"])
        result = DropDuplicates(subset=["a"]).apply(df)
        assert len(result) == 2

    def test_subset_invalid_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropDuplicates(subset=["nonexistent"]).apply(df)

    def test_describe_keys(self):
        op = DropDuplicates(keep="last")
        desc = op.describe()
        assert desc["keep"] == "last"
        assert desc["name"] == "drop_duplicates"


# ---------------------------------------------------------------------------
# DropColumns
# ---------------------------------------------------------------------------

class TestDropColumns:
    def test_drops_specified_columns(self):
        df = _make_df(a=[1, 2], b=[3, 4], c=[5, 6])
        result = DropColumns(columns=["a", "b"]).apply(df)
        assert "a" not in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_does_not_mutate_input(self):
        df = _make_df(a=[1, 2], b=[3, 4])
        _ = DropColumns(columns=["a"]).apply(df)
        assert "a" in df.columns

    def test_missing_column_raises_key_error(self):
        df = _make_df(a=[1, 2])
        with pytest.raises(KeyError):
            DropColumns(columns=["nonexistent"]).apply(df)

    def test_empty_columns_list_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one"):
            DropColumns(columns=[])

    def test_describe_lists_columns(self):
        op = DropColumns(columns=["x", "y"])
        desc = op.describe()
        assert desc["columns"] == ["x", "y"]
