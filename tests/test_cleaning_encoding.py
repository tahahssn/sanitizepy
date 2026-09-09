"""
Tests for sanitizepy.cleaning.encoding.EncodingRepairOperation

Task 6.3 – unit tests for encoding detection and repair
Requirements: 5.2, 5.3, 5.4

Covers:
  - Mojibake indicators detected and handled (Req 5.1, 5.2)
  - Replacement character (U+FFFD) repaired (Req 5.2)
  - Control characters removed (Req 5.2)
  - Successful repair records count (Req 5.4)
  - EncodingError raised when artifact cannot be safely repaired (Req 5.3)
  - DependencyError raised when ftfy is absent but mode="advanced" (Req 5.3, 16.3)
  - describe() exposes name, subset, mode, error_on_unrepaired
  - is_chunk_safe flag
  - Registry presence
  - Non-string columns untouched
  - Edge cases: empty DF, single-row, all-null, mixed-type, infinite numeric, wide, tall
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sanitizepy.cleaning.encoding import EncodingRepairOperation
from sanitizepy.exceptions import DependencyError, EncodingError

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

# A classic mojibake string: UTF-8 bytes mis-decoded as Latin-1.
MOJIBAKE_STRING: str = "CafÃ©"  # intended: "Café"

# A string containing the Unicode replacement character U+FFFD.
REPLACEMENT_CHAR_STRING: str = "hello\ufffdworld"

# A pure replacement-char string (unrecoverable without context)
PURE_REPLACEMENT_STRING: str = "\ufffd\ufffd\ufffd"

# A string containing ASCII control characters.
CONTROL_CHAR_STRING: str = "hello\x01world\x02"

# A completely clean string that should not be modified.
CLEAN_STRING: str = "This is a normal sentence."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**kwargs: object) -> pd.DataFrame:
    return pd.DataFrame(kwargs)


# ---------------------------------------------------------------------------
# 1. Replacement character (U+FFFD) repair (Requirement 5.2)
# ---------------------------------------------------------------------------


class TestReplacementCharRepair:
    """EncodingRepairOperation must remove/replace U+FFFD."""

    def test_embedded_replacement_char_removed(self) -> None:
        """U+FFFD embedded among valid text is stripped in core mode."""
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        # The replacement char should be stripped from the repaired cell
        assert "\ufffd" not in str(result["col"].iloc[0])
        assert result["col"].iloc[1] == CLEAN_STRING

    def test_pure_replacement_char_becomes_nan(self) -> None:
        """A cell consisting entirely of U+FFFD is unrecoverable → NaN."""
        df = _make_df(col=[PURE_REPLACEMENT_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])
        assert result["col"].iloc[1] == CLEAN_STRING

    def test_replacement_char_repair_count_in_result(self) -> None:
        """apply_with_result records values_repaired for replacement chars."""
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] >= 1

    def test_partial_replacement_char_stripped_not_nan(self) -> None:
        """Partial replacement char mixed with text → strip char, keep text."""
        df = _make_df(col=["hello\ufffdworld"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        cell = result["col"].iloc[0]
        assert isinstance(cell, str)
        assert "\ufffd" not in cell
        assert "hello" in cell
        assert "world" in cell


# ---------------------------------------------------------------------------
# 2. Control character removal (Requirement 5.2)
# ---------------------------------------------------------------------------


class TestControlCharRemoval:
    """EncodingRepairOperation must remove non-printable ASCII control chars."""

    def test_control_chars_removed(self) -> None:
        df = _make_df(col=[CONTROL_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        cell = result["col"].iloc[0]
        # Control chars \x01 and \x02 should be gone
        assert "\x01" not in str(cell)
        assert "\x02" not in str(cell)
        # Remaining text should be intact
        assert "hello" in str(cell)
        assert "world" in str(cell)

    def test_null_byte_removed(self) -> None:
        df = _make_df(col=["hello\x00world"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert "\x00" not in str(result["col"].iloc[0])

    def test_common_whitespace_preserved(self) -> None:
        """HT (\\x09), LF (\\x0a), CR (\\x0d) should be preserved."""
        df = _make_df(col=["hello\tworld\nnewline"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        cell = result["col"].iloc[0]
        assert isinstance(cell, str)
        assert "\t" in cell
        assert "\n" in cell

    def test_del_char_removed(self) -> None:
        """U+007F DELETE should be removed."""
        df = _make_df(col=["abc\x7fdef"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert "\x7f" not in str(result["col"].iloc[0])

    def test_control_char_only_cell_becomes_nan(self) -> None:
        """A cell of only control chars collapses to empty string → NaN."""
        df = _make_df(col=["\x01\x02\x03"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_control_char_repair_count_in_result(self) -> None:
        df = _make_df(col=[CONTROL_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] >= 1


# ---------------------------------------------------------------------------
# 3. Mojibake indicators (Requirement 5.1, 5.2)
# ---------------------------------------------------------------------------


class TestMojibakenIndicators:
    """EncodingRepairOperation must handle mojibake values."""

    def test_mojibake_string_is_an_artifact(self) -> None:
        """Confirm the mojibake string is detected as an artifact."""
        from sanitizepy.cleaning.encoding import _has_artifact

        assert _has_artifact(MOJIBAKE_STRING)

    def test_core_mode_leaves_mojibake_in_place_without_error_when_lenient(
        self,
    ) -> None:
        """Core mode cannot repair mojibake — lenient mode leaves it as-is."""
        df = _make_df(col=[MOJIBAKE_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        # Mojibake string is unchanged in core mode (no advanced repair)
        assert result["col"].iloc[0] == MOJIBAKE_STRING
        assert result["col"].iloc[1] == CLEAN_STRING

    def test_core_mode_raises_encoding_error_for_mojibake_strict(self) -> None:
        """Core mode + error_on_unrepaired=True raises EncodingError for mojibake."""
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=True)
        with pytest.raises(EncodingError):
            op.apply(df)

    def test_result_records_unrepaired_count_for_mojibake(self) -> None:
        """In lenient core mode, unrepaired mojibake cells are counted."""
        df = _make_df(col=[MOJIBAKE_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        # Mojibake cannot be repaired in core mode → it's unrepaired
        assert res.details["unrepaired_count"] >= 1


# ---------------------------------------------------------------------------
# 4. EncodingError on unrepaired artifact (Requirement 5.3)
# ---------------------------------------------------------------------------


class TestEncodingError:
    """EncodingRepairOperation raises EncodingError for irrecoverable artifacts."""

    def test_mojibake_raises_encoding_error_in_strict_mode(self) -> None:
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=True)
        with pytest.raises(EncodingError):
            op.apply(df)

    def test_encoding_error_message_identifies_column(self) -> None:
        df = _make_df(special_col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(
            subset=["special_col"], mode="core", error_on_unrepaired=True
        )
        with pytest.raises(EncodingError, match="special_col"):
            op.apply(df)

    def test_encoding_error_not_raised_in_lenient_mode(self) -> None:
        """error_on_unrepaired=False must NOT raise even for mojibake."""
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=False)
        # Should not raise
        result = op.apply(df)
        assert result["col"].iloc[0] == MOJIBAKE_STRING  # left unchanged

    def test_apply_with_result_raises_encoding_error_in_strict_mode(self) -> None:
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=True)
        with pytest.raises(EncodingError):
            op.apply_with_result(df)

    def test_clean_data_never_raises_encoding_error(self) -> None:
        """A fully clean DataFrame must not trigger EncodingError."""
        df = _make_df(col=[CLEAN_STRING, "Another clean string"])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=True)
        result = op.apply(df)
        # No exception raised
        assert list(result["col"]) == [CLEAN_STRING, "Another clean string"]


# ---------------------------------------------------------------------------
# 5. DependencyError for missing ftfy (Requirement 16.3, 5.3)
# ---------------------------------------------------------------------------


class TestDependencyError:
    """Raise DependencyError naming sanitizepy[text] when ftfy is absent."""

    def test_advanced_mode_raises_dependency_error_when_ftfy_missing(
        self,
    ) -> None:
        """Simulate missing ftfy via sys.modules monkeypatching."""
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="advanced")

        with (
            patch.dict(sys.modules, {"ftfy": None}),
            pytest.raises(DependencyError, match="sanitizepy\\[text\\]"),
        ):
            op.apply(df)

    def test_dependency_error_names_extra(self) -> None:
        """The DependencyError message must mention sanitizepy[text]."""
        df = _make_df(col=["test"])
        op = EncodingRepairOperation(mode="advanced")

        with (
            patch.dict(sys.modules, {"ftfy": None}),
            pytest.raises(DependencyError) as exc_info,
        ):
            op.apply(df)
        assert "sanitizepy[text]" in str(exc_info.value)

    def test_core_mode_never_raises_dependency_error(self) -> None:
        """core mode never needs ftfy, so DependencyError must not be raised."""
        df = _make_df(col=[REPLACEMENT_CHAR_STRING])
        op = EncodingRepairOperation(mode="core", error_on_unrepaired=False)
        # Must not raise DependencyError
        result = op.apply(df)
        assert result is not None

    def test_advanced_mode_with_ftfy_present_does_not_raise_dependency_error(
        self,
    ) -> None:
        """When ftfy IS importable, advanced mode must not raise DependencyError."""
        ftfy_mock = types.ModuleType("ftfy")
        ftfy_mock.fix_text = lambda text, **_kwargs: text  # type: ignore[attr-defined]

        df = _make_df(col=[CLEAN_STRING])
        op = EncodingRepairOperation(mode="advanced", error_on_unrepaired=False)

        with patch.dict(sys.modules, {"ftfy": ftfy_mock}):
            # Should not raise DependencyError
            result = op.apply(df)
        assert result is not None

    def test_apply_with_result_raises_dependency_error_advanced_mode(self) -> None:
        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="advanced")

        with (
            patch.dict(sys.modules, {"ftfy": None}),
            pytest.raises(DependencyError, match="sanitizepy\\[text\\]"),
        ):
            op.apply_with_result(df)


# ---------------------------------------------------------------------------
# 6. Successful repair and OperationResult (Requirement 5.4)
# ---------------------------------------------------------------------------


class TestSuccessfulRepair:
    """EncodingRepairOperation records count of repaired values."""

    def test_repair_success_count_replacement_char(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 1

    def test_repair_success_count_control_chars(self) -> None:
        df = _make_df(col=[CONTROL_CHAR_STRING, CONTROL_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 2

    def test_no_repair_needed_count_is_zero(self) -> None:
        df = _make_df(col=[CLEAN_STRING, "Another clean line"])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 0

    def test_before_after_shape_unchanged(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.before_shape == (2, 1)
        assert res.after_shape == (2, 1)

    def test_operation_name_in_result(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.operation_name == "encoding_repair"

    def test_affected_columns_listed(self) -> None:
        df = _make_df(a=[REPLACEMENT_CHAR_STRING], b=[CONTROL_CHAR_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert set(res.affected_columns) == {"a", "b"}

    def test_dry_run_does_not_mutate_input(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING])
        original = df.copy()
        result_df, res = EncodingRepairOperation(
            error_on_unrepaired=False
        ).apply_with_result(df, dry_run=True)
        pd.testing.assert_frame_equal(df, original)
        assert res.dry_run is True

    def test_dry_run_false_returns_repaired_data(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING])
        result_df, res = EncodingRepairOperation(
            error_on_unrepaired=False
        ).apply_with_result(df, dry_run=False)
        assert "\ufffd" not in str(result_df["col"].iloc[0])
        assert res.dry_run is False

    def test_advanced_mode_repairs_with_ftfy(self) -> None:
        """When ftfy is present, advanced mode calls fix_text."""
        fixed_value = "Café"
        ftfy_mock = types.ModuleType("ftfy")
        ftfy_mock.fix_text = lambda text, **_kwargs: fixed_value  # type: ignore[attr-defined]

        df = _make_df(col=[MOJIBAKE_STRING])
        op = EncodingRepairOperation(mode="advanced", error_on_unrepaired=False)

        with patch.dict(sys.modules, {"ftfy": ftfy_mock}):
            result = op.apply(df)
        assert result["col"].iloc[0] == fixed_value


# ---------------------------------------------------------------------------
# 7. Non-string columns untouched
# ---------------------------------------------------------------------------


class TestNonStringColumnsUntouched:
    def test_integer_column_not_modified(self) -> None:
        df = _make_df(
            a=[REPLACEMENT_CHAR_STRING, CLEAN_STRING],
            b=[1, 2],
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert list(result["b"]) == [1, 2]

    def test_float_column_not_modified(self) -> None:
        df = _make_df(
            a=[CONTROL_CHAR_STRING, CLEAN_STRING],
            b=[1.5, 2.5],
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert list(result["b"]) == pytest.approx([1.5, 2.5])

    def test_boolean_column_not_modified(self) -> None:
        df = pd.DataFrame(
            {
                "a": [REPLACEMENT_CHAR_STRING],
                "flag": pd.array([True], dtype=bool),
            }
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert list(result["flag"]) == [True]

    def test_datetime_column_not_modified(self) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-06-15"])
        df = pd.DataFrame(
            {"d": dates, "label": [REPLACEMENT_CHAR_STRING, CLEAN_STRING]}
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        pd.testing.assert_series_equal(result["d"], df["d"])


# ---------------------------------------------------------------------------
# 8. subset parameter
# ---------------------------------------------------------------------------


class TestSubsetParameter:
    def test_subset_restricts_columns(self) -> None:
        df = _make_df(
            a=[REPLACEMENT_CHAR_STRING, CLEAN_STRING],
            b=[CONTROL_CHAR_STRING, CLEAN_STRING],
        )
        op = EncodingRepairOperation(subset=["a"], error_on_unrepaired=False)
        result = op.apply(df)
        # Column a should be repaired
        assert "\ufffd" not in str(result["a"].iloc[0])
        # Column b should be unchanged
        assert result["b"].iloc[0] == CONTROL_CHAR_STRING

    def test_subset_invalid_column_raises_key_error(self) -> None:
        df = _make_df(a=[CLEAN_STRING])
        op = EncodingRepairOperation(subset=["nonexistent"], error_on_unrepaired=False)
        with pytest.raises(KeyError):
            op.apply(df)

    def test_subset_affects_count(self) -> None:
        df = _make_df(
            a=[REPLACEMENT_CHAR_STRING],
            b=[REPLACEMENT_CHAR_STRING],
        )
        op = EncodingRepairOperation(subset=["a"], error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 1  # only column a processed


# ---------------------------------------------------------------------------
# 9. describe() output
# ---------------------------------------------------------------------------


class TestDescribe:
    def test_describe_contains_required_keys(self) -> None:
        op = EncodingRepairOperation(
            subset=["col"], mode="core", error_on_unrepaired=True
        )
        desc = op.describe()
        assert desc["name"] == "encoding_repair"
        assert desc["subset"] == ["col"]
        assert desc["mode"] == "core"
        assert desc["error_on_unrepaired"] is True

    def test_describe_subset_none_when_not_set(self) -> None:
        desc = EncodingRepairOperation().describe()
        assert desc["subset"] is None

    def test_describe_advanced_mode(self) -> None:
        op = EncodingRepairOperation(mode="advanced")
        desc = op.describe()
        assert desc["mode"] == "advanced"

    def test_describe_lenient_mode(self) -> None:
        op = EncodingRepairOperation(error_on_unrepaired=False)
        desc = op.describe()
        assert desc["error_on_unrepaired"] is False


# ---------------------------------------------------------------------------
# 10. Classification attributes
# ---------------------------------------------------------------------------


class TestClassificationAttributes:
    def test_is_chunk_safe(self) -> None:
        assert EncodingRepairOperation.is_chunk_safe is True

    def test_is_inplace_safe_default(self) -> None:
        assert EncodingRepairOperation.is_inplace_safe is False

    def test_name_attribute(self) -> None:
        assert EncodingRepairOperation.name == "encoding_repair"


# ---------------------------------------------------------------------------
# 11. Registry presence
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_in_module_registry(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.contains("encoding_repair")

    def test_registry_returns_correct_class(self) -> None:
        from sanitizepy.cleaning.registry import registry

        assert registry.get("encoding_repair") is EncodingRepairOperation


# ---------------------------------------------------------------------------
# 12. Non-mutation guarantee
# ---------------------------------------------------------------------------


class TestNonMutation:
    def test_apply_does_not_mutate_input(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CONTROL_CHAR_STRING, CLEAN_STRING])
        original = df.copy()
        EncodingRepairOperation(error_on_unrepaired=False).apply(df)
        pd.testing.assert_frame_equal(df, original)

    def test_apply_with_result_does_not_mutate_input(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING, CLEAN_STRING])
        original = df.copy()
        EncodingRepairOperation(error_on_unrepaired=False).apply_with_result(
            df, dry_run=False
        )
        pd.testing.assert_frame_equal(df, original)


# ---------------------------------------------------------------------------
# 13. Already-clean values unchanged
# ---------------------------------------------------------------------------


class TestAlreadyCleanValues:
    def test_clean_string_unchanged(self) -> None:
        df = _make_df(col=[CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result["col"].iloc[0] == CLEAN_STRING

    def test_existing_nan_unchanged(self) -> None:
        df = _make_df(col=[np.nan, CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])

    def test_none_value_unchanged(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, CLEAN_STRING], dtype="object")})
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert pd.isna(result["col"].iloc[0])


# ---------------------------------------------------------------------------
# 14. Edge case: Empty DataFrame
# ---------------------------------------------------------------------------


class TestEdgeCaseEmpty:
    def test_apply_empty_dataframe_returns_empty(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result.shape == (0, 1)

    def test_apply_with_result_empty_zero_repaired(self) -> None:
        df = pd.DataFrame({"col": pd.Series([], dtype="object")})
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.before_shape == (0, 1)
        assert res.after_shape == (0, 1)
        assert res.details["values_repaired"] == 0

    def test_apply_completely_empty_dataframe(self) -> None:
        df = pd.DataFrame()
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result.empty


# ---------------------------------------------------------------------------
# 15. Edge case: Single row
# ---------------------------------------------------------------------------


class TestEdgeCaseSingleRow:
    def test_single_row_replacement_char_repaired(self) -> None:
        df = _make_df(col=[REPLACEMENT_CHAR_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert "\ufffd" not in str(result["col"].iloc[0])

    def test_single_row_clean_unchanged(self) -> None:
        df = _make_df(col=[CLEAN_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result["col"].iloc[0] == CLEAN_STRING

    def test_single_row_apply_with_result_count(self) -> None:
        df = _make_df(col=[CONTROL_CHAR_STRING])
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 1


# ---------------------------------------------------------------------------
# 16. Edge case: All-null column
# ---------------------------------------------------------------------------


class TestEdgeCaseAllNull:
    def test_all_null_column_unchanged(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None, None], dtype="object")})
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result["col"].isna().all()

    def test_all_null_apply_with_result_zero_repaired(self) -> None:
        df = pd.DataFrame({"col": pd.array([None, None], dtype="object")})
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 0


# ---------------------------------------------------------------------------
# 17. Edge case: Mixed-type column (object dtype with various Python types)
# ---------------------------------------------------------------------------


class TestEdgeCaseMixedType:
    def test_integer_objects_in_object_column_not_modified(self) -> None:
        df = pd.DataFrame(
            {"col": pd.array([1, REPLACEMENT_CHAR_STRING, 3.0, None], dtype="object")}
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        assert result["col"].iloc[0] == 1
        assert result["col"].iloc[2] == 3.0
        assert pd.isna(result["col"].iloc[3])

    def test_mixed_type_count_only_counts_string_artifacts(self) -> None:
        df = pd.DataFrame(
            {
                "col": pd.array(
                    [1, REPLACEMENT_CHAR_STRING, CONTROL_CHAR_STRING, 3.0],
                    dtype="object",
                )
            }
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        # Two string artifacts repaired; int and float not counted
        assert res.details["values_repaired"] == 2


# ---------------------------------------------------------------------------
# 18. Edge case: Infinite numeric values (won't appear in string cols, but
#     validate that mixed-numeric + string DFs work fine)
# ---------------------------------------------------------------------------


class TestEdgeCaseInfiniteNumeric:
    def test_dataframe_with_inf_in_numeric_col(self) -> None:
        df = pd.DataFrame(
            {
                "text": [REPLACEMENT_CHAR_STRING, CLEAN_STRING],
                "value": [float("inf"), float("-inf")],
            }
        )
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        # Numeric col unchanged
        assert result["value"].iloc[0] == float("inf")
        assert result["value"].iloc[1] == float("-inf")
        # Text col repaired
        assert "\ufffd" not in str(result["text"].iloc[0])


# ---------------------------------------------------------------------------
# 19. Edge case: Wide DataFrame (many columns)
# ---------------------------------------------------------------------------


class TestEdgeCaseWide:
    def test_wide_all_string_columns_processed(self) -> None:
        n_cols = 20
        data = {
            f"col{i}": [REPLACEMENT_CHAR_STRING, CLEAN_STRING] for i in range(n_cols)
        }
        df = pd.DataFrame(data)
        op = EncodingRepairOperation(error_on_unrepaired=False)
        result = op.apply(df)
        for col in df.columns:
            assert "\ufffd" not in str(result[col].iloc[0])
            assert result[col].iloc[1] == CLEAN_STRING

    def test_wide_apply_with_result_correct_count(self) -> None:
        n_cols = 10
        data = {
            f"c{i}": [REPLACEMENT_CHAR_STRING, CONTROL_CHAR_STRING, CLEAN_STRING]
            for i in range(n_cols)
        }
        df = pd.DataFrame(data)
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        # 2 artifacts per column × 10 columns
        assert res.details["values_repaired"] == 2 * n_cols


# ---------------------------------------------------------------------------
# 20. Edge case: Tall DataFrame (many rows)
# ---------------------------------------------------------------------------


class TestEdgeCaseTall:
    def test_tall_all_repaired(self) -> None:
        n_rows = 5_000
        col_values = [
            REPLACEMENT_CHAR_STRING if i % 2 == 0 else CLEAN_STRING
            for i in range(n_rows)
        ]
        df = _make_df(col=col_values)
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == n_rows // 2

    def test_tall_no_artifacts_zero_repaired(self) -> None:
        n_rows = 3_000
        df = _make_df(col=[CLEAN_STRING] * n_rows)
        op = EncodingRepairOperation(error_on_unrepaired=False)
        _, res = op.apply_with_result(df)
        assert res.details["values_repaired"] == 0


# ---------------------------------------------------------------------------
# 21. apply() raises TypeError for non-DataFrame input
# ---------------------------------------------------------------------------


class TestTypeError:
    def test_apply_raises_on_non_dataframe(self) -> None:
        op = EncodingRepairOperation(error_on_unrepaired=False)
        with pytest.raises(TypeError):
            op.apply([1, 2, 3])  # type: ignore[arg-type]

    def test_apply_with_result_raises_on_non_dataframe(self) -> None:
        op = EncodingRepairOperation(error_on_unrepaired=False)
        with pytest.raises(TypeError):
            op.apply_with_result([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 22. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            EncodingRepairOperation(mode="invalid")  # type: ignore[arg-type]
