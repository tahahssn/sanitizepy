"""
sanitizepy.cleaning.type_coercion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Safe, deterministic type coercion for named columns.

:class:`TypeCoercionOperation` converts one or more named columns to
caller-specified target dtypes. Conversion never silently discards data:

* Under the strict ``"raise"`` policy the operation raises
  :class:`~sanitizepy.exceptions.DataTypeConversionError` (or
  :class:`~sanitizepy.exceptions.DataValidationError` for malformed inputs)
  identifying the column, the target dtype, and the first non-convertible
  value.
* Under the ``"coerce"`` policy non-convertible values become the pandas
  missing marker and the number of affected values is recorded in the
  resulting :class:`~sanitizepy.cleaning.base.OperationResult`.

The per-cell conversion is independent of how rows are partitioned, so the
operation is chunk-safe.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from ..exceptions import DataTypeConversionError, DataValidationError
from .base import CleaningOperation, OperationResult

ErrorPolicy = Literal["raise", "coerce"]


class TypeCoercionOperation(CleaningOperation):
    """
    Convert named columns to target dtypes deterministically.

    Parameters
    ----------
    target_dtypes:
        Mapping of column name to the target dtype string (for example
        ``{"age": "int64", "signup": "datetime64[ns]"}``). Every listed
        column must exist in the dataframe at ``apply`` time.
    error_policy:
        ``"raise"`` (default) fails on the first non-convertible value.
        ``"coerce"`` replaces non-convertible values with the missing
        marker and records the affected count.

    Notes
    -----
    The operation exposes ``columns``, ``target_dtypes`` and
    ``error_policy`` through :meth:`describe`. Conversion is applied
    per cell and is therefore chunk-safe.
    """

    name = "type_coercion"
    is_chunk_safe = True

    def __init__(
        self,
        target_dtypes: dict[str, str],
        error_policy: ErrorPolicy = "raise",
    ) -> None:
        if not target_dtypes:
            raise ValueError("target_dtypes must contain at least one column mapping")

        if error_policy not in ("raise", "coerce"):
            raise ValueError("error_policy must be one of: 'raise', 'coerce'")

        self.target_dtypes: dict[str, str] = dict(target_dtypes)
        self.columns: list[str] = list(self.target_dtypes.keys())
        self.error_policy: ErrorPolicy = error_policy

    def _coerce_column(
        self, series: pd.Series, target_dtype: str
    ) -> tuple[pd.Series, int]:
        """
        Coerce a single column to ``target_dtype``.

        Returns the converted series and the number of previously-present
        values that became missing as a result of coercion.

        Raises
        ------
        DataTypeConversionError
            Under the strict policy, when a value cannot be converted.
        DataValidationError
            When the requested target dtype is not recognized by pandas.
        """
        original_missing = series.isna()

        # Datetime targets use pandas' dedicated parser which supports the
        # errors="coerce"/"raise" contract directly.
        is_datetime = "datetime" in target_dtype

        if self.error_policy == "coerce":
            if is_datetime:
                converted = pd.to_datetime(series, errors="coerce")
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                # A non-float numeric target (e.g. int64) cannot hold NaN. If
                # coercion produced any missing values, keep the values in a
                # float dtype so the recorded missing count is preserved
                # rather than raising; otherwise honor the requested dtype.
                if numeric.isna().any():
                    converted = numeric
                else:
                    try:
                        converted = numeric.astype(target_dtype)  # type: ignore[call-overload]
                    except (TypeError, ValueError) as exc:
                        raise DataValidationError(
                            f"Unrecognized target dtype '{target_dtype}' for "
                            f"column '{series.name}'."
                        ) from exc

            newly_missing = int((converted.isna() & ~original_missing).sum())
            return converted, newly_missing

        # Strict "raise" policy: identify the first non-convertible value.
        try:
            if is_datetime:
                probe = pd.to_datetime(series, errors="coerce")
            else:
                probe = pd.to_numeric(series, errors="coerce")
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                f"Unrecognized target dtype '{target_dtype}' for column "
                f"'{series.name}'."
            ) from exc

        failed_mask = probe.isna() & ~original_missing
        if bool(failed_mask.any()):
            first_bad_index = failed_mask.idxmax()
            first_bad_value = series.loc[first_bad_index]
            raise DataTypeConversionError(
                f"Cannot convert column '{series.name}' to dtype "
                f"'{target_dtype}': first non-convertible value "
                f"{first_bad_value!r} at index {first_bad_index!r}."
            )

        try:
            converted = (
                probe
                if is_datetime
                else series.astype(target_dtype)  # type: ignore[call-overload]
            )
        except (TypeError, ValueError) as exc:
            raise DataTypeConversionError(
                f"Cannot convert column '{series.name}' to dtype "
                f"'{target_dtype}': {exc}"
            ) from exc

        return converted, 0

    def apply(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        super().apply(dataframe)

        missing_columns = [
            column for column in self.columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        result = dataframe.copy()
        for column, target_dtype in self.target_dtypes.items():
            converted, _ = self._coerce_column(result[column], target_dtype)
            result[column] = converted

        return result

    def apply_with_result(
        self, dataframe: pd.DataFrame, dry_run: bool = False
    ) -> tuple[pd.DataFrame, OperationResult]:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(dataframe).__name__}"
            )

        missing_columns = [
            column for column in self.columns if column not in dataframe.columns
        ]
        if missing_columns:
            raise KeyError(f"Columns not found in dataframe: {missing_columns}")

        before_shape = dataframe.shape
        transformed = dataframe.copy()

        coerced_to_missing: dict[str, int] = {}
        total_coerced_to_missing = 0
        for column, target_dtype in self.target_dtypes.items():
            converted, newly_missing = self._coerce_column(
                transformed[column], target_dtype
            )
            transformed[column] = converted
            if newly_missing:
                coerced_to_missing[column] = newly_missing
                total_coerced_to_missing += newly_missing

        after_shape = transformed.shape

        desc = self.describe()
        desc["coerced_to_missing"] = coerced_to_missing
        desc["values_coerced_to_missing"] = total_coerced_to_missing

        result = OperationResult(
            operation_name=self.name,
            affected_columns=list(self.columns),
            rows_affected=total_coerced_to_missing,
            columns_affected=len(self.columns),
            before_shape=before_shape,
            after_shape=after_shape,
            strategy_description=(
                f"{self.name} ({self.error_policy}) on columns "
                f"{self.columns}; {total_coerced_to_missing} value(s) "
                f"coerced to missing"
            ),
            dry_run=dry_run,
            details=desc,
        )

        final_df = dataframe.copy() if dry_run else transformed
        return final_df, result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": list(self.columns),
            "target_dtypes": dict(self.target_dtypes),
            "error_policy": self.error_policy,
        }
