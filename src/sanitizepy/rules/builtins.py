"""
sanitizepy.rules.builtins
~~~~~~~~~~~~~~~~~~~~~~

Built-in rules shipped with Cleaner.

Historically these rules were registered as no-op stubs that always
reported success. They are now real, deterministic validators: each rule
inspects the DataFrame for its named concern (missing values, duplicate
rows, whitespace, string casing, column names, and the remaining
built-ins) and returns a genuine :class:`~sanitizepy.rules.rule.RuleResult`
whose ``passed`` flag reflects the actual state of the data.

Every rule:

* is deterministic (no randomness, stable ordering of affected columns);
* is read-only (it never mutates the DataFrame it inspects);
* returns a passing, informational result on a clean DataFrame;
* returns a failing result -- with affected columns, affected rows, and a
  small, JSON-serializable metadata summary -- when its concern is
  violated.

Rule names and the :func:`register_builtin_rules` signature are preserved
for backward compatibility.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pandas.api import types as pdtypes

from .base import BaseRule
from .registry import RuleRegistry
from .rule import RuleCategory, RuleResult, RuleSeverity

# Threshold above which an object column is considered "high cardinality":
# the ratio of distinct non-missing values to non-missing rows.
_HIGH_CARDINALITY_RATIO: float = 0.9

# IQR multiplier used for the deterministic outlier check.
_IQR_MULTIPLIER: float = 1.5

# Matches leading/trailing whitespace or any run of two-or-more internal
# whitespace characters (irregular whitespace).
_IRREGULAR_WHITESPACE = re.compile(r"^\s|\s$|\s{2,}")

# A normalized column name is lowercase, uses single underscores, and holds
# only ASCII letters, digits, and underscores.
_NORMALIZED_COLUMN_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _string_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Return object/string columns in stable column order.
    """
    return [
        str(column)
        for column in dataframe.columns
        if pdtypes.is_object_dtype(dataframe[column])
        or pdtypes.is_string_dtype(dataframe[column])
    ]


def _pass(
    *,
    rule: str,
    category: RuleCategory,
    message: str,
) -> RuleResult:
    """
    Build a passing, informational ``RuleResult``.
    """
    return RuleResult(
        rule=rule,
        passed=True,
        severity=RuleSeverity.INFO,
        category=category,
        message=message,
    )


def _fail(
    *,
    rule: str,
    category: RuleCategory,
    message: str,
    affected_columns: tuple[str, ...] = (),
    affected_rows: int = 0,
    metadata: dict[str, Any] | None = None,
    severity: RuleSeverity = RuleSeverity.WARNING,
) -> RuleResult:
    """
    Build a failing ``RuleResult`` with affected columns/rows wiring.
    """
    return RuleResult(
        rule=rule,
        passed=False,
        severity=severity,
        category=category,
        message=message,
        affected_columns=affected_columns,
        affected_rows=affected_rows,
        metadata=metadata or {},
    )


# ==========================================================
# Concrete built-in rules
# ==========================================================


class MissingValuesRule(BaseRule):
    """
    Flag columns and cells that contain missing values.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        per_column = dataframe.isna().sum()
        affected = tuple(
            str(column) for column in dataframe.columns if int(per_column[column]) > 0
        )
        total_missing = int(per_column.sum())

        if total_missing == 0:
            return _pass(
                rule=self.name,
                category=RuleCategory.COMPLETENESS,
                message="No missing values detected.",
            )

        affected_rows = int(dataframe.isna().any(axis=1).sum())
        return _fail(
            rule=self.name,
            category=RuleCategory.COMPLETENESS,
            message=(
                f"{total_missing} missing value(s) across "
                f"{len(affected)} column(s)."
            ),
            affected_columns=affected,
            affected_rows=affected_rows,
            metadata={"missing_cells": total_missing},
        )


class DuplicateRowsRule(BaseRule):
    """
    Flag exact duplicate rows.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        duplicated_mask = dataframe.duplicated(keep=False)
        duplicate_rows = int(duplicated_mask.sum())

        if duplicate_rows == 0:
            return _pass(
                rule=self.name,
                category=RuleCategory.UNIQUENESS,
                message="No duplicate rows detected.",
            )

        extra_rows = int(dataframe.duplicated(keep="first").sum())
        return _fail(
            rule=self.name,
            category=RuleCategory.UNIQUENESS,
            message=(
                f"{duplicate_rows} row(s) participate in duplicate groups "
                f"({extra_rows} redundant row(s))."
            ),
            affected_rows=duplicate_rows,
            metadata={"duplicate_rows": duplicate_rows, "redundant_rows": extra_rows},
        )


class DuplicateColumnsRule(BaseRule):
    """
    Flag columns that are duplicated by name or by content.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        columns = [str(column) for column in dataframe.columns]

        duplicate_names = sorted({name for name in columns if columns.count(name) > 1})

        content_duplicates: list[str] = []
        seen: dict[tuple[Any, ...], str] = {}
        for position in range(dataframe.shape[1]):
            series = dataframe.iloc[:, position]
            key = tuple(series.fillna("__sanitizepy_na__").tolist())
            name = str(dataframe.columns[position])
            if key in seen:
                content_duplicates.append(name)
            else:
                seen[key] = name

        affected = tuple(sorted(set(duplicate_names) | set(content_duplicates)))

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.UNIQUENESS,
                message="No duplicate columns detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.UNIQUENESS,
            message=(
                f"{len(affected)} duplicate column(s) detected "
                f"(by name or content)."
            ),
            affected_columns=affected,
            metadata={
                "duplicate_names": duplicate_names,
                "content_duplicates": sorted(set(content_duplicates)),
            },
        )


class InvalidDtypesRule(BaseRule):
    """
    Flag object columns whose non-missing values are entirely numeric.

    Such columns are typically stored with an incorrect (object) dtype
    and should be coerced to a numeric dtype.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        for column in _string_columns(dataframe):
            series = dataframe[column]
            non_missing = series[series.notna()]
            if non_missing.empty:
                continue
            coerced = pd.to_numeric(non_missing, errors="coerce")
            if coerced.notna().all():
                affected.append(column)

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.VALIDITY,
                message="No columns with misclassified numeric data detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.VALIDITY,
            message=(
                f"{len(affected)} object column(s) hold entirely numeric "
                f"values and should be coerced."
            ),
            affected_columns=tuple(affected),
            metadata={"numeric_object_columns": affected},
        )


class OutliersRule(BaseRule):
    """
    Flag numeric columns containing IQR-based outliers.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        total_outliers = 0
        for column in dataframe.columns:
            series = dataframe[column]
            if not pdtypes.is_numeric_dtype(series) or pdtypes.is_bool_dtype(series):
                continue
            non_missing = series.dropna()
            if non_missing.empty:
                continue
            q1 = float(non_missing.quantile(0.25))
            q3 = float(non_missing.quantile(0.75))
            iqr = q3 - q1
            if iqr <= 0:
                continue
            lower = q1 - _IQR_MULTIPLIER * iqr
            upper = q3 + _IQR_MULTIPLIER * iqr
            count = int(((non_missing < lower) | (non_missing > upper)).sum())
            if count > 0:
                affected.append(str(column))
                total_outliers += count

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.DATA_QUALITY,
                message="No statistical outliers detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.DATA_QUALITY,
            message=(
                f"{total_outliers} outlier value(s) across "
                f"{len(affected)} numeric column(s)."
            ),
            affected_columns=tuple(affected),
            affected_rows=total_outliers,
            metadata={"outlier_count": total_outliers},
        )


class ConstantColumnsRule(BaseRule):
    """
    Flag columns whose non-missing values are all identical.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        for column in dataframe.columns:
            non_missing = dataframe[column].dropna()
            if non_missing.empty:
                continue
            if non_missing.nunique(dropna=True) <= 1:
                affected.append(str(column))

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.DATA_QUALITY,
                message="No constant columns detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.DATA_QUALITY,
            message=f"{len(affected)} constant column(s) detected.",
            affected_columns=tuple(affected),
            metadata={"constant_columns": affected},
        )


class HighCardinalityRule(BaseRule):
    """
    Flag object columns whose distinct-value ratio is very high.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        for column in _string_columns(dataframe):
            non_missing = dataframe[column].dropna()
            total = int(non_missing.shape[0])
            if total < 2:
                continue
            ratio = non_missing.nunique(dropna=True) / total
            if ratio > _HIGH_CARDINALITY_RATIO:
                affected.append(column)

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.DATA_QUALITY,
                message="No high-cardinality columns detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.DATA_QUALITY,
            message=(
                f"{len(affected)} high-cardinality object column(s) "
                f"exceed a {_HIGH_CARDINALITY_RATIO:.0%} distinct ratio."
            ),
            affected_columns=tuple(affected),
            metadata={"threshold": _HIGH_CARDINALITY_RATIO},
        )


class WhitespaceRule(BaseRule):
    """
    Flag string cells with leading, trailing, or irregular whitespace.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        total_cells = 0
        for column in _string_columns(dataframe):
            series = dataframe[column]
            non_missing = series[series.notna()]
            if non_missing.empty:
                continue
            as_text = non_missing.astype(str)
            flagged = as_text.map(
                lambda value: _IRREGULAR_WHITESPACE.search(value) is not None
            )
            count = int(flagged.astype(bool).sum())
            if count > 0:
                affected.append(column)
                total_cells += count

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.CONSISTENCY,
                message="No irregular whitespace detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.CONSISTENCY,
            message=(
                f"{total_cells} cell(s) across {len(affected)} column(s) "
                f"contain leading, trailing, or irregular whitespace."
            ),
            affected_columns=tuple(affected),
            affected_rows=total_cells,
            metadata={"whitespace_cells": total_cells},
        )


class StringCaseRule(BaseRule):
    """
    Flag string columns with inconsistent casing.

    Casing is inconsistent when two distinct values are equal after
    lowercasing (for example ``"Yes"`` and ``"yes"``).
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected: list[str] = []
        for column in _string_columns(dataframe):
            series = dataframe[column]
            non_missing = series[series.notna()].astype(str)
            if non_missing.empty:
                continue
            distinct = set(non_missing.unique())
            lowered = {value.lower() for value in distinct}
            if len(lowered) < len(distinct):
                affected.append(column)

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.CONSISTENCY,
                message="No inconsistent string casing detected.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.CONSISTENCY,
            message=(
                f"{len(affected)} column(s) contain values that differ "
                f"only by letter case."
            ),
            affected_columns=tuple(affected),
            metadata={"inconsistent_case_columns": affected},
        )


class ColumnNamesRule(BaseRule):
    """
    Flag column names that are not normalized.

    A normalized name is lowercase snake_case using only ASCII letters,
    digits, and single underscores, with no surrounding whitespace.
    """

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        **kwargs: Any,
    ) -> RuleResult:
        affected = tuple(
            str(column)
            for column in dataframe.columns
            if _NORMALIZED_COLUMN_NAME.fullmatch(str(column)) is None
        )

        if not affected:
            return _pass(
                rule=self.name,
                category=RuleCategory.STRUCTURE,
                message="All column names are normalized.",
            )

        return _fail(
            rule=self.name,
            category=RuleCategory.STRUCTURE,
            message=f"{len(affected)} column name(s) are not normalized.",
            affected_columns=affected,
            metadata={"non_normalized_names": list(affected)},
        )


def register_builtin_rules(
    registry: RuleRegistry,
) -> None:
    """
    Register all built-in Cleaner rules.

    Parameters
    ----------
    registry:
        Rule registry instance.
    """

    registry.register(
        MissingValuesRule(
            name="missing_values",
            description="Handle missing values.",
        )
    )

    registry.register(
        DuplicateRowsRule(
            name="duplicate_rows",
            description="Remove duplicated rows.",
        )
    )

    registry.register(
        DuplicateColumnsRule(
            name="duplicate_columns",
            description="Remove duplicated columns.",
        )
    )

    registry.register(
        InvalidDtypesRule(
            name="invalid_dtypes",
            description="Correct invalid data types.",
        )
    )

    registry.register(
        OutliersRule(
            name="outliers",
            description="Handle statistical outliers.",
        )
    )

    registry.register(
        ConstantColumnsRule(
            name="constant_columns",
            description="Remove constant columns.",
        )
    )

    registry.register(
        HighCardinalityRule(
            name="high_cardinality",
            description="Handle high-cardinality features.",
        )
    )

    registry.register(
        WhitespaceRule(
            name="whitespace",
            description="Normalize whitespace.",
        )
    )

    registry.register(
        StringCaseRule(
            name="string_case",
            description="Normalize string casing.",
        )
    )

    registry.register(
        ColumnNamesRule(
            name="column_names",
            description="Normalize column names.",
        )
    )


__all__ = [
    "MissingValuesRule",
    "DuplicateRowsRule",
    "DuplicateColumnsRule",
    "InvalidDtypesRule",
    "OutliersRule",
    "ConstantColumnsRule",
    "HighCardinalityRule",
    "WhitespaceRule",
    "StringCaseRule",
    "ColumnNamesRule",
    "register_builtin_rules",
]
