"""
sanitizepy.simple
~~~~~~~~~~~~~~~~~

High-level convenience API for sanitizepy.
Provides direct, chainable functions for inspection, cleaning, transformation,
feature engineering, contract validation, and reporting.
"""

from __future__ import annotations

import functools
import json
import re
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from sanitizepy.cleaning.encoding import EncodingRepairOperation
from sanitizepy.cleaning.engine import CleaningResult
from sanitizepy.cleaning.missing_tokens import MissingTokenOperation
from sanitizepy.cleaning.near_duplicates import NearDuplicateRemovalOperation
from sanitizepy.cleaning.operations import (
    DropColumns,
    DropDuplicates,
    DropMissingColumns,
    DropMissingRows,
    FillMissing,
)
from sanitizepy.cleaning.plan import CleaningPlan
from sanitizepy.cleaning.text_normalization import TextNormalizationOperation
from sanitizepy.cleaning.type_coercion import TypeCoercionOperation
from sanitizepy.exceptions import (
    DataValidationError,
    DependencyError,
)
from sanitizepy.inspection.anomalies import AnomalyInspector, AnomalyResult
from sanitizepy.inspection.datatypes import DatatypeInspectionResult, DatatypeInspector
from sanitizepy.inspection.duplicates import DuplicateInspectionResult, DuplicateInspector
from sanitizepy.inspection.memory import MemoryInspectionResult, MemoryInspector
from sanitizepy.inspection.missing import MissingInspectionResult, MissingValueInspector
from sanitizepy.inspection.near_duplicates import NearDuplicateDetector, NearDuplicateResult
from sanitizepy.inspection.profile import DatasetProfiler, profile_to_report
from sanitizepy.inspection.statistics import StatisticsInspectionResult, StatisticsInspector
from sanitizepy.inspection.text_quality import TextQualityAnalyzer, TextQualityResult
from sanitizepy.models.contracts import ColumnContract, DataContract
from sanitizepy.models.profile import DatasetProfile
from sanitizepy.models.replay import ReplayablePlan
from sanitizepy.preprocessing.operations import (
    ColumnInteraction,
    DatetimeFeatures,
    LogFeature,
    PolynomialFeature,
    RatioFeature,
)
from sanitizepy.reports.exporters import FileExporter
from sanitizepy.reports.renderers import JSONRenderer, TextRenderer
from sanitizepy.reports.report import Report
from sanitizepy.rules.builtins import register_builtin_rules
from sanitizepy.rules.engine import RuleEngine
from sanitizepy.rules.registry import RuleRegistry
from sanitizepy.rules.rule import RuleResult
from sanitizepy.ui import (
    COLOR_FAIL,
    COLOR_OK,
    SYMBOL_FAIL,
    SYMBOL_OK,
    Text,
    render_footer,
    render_header,
    render_metric,
    render_status_row,
    render_table,
)

# ===========================================================================
# Lightweight Tuple Wrappers for Rich Rendering
# ===========================================================================

class _TextQualityTuple(tuple):
    """Plain tuple of TextQualityResult with __rich_console__ support."""

    _footer_shape: tuple[int, ...] | None = None

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in self
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        yield render_header("text quality")
        yield Text("")

        headers = ["column", "avg chars", "avg tokens", "empty", "encoding"]
        rows = []
        for r in self:
            pct_empty = (r.empty_after_strip_count / max(1, r.non_null_count)) * 100
            rows.append(
                [
                    r.column,
                    f"{r.character_length_mean:.1f}",
                    f"{r.token_count_mean:.1f}",
                    f"{pct_empty:.1f}%",
                    f"{r.encoding_garbage_count:,}",
                ]
            )

        yield render_table(headers, rows)
        yield Text("")
        if self._footer_shape:
            yield render_footer(self._footer_shape)


class _RuleResultTuple(tuple):
    """Plain tuple of RuleResult with __rich_console__ support."""

    _footer_shape: tuple[int, ...] | None = None

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in self
        ]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __rich_console__(self, console: Any, options: Any) -> Any:
        yield render_header("validate")
        yield Text("")

        total = len(self)
        passed = sum(1 for r in self if r.passed)
        yield render_metric("Validation", f"{passed} / {total} passed", status=(passed == total))
        yield Text("")

        for r in self:
            sym = SYMBOL_OK if r.passed else SYMBOL_FAIL
            if not r.passed:
                if r.affected_rows > 0:
                    msg = f"{r.rule} — {r.affected_rows:,} invalid values"
                elif r.message:
                    msg = f"{r.rule} — {r.message}"
                else:
                    msg = r.rule
            else:
                msg = r.rule
            yield render_status_row(sym, msg)

        yield Text("")
        status_text = Text("Status: ", style="bold")
        if passed == total:
            status_text.append("PASSED", style=COLOR_OK)
        else:
            status_text.append("FAILED", style=COLOR_FAIL)
        yield status_text

        yield Text("")
        if self._footer_shape:
            yield render_footer(self._footer_shape)


# ===========================================================================
# Plain English Error Handling Decorator
# ===========================================================================

def _handle_errors(func: Any) -> Any:
    @functools.wraps(func)
    def wrapper(df: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"First argument must be a pandas DataFrame, got {type(df).__name__}."
            )
        if df.empty:
            raise ValueError("DataFrame is empty — load your data first")
        try:
            return func(df, *args, **kwargs)
        except KeyError as e:
            col = str(e).strip("'\"")
            raise KeyError(
                f"Column '{col}' not found. Your columns are: {list(df.columns)}"
            ) from None
        except DependencyError as e:
            msg = str(e).lower()
            if "rapidfuzz" in msg or "fuzzy" in msg:
                raise DependencyError(
                    "Run:  pip install 'sanitizepy[fuzzy]'  then try again"
                ) from None
            if "ftfy" in msg or "text" in msg:
                raise DependencyError(
                    "Run:  pip install 'sanitizepy[text]'  then try again"
                ) from None
            raise
        except DataValidationError as e:
            msg = str(e)
            m = re.search(r"['\"]([^'\"]+)['\"]", msg)
            col = m.group(1) if m else "column"
            if "median" in msg.lower() or "numeric" in msg.lower():
                raise DataValidationError(
                    f"'{col}' is text — run sp.fix_types(df) first or use strategy='mode'"
                ) from None
            raise
        except ValueError as e:
            msg = str(e)
            if "empty" in msg.lower():
                raise ValueError("DataFrame is empty — load your data first") from None
            raise

    return wrapper


# ===========================================================================
# Inspection Functions
# ===========================================================================

@_handle_errors
def inspect(df: pd.DataFrame) -> DatasetProfile:
    """Inspect and profile dataset quality returning a DatasetProfile."""
    return DatasetProfiler().profile(df)


@_handle_errors
def profile(df: pd.DataFrame) -> DatasetProfile:
    """Alias for inspect(df)."""
    return inspect(df)


@_handle_errors
def missing(df: pd.DataFrame, threshold: float = 0.0) -> MissingInspectionResult:
    """Inspect missing values across all columns."""
    return MissingValueInspector().inspect(df, threshold=threshold)


@_handle_errors
def duplicates(
    df: pd.DataFrame, subset: list[str] | tuple[str, ...] | None = None
) -> DuplicateInspectionResult:
    """Inspect exact duplicate rows."""
    return DuplicateInspector().inspect(df, subset=subset)


@_handle_errors
def dtypes(df: pd.DataFrame) -> DatatypeInspectionResult:
    """Inspect and recommend column datatypes."""
    return DatatypeInspector().inspect(df)


@_handle_errors
def stats(df: pd.DataFrame) -> StatisticsInspectionResult:
    """Inspect descriptive statistics for numeric columns."""
    return StatisticsInspector().inspect(df)


@_handle_errors
def memory(df: pd.DataFrame) -> MemoryInspectionResult:
    """Inspect memory usage and optimization potential."""
    return MemoryInspector().inspect(df)


@_handle_errors
def anomalies(
    df: pd.DataFrame, method: str = "iqr", seed: int | None = None
) -> AnomalyResult:
    """Detect anomalies in numeric columns using 'iqr' or 'zscore'."""
    return AnomalyInspector().inspect(df, method=method, seed=seed)  # type: ignore


@_handle_errors
def near_duplicates(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "exact_normalized",
) -> NearDuplicateResult:
    """Detect near-duplicate records by normalized matching or similarity."""
    return NearDuplicateDetector().detect(df, subset=cols, method=method)  # type: ignore


@_handle_errors
def text_quality(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> tuple[TextQualityResult, ...]:
    """Analyze text quality metrics for text/object columns."""
    analyzer = TextQualityAnalyzer()
    res = analyzer.analyze(df, subset=cols)
    wrapped = _TextQualityTuple(res)
    wrapped._footer_shape = (len(df), len(df.columns))
    return wrapped


# ===========================================================================
# Auto Clean
# ===========================================================================

@_handle_errors
def clean(
    df: pd.DataFrame, verbose: bool = True, dry_run: bool = False
) -> CleaningResult:
    """
    Safely clean a DataFrame using conservative operations:
      1. MissingTokenOperation()
      2. TextNormalizationOperation on object columns (NFKC + whitespace)
      3. EncodingRepairOperation on object columns
      4. DropDuplicates()

    Does NOT auto-fill missing values (reports remaining).
    Does NOT auto-cast ambiguous dtypes.
    """
    start_time = time.perf_counter()
    working_df = df.copy()
    b_shape = df.shape
    b_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Calculate initial health estimate
    missing_cells = int(working_df.isna().sum().sum())
    total_cells = max(1, working_df.size)
    dup_rows = int(working_df.duplicated().sum())
    comp_score = max(0.0, 100.0 * (1.0 - (missing_cells / total_cells)))
    uniq_score = max(0.0, 100.0 * (1.0 - (dup_rows / max(1, len(working_df)))))
    health_before = max(0, min(100, int(round(0.5 * comp_score + 0.5 * uniq_score))))

    operations: list[Any] = []
    audit_log: list[dict[str, Any]] = []

    # 1. Missing tokens
    token_op = MissingTokenOperation()
    working_df, token_res = token_op.apply_with_result(working_df, dry_run=dry_run)
    operations.append(token_res)

    # 2. Text normalization on object/string columns
    obj_cols = [
        c for c in working_df.columns if is_object_dtype(working_df[c]) or is_string_dtype(working_df[c])
    ]
    if obj_cols:
        norm_op = TextNormalizationOperation(
            subset=obj_cols,
            unicode_form="NFKC",
            normalize_whitespace=True,
            case="none",
        )
        working_df, norm_res = norm_op.apply_with_result(working_df, dry_run=dry_run)
        operations.append(norm_res)

    # 3. Encoding repair on object/string columns
    if obj_cols:
        enc_op = EncodingRepairOperation(
            subset=obj_cols,
            mode="core",
            error_on_unrepaired=False,
        )
        working_df, enc_res = enc_op.apply_with_result(working_df, dry_run=dry_run)
        operations.append(enc_res)

    # 4. Drop duplicates
    dup_op = DropDuplicates()
    working_df, dup_res = dup_op.apply_with_result(working_df, dry_run=dry_run)
    operations.append(dup_res)

    duration = time.perf_counter() - start_time
    a_shape = working_df.shape
    a_mb = working_df.memory_usage(deep=True).sum() / (1024 * 1024)

    rem_missing = int(working_df.isna().sum().sum())
    rem_dup = int(working_df.duplicated().sum())
    comp_after = max(0.0, 100.0 * (1.0 - (rem_missing / max(1, working_df.size))))
    uniq_after = max(0.0, 100.0 * (1.0 - (rem_dup / max(1, len(working_df)))))
    health_after = max(0, min(100, int(round(0.5 * comp_after + 0.5 * uniq_after))))

    attention: list[str] = []
    if rem_missing > 0:
        attention.append(f"{rem_missing:,} missing values remain — use sp.fill_missing(df)")

    # Build audit log
    for idx, op in enumerate(operations, start=1):
        audit_log.append(
            {
                "order": idx,
                "operation": op.operation_name,
                "rows_affected": op.rows_affected,
                "columns_affected": op.columns_affected,
                "before_shape": list(op.before_shape),
                "after_shape": list(op.after_shape),
            }
        )

    res = CleaningResult(
        data=working_df,
        operations=operations,
        dry_run=dry_run,
        duration_seconds=duration,
        audit_log=audit_log,
        health_before=health_before,
        health_after=health_after,
        before_shape=b_shape,
        after_shape=a_shape,
        before_mb=b_mb,
        after_mb=a_mb,
        still_needs_attention=attention,
    )
    return res


# ===========================================================================
# Manual Clean
# ===========================================================================

@_handle_errors
def drop_duplicates(
    df: pd.DataFrame,
    subset: list[str] | tuple[str, ...] | None = None,
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate rows from DataFrame."""
    return DropDuplicates(subset=subset, keep=keep).apply(df.copy())  # type: ignore


@_handle_errors
def drop_missing_rows(
    df: pd.DataFrame, subset: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Drop rows containing missing values."""
    return DropMissingRows(subset=subset).apply(df.copy())


@_handle_errors
def drop_missing_cols(
    df: pd.DataFrame, subset: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Drop columns that contain only or excessive missing values."""
    return DropMissingColumns(subset=subset).apply(df.copy())


@_handle_errors
def drop_cols(df: pd.DataFrame, cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Drop specified columns from DataFrame."""
    return DropColumns(columns=cols).apply(df.copy())


@_handle_errors
def fill_missing(
    df: pd.DataFrame,
    strategy: str | None = None,
    value: Any = None,
    cols: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Fill missing values.
    If value is provided -> fill with fixed value.
    If strategy is provided -> fill with strategy ('mean', 'median', 'mode', 'constant').
    If neither provided -> auto-split (median for numeric, mode for text/object).
    """
    out = df.copy()
    if value is not None:
        return FillMissing(strategy="constant", value=value, subset=cols).apply(out)
    if strategy is not None:
        return FillMissing(strategy=strategy, subset=cols).apply(out)

    target_cols = list(cols) if cols else list(out.columns)
    num_cols = [c for c in target_cols if is_numeric_dtype(out[c])]
    text_cols = [c for c in target_cols if not is_numeric_dtype(out[c])]

    if num_cols:
        out = FillMissing(strategy="median", subset=num_cols).apply(out)
    if text_cols:
        out = FillMissing(strategy="mode", subset=text_cols).apply(out)
    return out


@_handle_errors
def fix_types(
    df: pd.DataFrame, types: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Coerce column datatypes.
    If types dict is provided -> coerce specified columns.
    If types=None -> auto-cast only where object column has >95% numeric values
    and column name does NOT match id, code, phone, zip, postal patterns.
    """
    out = df.copy()
    if types is not None:
        return TypeCoercionOperation(target_dtypes=types, error_policy="coerce").apply(out)

    skip_pattern = re.compile(r"(?i)(^|_)(id|code|phone|zip|postal)($|_)")
    auto_types: dict[str, str] = {}

    for c in out.columns:
        if not is_object_dtype(out[c]) and not is_string_dtype(out[c]):
            continue
        if skip_pattern.search(str(c)):
            continue

        non_null = out[c].dropna()
        if len(non_null) == 0:
            continue

        converted = pd.to_numeric(non_null, errors="coerce")
        valid_ratio = converted.notna().mean()
        if valid_ratio > 0.95:
            # Check if all valid numbers are integer
            valid_vals = converted.dropna()
            if (valid_vals.astype(int) == valid_vals).all():
                auto_types[c] = "int64"
            else:
                auto_types[c] = "float64"

    if auto_types:
        return TypeCoercionOperation(target_dtypes=auto_types, error_policy="coerce").apply(out)
    return out


@_handle_errors
def fix_tokens(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    extra: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Normalize string missing tokens ('N/A', 'none', 'null', etc.) into np.nan."""
    return MissingTokenOperation(extra_tokens=extra, subset=cols).apply(df.copy())


@_handle_errors
def drop_near_duplicates(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    method: str = "exact_normalized",
    keep: str = "first",
) -> pd.DataFrame:
    """Drop near-duplicate rows based on normalized similarity."""
    return NearDuplicateRemovalOperation(
        subset=cols, method=method, keep=keep  # type: ignore
    ).apply(df.copy())


# ===========================================================================
# Transform Functions
# ===========================================================================

@_handle_errors
def dummies(
    df: pd.DataFrame, col: str | None = None, cols: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode categorical column(s)."""
    columns = [col] if col else cols
    return pd.get_dummies(df.copy(), columns=columns)


@_handle_errors
def normalize(
    df: pd.DataFrame, col: str | None = None, cols: list[str] | None = None
) -> pd.DataFrame:
    """Min-max scale numeric column(s) to [0, 1]."""
    out = df.copy()
    target_cols = [col] if col else (cols if cols else [c for c in out.columns if is_numeric_dtype(out[c])])
    for c in target_cols:
        min_v = out[c].min()
        max_v = out[c].max()
        denom = max_v - min_v
        if denom == 0 or pd.isna(denom):
            out[c] = 0.0
        else:
            out[c] = (out[c] - min_v) / denom
    return out


@_handle_errors
def standardize(
    df: pd.DataFrame, col: str | None = None, cols: list[str] | None = None
) -> pd.DataFrame:
    """Standardize numeric column(s) to mean 0, std 1."""
    out = df.copy()
    target_cols = [col] if col else (cols if cols else [c for c in out.columns if is_numeric_dtype(out[c])])
    for c in target_cols:
        mean_v = out[c].mean()
        std_v = out[c].std()
        if std_v == 0 or pd.isna(std_v):
            out[c] = 0.0
        else:
            out[c] = (out[c] - mean_v) / std_v
    return out


@_handle_errors
def rename(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns via dictionary mapping."""
    return df.copy().rename(columns=mapping)


@_handle_errors
def select(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Select a subset of columns."""
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Column '{missing_cols[0]}' not found. Your columns are: {list(df.columns)}")
    return df.copy()[cols]


@_handle_errors
def cast(df: pd.DataFrame, col: str, to: str) -> pd.DataFrame:
    """Cast a column to target datatype."""
    return TypeCoercionOperation(target_dtypes={col: to}, error_policy="coerce").apply(df.copy())


# ===========================================================================
# Text Functions
# ===========================================================================

@_handle_errors
def clean_text(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """
    Fully clean text columns:
      1. Missing tokens normalized
      2. Encoding repaired
      3. Unicode normalized (NFKC) & whitespace collapsed
    """
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    if not target:
        return df.copy()
    out = MissingTokenOperation(subset=target).apply(df.copy())
    out = EncodingRepairOperation(subset=target, error_on_unrepaired=False).apply(out)
    out = TextNormalizationOperation(
        subset=target, unicode_form="NFKC", normalize_whitespace=True
    ).apply(out)
    return out


@_handle_errors
def normalize_text(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Normalize text unicode to NFKC and collapse internal whitespace."""
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    return TextNormalizationOperation(
        subset=target, unicode_form="NFKC", normalize_whitespace=True
    ).apply(df.copy())


@_handle_errors
def lowercase(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Convert text column(s) to lowercase."""
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    return TextNormalizationOperation(subset=target, case="lower").apply(df.copy())


@_handle_errors
def uppercase(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Convert text column(s) to uppercase."""
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    return TextNormalizationOperation(subset=target, case="upper").apply(df.copy())


@_handle_errors
def titlecase(
    df: pd.DataFrame, cols: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Convert text column(s) to title case."""
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    return TextNormalizationOperation(subset=target, case="title").apply(df.copy())


@_handle_errors
def fix_encoding(
    df: pd.DataFrame,
    cols: list[str] | tuple[str, ...] | None = None,
    mode: str = "core",
) -> pd.DataFrame:
    """Repair mojibake and encoding artifacts in text columns."""
    target = cols or [
        c for c in df.columns if is_object_dtype(df[c]) or is_string_dtype(df[c])
    ]
    return EncodingRepairOperation(
        subset=target, mode=mode, error_on_unrepaired=False  # type: ignore
    ).apply(df.copy())


# ===========================================================================
# Feature Engineering
# ===========================================================================

@_handle_errors
def log(
    df: pd.DataFrame,
    col: str,
    offset: float = 1.0,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Create natural-log transformed feature."""
    return LogFeature(column=col, offset=offset, output_column=output_col).fit_transform(
        df.copy()
    )


@_handle_errors
def poly(
    df: pd.DataFrame,
    col: str,
    degree: int = 2,
    output_prefix: str | None = None,
) -> pd.DataFrame:
    """Generate polynomial powers for a numeric column."""
    return PolynomialFeature(
        column=col, degree=degree, output_prefix=output_prefix
    ).fit_transform(df.copy())


@_handle_errors
def interaction(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Create interaction feature: col_a * col_b."""
    return ColumnInteraction(
        column_a=col_a, column_b=col_b, output_column=output_col
    ).fit_transform(df.copy())


@_handle_errors
def ratio(
    df: pd.DataFrame,
    num: str,
    denom: str,
    output_col: str | None = None,
    zero_division: str = "nan",
) -> pd.DataFrame:
    """Create ratio feature: num / denom."""
    return RatioFeature(
        numerator=num,
        denominator=denom,
        output_column=output_col,
        zero_division=zero_division,  # type: ignore
    ).fit_transform(df.copy())


@_handle_errors
def datetime_features(
    df: pd.DataFrame,
    col: str,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Extract calendar features (year, month, day, etc.) from datetime column."""
    out = df.copy()
    if not is_datetime64_any_dtype(out[col]):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    feat_list = features or [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year",
        "week",
        "quarter",
        "hour",
        "minute",
        "second",
    ]
    return DatetimeFeatures(column=col, features=feat_list).fit_transform(out)


# ===========================================================================
# Validate & Contract
# ===========================================================================

@_handle_errors
def validate(
    df: pd.DataFrame, rules: list[str] | None = None
) -> tuple[RuleResult, ...]:
    """
    Validate DataFrame against built-in quality rules.
    Returns RuleResult tuple rendered via Rich on display.
    """
    reg = RuleRegistry()
    register_builtin_rules(reg)
    engine = RuleEngine(registry=reg)
    all_results = engine.run(df)
    if rules is not None:
        filtered = tuple(r for r in all_results if r.rule in rules)
    else:
        filtered = all_results

    wrapped = _RuleResultTuple(filtered)
    wrapped._footer_shape = (len(df), len(df.columns))
    return wrapped


@_handle_errors
def contract(df: pd.DataFrame, spec: dict[str, dict[str, Any]]) -> tuple[RuleResult, ...]:
    """
    Validate DataFrame against a declarative contract specification dictionary.
    Example:
      contract(df, {"email": {"nullable": False, "regex": "..."}})
    """
    cols_dict: dict[str, ColumnContract] = {}
    for col_name, rule_dict in spec.items():
        min_v = rule_dict.get("min_value", rule_dict.get("min"))
        max_v = rule_dict.get("max_value", rule_dict.get("max"))
        allowed = rule_dict.get("allowed_values")
        cols_dict[col_name] = ColumnContract(
            dtype=rule_dict.get("dtype"),
            nullable=rule_dict.get("nullable"),
            allowed_values=tuple(allowed) if allowed is not None else None,
            min_value=min_v,
            max_value=max_v,
            regex=rule_dict.get("regex"),
            unique=rule_dict.get("unique"),
        )
    contract_obj = DataContract(columns=cols_dict)
    results = RuleEngine().validate_contract(df, contract_obj)
    wrapped = _RuleResultTuple(results)
    wrapped._footer_shape = (len(df), len(df.columns))
    return wrapped


# ===========================================================================
# Report & Serialize
# ===========================================================================

@_handle_errors
def report(df: pd.DataFrame, save: str | None = None) -> Report:
    """Generate comprehensive quality Report and optionally export to file."""
    prof = DatasetProfiler().profile(df)
    rep = profile_to_report(prof)
    if save:
        if save.endswith(".json"):
            FileExporter(save).export(rep, JSONRenderer())
        else:
            FileExporter(save).export(rep, TextRenderer())
    return rep


def serialize(plan: CleaningPlan) -> str:
    """Serialize a CleaningPlan to JSON string."""
    return plan.serialize().to_json()


@_handle_errors
def replay(plan_json: str, df: pd.DataFrame) -> CleaningResult:
    """Replay a serialized CleaningPlan on DataFrame."""
    restored = ReplayablePlan.from_json(plan_json)
    return CleaningPlan.deserialize(restored).apply(df)


__all__ = [
    # Inspection
    "inspect",
    "profile",
    "missing",
    "duplicates",
    "dtypes",
    "stats",
    "memory",
    "anomalies",
    "near_duplicates",
    "text_quality",
    # Auto clean
    "clean",
    # Manual clean
    "drop_duplicates",
    "drop_missing_rows",
    "drop_missing_cols",
    "drop_cols",
    "fill_missing",
    "fix_types",
    "fix_tokens",
    "drop_near_duplicates",
    # Transform
    "dummies",
    "normalize",
    "standardize",
    "rename",
    "select",
    "cast",
    # Text
    "clean_text",
    "normalize_text",
    "lowercase",
    "uppercase",
    "titlecase",
    "fix_encoding",
    # Feature engineering
    "log",
    "poly",
    "interaction",
    "ratio",
    "datetime_features",
    # Validate & contract
    "validate",
    "contract",
    # Report & serialize
    "report",
    "serialize",
    "replay",
]
