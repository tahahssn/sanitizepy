"""
Cleaner Inspection Engine

The inspection package provides read-only dataset analysis.

It contains independent inspectors responsible for analyzing
different aspects of a pandas DataFrame without modifying it.

Available Inspectors
--------------------
- MissingValueInspector
- DuplicateInspector
- DatatypeInspector
- MemoryInspector
- StatisticsInspector
"""

from .datatypes import (
    ColumnTypeReport,
    DatatypeInspectionResult,
    DatatypeInspector,
    DatatypeSummary,
)

from .duplicates import (
    DuplicateInspectionResult,
    DuplicateInspector,
    DuplicateSummary,
)

from .memory import (
    MemoryColumnReport,
    MemoryInspectionResult,
    MemoryInspector,
    MemorySummary,
)

from .missing import (
    MissingColumnReport,
    MissingInspectionResult,
    MissingSummary,
    MissingValueInspector,
)

from .statistics import (
    NumericColumnStatistics,
    StatisticsInspectionResult,
    StatisticsInspector,
    StatisticsSummary,
)

__all__ = [
    # Missing Values
    "MissingValueInspector",
    "MissingInspectionResult",
    "MissingSummary",
    "MissingColumnReport",

    # Duplicates
    "DuplicateInspector",
    "DuplicateInspectionResult",
    "DuplicateSummary",

    # Datatypes
    "DatatypeInspector",
    "DatatypeInspectionResult",
    "DatatypeSummary",
    "ColumnTypeReport",

    # Memory
    "MemoryInspector",
    "MemoryInspectionResult",
    "MemorySummary",
    "MemoryColumnReport",

    # Statistics
    "StatisticsInspector",
    "StatisticsInspectionResult",
    "StatisticsSummary",
    "NumericColumnStatistics",
]