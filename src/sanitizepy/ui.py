"""
sanitizepy.ui
~~~~~~~~~~~~~

Shared presentation layer for sanitizepy.
Provides Rich console detection, symbol & color constants, and rendering primitives.
Presentation only — no business logic.
"""

from __future__ import annotations

from typing import Any, Callable

from rich import box
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

# ===========================================================================
# Symbols & Colors (System-wide constants)
# ===========================================================================

SYMBOL_OK: str = "✓"
SYMBOL_WARN: str = "⚠"
SYMBOL_FAIL: str = "✗"
SYMBOL_INFO: str = "•"

COLOR_OK: str = "green"
COLOR_WARN: str = "yellow"
COLOR_FAIL: str = "red"
COLOR_INFO: str = "white"
COLOR_META: str = "cyan"  # headings, labels, metadata


# ===========================================================================
# SanitizeConsole
# ===========================================================================

class SanitizeConsole:
    """
    Console wrapper with environment detection and width responsiveness.

    Modes:
      - < 80 cols   -> compact mode, fewer secondary columns, truncate strings
      - 80-120 cols -> standard mode
      - > 120 cols  -> wide mode, show additional columns
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console

    def is_notebook(self) -> bool:
        """Detect whether running in Jupyter Notebook, Lab, or Google Colab."""
        try:
            from IPython import get_ipython  # type: ignore

            ip = get_ipython()
            if ip is None:
                return False
            name = ip.__class__.__name__
            if name in ("ZMQInteractiveShell", "Shell"):
                return True
            if "google.colab" in str(ip):
                return True
            return False
        except (ImportError, NameError):
            return False

    def get_console(self) -> Console:
        """Return the active Rich console, auto-detecting terminal/notebook."""
        if self._console is None:
            self._console = Console()
        return self._console

    def get_width(self) -> int:
        """Return the current terminal/console width."""
        return self.get_console().width

    def get_mode(self) -> str:
        """Return current responsive mode ('compact', 'standard', 'wide')."""
        w = self.get_width()
        if w < 80:
            return "compact"
        if w <= 120:
            return "standard"
        return "wide"


_global_console = SanitizeConsole()


def get_console() -> Console:
    """Return shared default Rich console."""
    return _global_console.get_console()


# ===========================================================================
# Rendering Primitives
# ===========================================================================

def render_header(title: str, width: int = 40) -> Text:
    """
    Render standardized block header.

    Example:
      sanitizepy / inspect
      ────────────────────────────────────────
    """
    text = Text()
    text.append("sanitizepy", style=COLOR_META)
    text.append(" / ")
    text.append(title, style="bold")
    text.append("\n")
    text.append("─" * width, style="dim")
    return text


def render_footer(
    df_or_shape: Any,
    duration_seconds: float | None = None,
) -> Text:
    """
    Render standardized block footer.

    Example:
      30,482 rows × 18 columns  •  24.8 MB
    """
    parts: list[str] = []

    if hasattr(df_or_shape, "shape"):
        rows, cols = df_or_shape.shape
        try:
            mb = df_or_shape.memory_usage(deep=True).sum() / (1024 * 1024)
            parts.append(f"{rows:,} rows × {cols:,} columns  •  {mb:.1f} MB")
        except Exception:
            parts.append(f"{rows:,} rows × {cols:,} columns")
    elif isinstance(df_or_shape, tuple):
        if len(df_or_shape) == 2:
            parts.append(f"{df_or_shape[0]:,} rows × {df_or_shape[1]:,} columns")
        elif len(df_or_shape) >= 3:
            parts.append(
                f"{df_or_shape[0]:,} rows × {df_or_shape[1]:,} columns  •  "
                f"{df_or_shape[2]:.1f} MB"
            )
    else:
        # Support objects with row_count / column_count / memory
        rows = getattr(df_or_shape, "row_count", None)
        if rows is None:
            rows = getattr(df_or_shape, "rows", None)
        cols = getattr(df_or_shape, "column_count", None)
        if cols is None:
            cols = getattr(df_or_shape, "columns", None)

        mb: float | None = getattr(df_or_shape, "memory_mb", None)
        if (
            mb is None
            and hasattr(df_or_shape, "memory")
            and hasattr(df_or_shape.memory, "summary")
        ):
            mb = df_or_shape.memory.summary.total_memory_mb

        if rows is not None and cols is not None:
            if mb is not None:
                parts.append(f"{rows:,} rows × {cols:,} columns  •  {mb:.1f} MB")
            else:
                parts.append(f"{rows:,} rows × {cols:,} columns")
        elif isinstance(df_or_shape, str):
            parts.append(df_or_shape)
        else:
            parts.append(str(df_or_shape))

    if duration_seconds is not None:
        parts.append(f"{duration_seconds:.2f}s")

    return Text("  •  ".join(parts), style="dim")


def render_table(
    headers: list[str],
    rows: list[list[Any]],
    col_styles: dict[int, str] | None = None,
    col_align: dict[int, str] | None = None,
    max_col_width: int | None = None,
) -> Table:
    """
    Render table adapting to terminal width, truncating long strings with '...'.
    """
    tbl = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        header_style=f"bold {COLOR_META}",
    )

    mode = _global_console.get_mode()
    truncate_limit = max_col_width or (20 if mode == "compact" else 35)

    for idx, h in enumerate(headers):
        align = "left"
        if col_align and idx in col_align:
            align = col_align[idx]
        elif h.lower() in (
            "%",
            "missing",
            "count",
            "nulls",
            "unique",
            "min",
            "max",
            "rows",
            "time",
            "flagged",
            "empty",
            "encoding",
            "affected",
        ):
            align = "right"

        style = col_styles.get(idx) if col_styles else None
        tbl.add_column(h, justify=align, style=style, no_wrap=(mode == "compact"))

    for row in rows:
        formatted_row: list[str] = []
        for cell in row:
            val_str = str(cell) if cell is not None else "—"
            if len(val_str) > truncate_limit:
                val_str = val_str[: truncate_limit - 3] + "..."
            formatted_row.append(val_str)
        tbl.add_row(*formatted_row)

    return tbl


def render_metric(
    label: str,
    value: Any,
    status: str | bool | None = None,
) -> Text:
    """
    Render single line metric.
    Example: "  Rows                     30,482"
    """
    text = Text("  ")
    text.append(f"{label:<25}", style="bold")

    val_str = str(value)
    style: str | None = None
    if status is True or status == "ok" or status == SYMBOL_OK:
        style = COLOR_OK
    elif status == "warn" or status == SYMBOL_WARN:
        style = COLOR_WARN
    elif status is False or status == "fail" or status == SYMBOL_FAIL:
        style = COLOR_FAIL
    elif isinstance(status, str):
        style = severity_to_style(status)

    text.append(f"{val_str:>12}", style=style)
    return text


def render_section(title: str) -> Text:
    """
    Render section header within an output block.
    Example:
      DATASET
      ────────────────────
    """
    text = Text()
    text.append(f"\n{title}\n", style=f"bold {COLOR_META}")
    sep_len = max(len(title), 20)
    text.append("─" * sep_len, style="dim")
    return text


def render_status_row(
    symbol: str,
    message: str,
    count: int | None = None,
) -> Text:
    """
    Render a single status row.
    Example: "  ✓ 1,204 missing tokens normalized"
    """
    text = Text("  ")
    if symbol == SYMBOL_OK:
        text.append(f"{symbol} ", style=COLOR_OK)
    elif symbol == SYMBOL_WARN:
        text.append(f"{symbol} ", style=COLOR_WARN)
    elif symbol == SYMBOL_FAIL:
        text.append(f"{symbol} ", style=COLOR_FAIL)
    else:
        text.append(f"{symbol} ", style=COLOR_INFO)

    if count is not None:
        text.append(f"{count:,} ")
    text.append(message)
    return text


def render_summary_box(
    title: str,
    content_fn: Callable[[], Any],
    footer_fn: Callable[[], Any] | None = None,
) -> Group:
    """Wrap content in a standardized presentation block with header and footer."""
    items: list[RenderableType] = [render_header(title), Text("")]
    content = content_fn()
    if isinstance(content, (list, tuple)):
        items.extend(content)
    elif content is not None:
        items.append(content)

    if footer_fn is not None:
        items.append(Text(""))
        footer = footer_fn()
        if footer is not None:
            items.append(footer)

    return Group(*items)


def render_health_score(score: int) -> Text:
    """
    Render health score line.
    Example: "  Health                   82 / 100"
    """
    color = COLOR_OK if score >= 80 else (COLOR_WARN if score >= 60 else COLOR_FAIL)
    text = Text("  Health                   ", style="bold")
    text.append(f"{score} / 100", style=color)
    return text


def render_before_after(
    before_shape: tuple[int, int],
    after_shape: tuple[int, int],
    before_mb: float,
    after_mb: float,
) -> Text:
    """
    Render before/after comparison lines.
    """
    b_rows, b_cols = before_shape
    a_rows, a_cols = after_shape
    text = Text()
    text.append(
        f"Before    {b_rows:,} rows × {b_cols:,} columns  •  {before_mb:.1f} MB\n"
    )
    text.append(
        f"After     {a_rows:,} rows × {a_cols:,} columns  •  {after_mb:.1f} MB"
    )
    return text


def render_recommendation(text: str) -> Text:
    """Render recommendation line in cyan."""
    return Text(f"  {SYMBOL_OK} {text}", style=COLOR_META)


def render_changes(changes: list[dict[str, Any]]) -> Text:
    """Render the CHANGES section of clean output."""
    text = Text("\nCHANGES\n", style=f"bold {COLOR_META}")
    for change in changes:
        sym = change.get("symbol", SYMBOL_OK)
        msg = change.get("message", "")
        cnt = change.get("count")
        text.append_text(render_status_row(sym, msg, count=cnt))
        text.append("\n")
    return text


def severity_to_style(severity: str) -> str:
    """Map severity string to UI color style."""
    s = str(severity).upper()
    if s in ("CRITICAL", "HIGH", "ERROR", "FAIL"):
        return COLOR_FAIL
    if s in ("MEDIUM", "WARNING", "WARN"):
        return COLOR_WARN
    if s in ("LOW", "OK", "SUCCESS", "PASS"):
        return COLOR_OK
    return COLOR_INFO


def status_symbol(passed: bool) -> str:
    """Convert boolean pass/fail to standard symbol."""
    return SYMBOL_OK if passed else SYMBOL_FAIL


__all__ = [
    "SYMBOL_OK",
    "SYMBOL_WARN",
    "SYMBOL_FAIL",
    "SYMBOL_INFO",
    "COLOR_OK",
    "COLOR_WARN",
    "COLOR_FAIL",
    "COLOR_INFO",
    "COLOR_META",
    "SanitizeConsole",
    "get_console",
    "render_header",
    "render_footer",
    "render_table",
    "render_metric",
    "render_section",
    "render_status_row",
    "render_summary_box",
    "render_health_score",
    "render_before_after",
    "render_recommendation",
    "render_changes",
    "severity_to_style",
    "status_symbol",
]
