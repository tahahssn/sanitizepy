"""
Dataset Health Model & Scoring Engine.

Computes composite Dataset Health Score (0-100) based on 5 quality dimensions:
1. Completeness (% non-missing values)
2. Uniqueness (% non-duplicate rows/keys)
3. Consistency (casing, format, dtype alignment)
4. Validity (outlier ratio, plausible ranges)
5. Integrity (constant/empty columns)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sanitizepy.models.recommendations import Recommendation


@dataclass
class DatasetIssue:
    """
    Represents a specific data quality issue identified during dataset inspection.
    """

    title: str
    description: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    category: str
    column: str | None = None
    evidence: str = ""
    recommendation_text: str = ""


@dataclass
class DatasetHealthReport:
    """
    Comprehensive Dataset Health Report containing composite quality scores,
    identified issues, memory footprint, and actionable recommendations.
    """

    health_score: int  # 0 to 100
    completeness_score: float
    uniqueness_score: float
    consistency_score: float
    validity_score: float
    integrity_score: float

    rows: int
    columns: int
    memory_mb: float

    issues: list[DatasetIssue] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)

    @property
    def critical_issues(self) -> list[DatasetIssue]:
        """Return all CRITICAL severity issues."""
        return [i for i in self.issues if i.severity.upper() == "CRITICAL"]

    @property
    def warning_issues(self) -> list[DatasetIssue]:
        """Return all WARNING severity issues."""
        return [i for i in self.issues if i.severity.upper() == "WARNING"]

    @property
    def info_issues(self) -> list[DatasetIssue]:
        """Return all INFO severity issues."""
        return [i for i in self.issues if i.severity.upper() == "INFO"]

    def show(self) -> None:
        """
        Display a rich, formatted dataset health report in the terminal.
        """
        console = Console()

        color = (
            "green"
            if self.health_score >= 80
            else ("yellow" if self.health_score >= 60 else "red")
        )
        header_text = (
            "[bold]Dataset Health Score: "
            f"[{color}]{self.health_score}/100[/{color}][/bold]\n"
            f"Rows: {self.rows:,} | Columns: {self.columns:,} | "
            f"Memory: {self.memory_mb:.2f} MB\n"
            f"Completeness: {self.completeness_score:.1f}% | "
            f"Uniqueness: {self.uniqueness_score:.1f}% | "
            f"Consistency: {self.consistency_score:.1f}% | "
            f"Validity: {self.validity_score:.1f}% | "
            f"Integrity: {self.integrity_score:.1f}%"
        )
        console.print(Panel(header_text, title="DATASET HEALTH REPORT", expand=False))

        if self.issues:
            table = Table(
                title="IDENTIFIED ISSUES & RECOMMENDATIONS",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Severity", style="bold", width=10)
            table.add_column("Column", style="cyan", width=15)
            table.add_column("Issue & Evidence", style="white")
            table.add_column("Recommendation", style="green")

            for issue in self.issues:
                sev_upper = issue.severity.upper()
                sev_style = (
                    "bold red"
                    if sev_upper == "CRITICAL"
                    else ("bold yellow" if sev_upper == "WARNING" else "dim blue")
                )
                col_name = issue.column or "-"
                details = (
                    f"[bold]{issue.title}[/bold]\n[dim]{issue.evidence}[/dim]"
                    if issue.evidence
                    else issue.title
                )
                table.add_row(
                    f"[{sev_style}]{sev_upper}[/{sev_style}]",
                    col_name,
                    details,
                    issue.recommendation_text,
                )

            console.print(table)
        else:
            console.print(
                "[green]No data quality issues detected! Dataset is in prime "
                "condition.[/green]"
            )
