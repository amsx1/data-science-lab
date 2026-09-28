"""Typed containers returned by every analysis module.

Keeping the results in explicit dataclasses (instead of loose dictionaries) means
the UI, the report writers and the test-suite all agree on field names, and it
makes the whole pipeline easy to reason about.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from src import config
from src.utils import to_builtin


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
class Severity(str, Enum):
    """How much attention a finding deserves."""

    INFO = "info"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        """Numeric ordering, useful for sorting (critical = 4)."""
        return config.SEVERITY_RANK[self.value]

    @property
    def color(self) -> str:
        """Hex colour used by the UI and the HTML report."""
        return config.SEVERITY_COLORS[self.value]

    @property
    def label(self) -> str:
        """Title-cased severity name."""
        return self.value.capitalize()


@dataclass
class Finding:
    """A single, evidence-backed observation about the dataset."""

    category: str
    title: str
    detail: str
    severity: Severity = Severity.INFO
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation, safe for JSON/HTML rendering."""
        payload = to_builtin(asdict(self))
        payload["severity"] = self.severity.value
        return payload


def sort_findings(findings: list[Finding]) -> list[Finding]:
    """Sort findings from most to least severe, keeping insertion order on ties."""
    return sorted(findings, key=lambda f: -f.severity.rank)


# --------------------------------------------------------------------------- #
# Dataset structure
# --------------------------------------------------------------------------- #
@dataclass
class ColumnTypes:
    """Column names grouped by the role the tool assigns to them."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    other: list[str] = field(default_factory=list)
    #: Text columns whose unique-value ratio suggests free text / identifiers.
    text_like: list[str] = field(default_factory=list)

    @property
    def all_columns(self) -> list[str]:
        """Every analysed column, in group order."""
        return (
            self.numeric + self.categorical + self.datetime + self.boolean + self.other
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view of the grouping."""
        return to_builtin(asdict(self))


@dataclass
class DatasetOverview:
    """Structural description shown at the top of the dashboard."""

    filename: str
    n_rows: int
    n_columns: int
    memory_bytes: int
    total_cells: int
    missing_cells: int
    duplicate_rows: int
    column_types: ColumnTypes
    dtype_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    load_warnings: list[str] = field(default_factory=list)

    @property
    def numeric_columns(self) -> list[str]:
        """Names of the numeric columns."""
        return self.column_types.numeric

    @property
    def categorical_columns(self) -> list[str]:
        """Names of the categorical columns."""
        return self.column_types.categorical

    @property
    def datetime_columns(self) -> list[str]:
        """Names of the datetime columns."""
        return self.column_types.datetime

    @property
    def missing_rate(self) -> float:
        """Share of all cells that are missing."""
        if self.total_cells == 0:
            return 0.0
        return self.missing_cells / self.total_cells


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
@dataclass
class MissingColumn:
    """Missingness of a single column."""

    column: str
    missing: int
    missing_rate: float
    dtype: str

    @property
    def present(self) -> int:
        """Number of non-missing values."""
        return max(self.missing, 0)


@dataclass
class MissingReport:
    """Missing-value analysis for the whole table."""

    total_cells: int
    missing_cells: int
    missing_rate: float
    affected_columns: list[MissingColumn] = field(default_factory=list)
    complete_rows: int = 0
    rows_with_missing: int = 0
    columns_with_missing: int = 0

    @property
    def complete_row_rate(self) -> float:
        """Share of rows that have no missing value at all."""
        if self.complete_rows + self.rows_with_missing == 0:
            return 0.0
        return self.complete_rows / (self.complete_rows + self.rows_with_missing)


@dataclass
class DuplicateReport:
    """Duplicate-record analysis."""

    total_rows: int
    duplicate_rows: int
    duplicate_rate: float
    examples: pd.DataFrame = field(default_factory=pd.DataFrame)
    duplicate_group_count: int = 0

    @property
    def unique_rows(self) -> int:
        """Number of distinct rows."""
        return max(self.total_rows - self.duplicate_rows, 0)


@dataclass
class VarianceColumn:
    """Result of a constant / near-constant check on one column."""

    column: str
    n_unique: int
    dominant_value: Any
    dominant_rate: float
    is_constant: bool
    is_near_constant: bool
    note: str


@dataclass
class DtypeIssue:
    """A column whose stored type does not match its content."""

    column: str
    issue: str
    stored_dtype: str
    suggested_dtype: str
    detail: str
    affected_rate: float
    example_values: list[Any] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
@dataclass
class NumericColumnStats:
    """Descriptive statistics for one numeric column."""

    column: str
    count: int
    missing: int
    mean: float | None
    median: float | None
    std: float | None
    minimum: float | None
    q1: float | None
    q3: float | None
    maximum: float | None
    iqr: float | None
    skewness: float | None
    kurtosis: float | None
    n_unique: int
    zero_share: float
    coefficient_of_variation: float | None

    @property
    def is_binary_flag(self) -> bool:
        """True for 0/1 indicator columns, which are handled separately."""
        return self.n_unique == 2

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return to_builtin(asdict(self))


@dataclass
class CategoricalColumnStats:
    """Descriptive statistics for one categorical column."""

    column: str
    count: int
    missing: int
    n_unique: int
    unique_ratio: float
    top_value: Any
    top_count: int
    top_rate: float
    second_value: Any = None
    second_rate: float = 0.0
    rare_categories: int = 0
    entropy: float | None = None
    is_dominant: bool = False
    is_high_cardinality: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return to_builtin(asdict(self))


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
@dataclass
class OutlierResult:
    """Outliers found in one column by one method."""

    column: str
    method: str
    count: int
    rate: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    threshold: float | None = None
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    available: bool = True
    reason: str = ""

    @property
    def n_usable(self) -> int:
        """Number of observations the method was able to inspect."""
        return len(self.indices)


@dataclass
class OutlierSummary:
    """Outlier findings for every numeric column, across all methods."""

    results: list[OutlierResult] = field(default_factory=list)
    columns_flagged: list[str] = field(default_factory=list)
    rows_flagged_by_any: set[int] = field(default_factory=set)

    def by_column(self, column: str) -> list[OutlierResult]:
        """All method results recorded for ``column``."""
        return [r for r in self.results if r.column == column]

    @property
    def total_columns_with_outliers(self) -> int:
        """How many distinct columns contain at least one outlier."""
        return len(self.columns_flagged)


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
@dataclass
class CorrelationPair:
    """A single non-trivial column pair."""

    column_a: str
    column_b: str
    r: float
    abs_r: float
    p_value: float | None = None
    n_observations: int = 0
    reliable: bool = True
    strength: str = ""
    direction: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return to_builtin(asdict(self))


@dataclass
class CorrelationReport:
    """Correlation analysis across numerical variables."""

    method: str
    matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: list[CorrelationPair] = field(default_factory=list)
    strong_positive: list[CorrelationPair] = field(default_factory=list)
    strong_negative: list[CorrelationPair] = field(default_factory=list)
    multicollinear: list[CorrelationPair] = field(default_factory=list)
    columns_analysed: list[str] = field(default_factory=list)
    unavailable_reason: str = ""


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
@dataclass
class AnomalyResult:
    """Isolation Forest output for the dataset."""

    available: bool
    reason: str = ""
    contamination: float = 0.0
    n_rows_scored: int = 0
    n_anomalies: int = 0
    anomaly_rate: float = 0.0
    features_used: list[str] = field(default_factory=list)
    records: pd.DataFrame = field(default_factory=pd.DataFrame)
    top_deviating_features: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def mean_score(self) -> float | None:
        """Mean anomaly score across scored rows (lower = more anomalous)."""
        if self.records.empty or "anomaly_score" not in self.records:
            return None
        return float(self.records["anomaly_score"].mean())


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
@dataclass
class LeakageSignal:
    """One potential-leakage signal raised for a column.

    Signals are *heuristics*: they never assert that leakage exists, only that a
    column is worth investigating before it is used as a model feature.
    """

    column: str
    signal: str
    severity: Severity
    reasoning: str
    recommendation: str
    confidence: str = "low"
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        payload = to_builtin(asdict(self))
        payload["severity"] = self.severity.value
        return payload


@dataclass
class LeakageReport:
    """All leakage signals raised for the dataset."""

    signals: list[LeakageSignal] = field(default_factory=list)
    target: str | None = None
    columns_flagged: list[str] = field(default_factory=list)
    candidate_targets: list[str] = field(default_factory=list)

    @property
    def n_columns_flagged(self) -> int:
        """Number of distinct columns with at least one signal."""
        return len(self.columns_flagged)


# --------------------------------------------------------------------------- #
# Health score
# --------------------------------------------------------------------------- #
@dataclass
class HealthComponent:
    """One measurable dimension of the dataset health score."""

    key: str
    label: str
    score: float            # 0-100 sub-score
    weight: float           # contribution weight, sums to 1.0 across components
    weighted_score: float   # score * weight
    measurement: str        # what was measured, in plain language
    penalty_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return to_builtin(asdict(self))


@dataclass
class HealthScore:
    """Overall dataset health score with a full audit trail."""

    score: float
    grade: str
    components: list[HealthComponent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view."""
        return to_builtin(asdict(self))


# --------------------------------------------------------------------------- #
# Full investigation
# --------------------------------------------------------------------------- #
@dataclass
class InvestigationReport:
    """Everything DATA AUTOPSY learned about one dataset.

    This is the single object the dashboard renders and the report writers turn
    into Markdown / HTML.
    """

    generated_at: str
    overview: DatasetOverview
    missing: MissingReport
    duplicates: DuplicateReport
    variance: list[VarianceColumn]
    dtype_issues: list[DtypeIssue]
    numeric_stats: list[NumericColumnStats]
    categorical_stats: list[CategoricalColumnStats]
    outliers: OutlierSummary
    correlations: CorrelationReport
    anomalies: AnomalyResult
    leakage: LeakageReport
    health: HealthScore
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    analysis_seconds: float = 0.0

    @property
    def dataset_name(self) -> str:
        """Filename of the investigated dataset."""
        return self.overview.filename

    def findings_by_category(self, category: str) -> list[Finding]:
        """Findings belonging to one analysis category."""
        return [f for f in self.findings if f.category == category]

    def critical_findings(self) -> list[Finding]:
        """Findings at high or critical severity."""
        return [f for f in self.findings if f.severity.rank >= Severity.HIGH.rank]
