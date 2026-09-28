"""Data-quality analysis: missingness, duplicates, low variance and dtype drift.

Each public function returns a typed result object for the dashboard *and* a list
of :class:`~src.schema.Finding` objects that carry evidence and a recommended
action. No function raises on unusual input: unsupported cases produce an
explicit "not available" result instead.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.schema import (
    ColumnTypes,
    DtypeIssue,
    DuplicateReport,
    Finding,
    MissingColumn,
    MissingReport,
    Severity,
    VarianceColumn,
)
from src.utils import (
    format_percent,
    is_probably_numeric_text,
    join_names,
    safe_divide,
    truncate_frame,
)


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def analyze_missing(frame: pd.DataFrame) -> MissingReport:
    """Quantify missingness per column and per row."""
    n_rows, n_cols = frame.shape
    if n_rows == 0 or n_cols == 0:
        return MissingReport(
            total_cells=0, missing_cells=0, missing_rate=0.0,
            columns_with_missing=0, complete_rows=0, rows_with_missing=0,
        )

    missing_per_column = frame.isna().sum()
    total_cells = int(n_rows * n_cols)
    missing_cells = int(missing_per_column.sum())

    affected = [
        MissingColumn(
            column=str(column),
            missing=int(count),
            missing_rate=float(count / n_rows),
            dtype=str(frame[column].dtype),
        )
        for column, count in missing_per_column.items()
        if count > 0
    ]
    affected.sort(key=lambda c: c.missing_rate, reverse=True)

    rows_with_missing = int(frame.isna().any(axis=1).sum())

    return MissingReport(
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_rate=safe_divide(missing_cells, total_cells, 0.0),
        affected_columns=affected,
        complete_rows=int(n_rows - rows_with_missing),
        rows_with_missing=rows_with_missing,
        columns_with_missing=len(affected),
    )


def missing_findings(report: MissingReport, n_columns: int, n_rows: int) -> list[Finding]:
    """Turn a missing-value report into ranked, evidence-backed findings."""
    findings: list[Finding] = []
    if report.missing_cells == 0:
        findings.append(
            Finding(
                category="missing_data",
                title="No missing values detected",
                detail=(
                    "Every cell in the dataset is populated. That is unusual for real "
                    "data and can indicate that missing values were already imputed, "
                    "encoded as sentinel values (0, -1, 9999, 'N/A'), or that the "
                    "extract was filtered."
                ),
                severity=Severity.INFO,
                evidence={"missing_cells": 0, "total_cells": report.total_cells},
                recommendation=(
                    "Check whether missing values were encoded as sentinel values "
                    "before treating this dataset as complete."
                ),
            )
        )
        return findings

    findings.append(
        Finding(
            category="missing_data",
            title=(
                f"{report.missing_cells:,} missing values across "
                f"{report.columns_with_missing} column(s)"
            ),
            detail=(
                f"{format_percent(report.missing_rate)} of all cells are missing "
                f"({report.complete_rows:,} of {report.complete_rows + report.rows_with_missing:,} "
                "rows are complete)."
            ),
            severity=_missing_severity(report.missing_rate),
            evidence={
                "missing_cells": report.missing_cells,
                "missing_rate": report.missing_rate,
                "columns_with_missing": report.columns_with_missing,
                "complete_rows": report.complete_rows,
            },
            recommendation=(
                "Before imputing, establish whether values are missing completely at "
                "random (MCAR), at random (MAR) or not at random (MNAR) — the "
                "mechanism determines whether imputation is even valid."
            ),
        )
    )

    for column in report.affected_columns[: config.MAX_EXAMPLE_ROWS]:
        severity = _missing_severity(column.missing_rate)
        if severity.rank < Severity.MODERATE.rank:
            continue
        findings.append(
            Finding(
                category="missing_data",
                title=f"Column '{column.column}' is {format_percent(column.missing_rate)} missing",
                detail=(
                    f"{column.missing:,} of {n_rows:,} values are missing in "
                    f"'{column.column}' (dtype {column.dtype})."
                ),
                severity=severity,
                evidence={
                    "column": column.column,
                    "missing": column.missing,
                    "missing_rate": column.missing_rate,
                },
                recommendation=(
                    "Investigate why this variable is missing before applying "
                    "imputation: if missingness correlates with the target, dropping "
                    "or imputing it can bias the model."
                    if column.missing_rate < config.MISSING_CRITICAL
                    else "This column is so sparse that it may be better to drop it "
                    "than to impute the majority of its values."
                ),
            )
        )

    return findings


def _missing_severity(rate: float) -> Severity:
    """Map a missing rate onto a severity band."""
    if rate >= config.MISSING_CRITICAL:
        return Severity.CRITICAL
    if rate >= config.MISSING_HIGH:
        return Severity.HIGH
    if rate >= config.MISSING_MODERATE:
        return Severity.MODERATE
    if rate >= config.MISSING_LOW:
        return Severity.LOW
    return Severity.INFO


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def analyze_duplicates(frame: pd.DataFrame, subset: list[str] | None = None) -> DuplicateReport:
    """Count exact duplicate rows and collect representative examples."""
    n_rows = len(frame)
    if n_rows == 0:
        return DuplicateReport(total_rows=0, duplicate_rows=0, duplicate_rate=0.0)

    if subset:
        subset = [c for c in subset if c in frame.columns]
        mask = frame.duplicated(subset=subset, keep="first")
    else:
        mask = frame.duplicated(keep="first")

    duplicate_rows = int(mask.sum())
    examples = pd.DataFrame()

    if duplicate_rows > 0:
        duplicated_mask = frame.duplicated(keep=False)
        group_columns = subset or list(frame.columns)
        duplicated_rows = frame[duplicated_mask]
        examples, _ = truncate_frame(
            duplicated_rows.sort_values(by=group_columns, kind="stable"),
            config.MAX_EXAMPLE_ROWS,
        )
        try:
            duplicate_group_count = int(
                duplicated_rows.groupby(group_columns, dropna=False, observed=True).ngroups
            )
        except Exception:  # pragma: no cover - defensive: unhashable values
            duplicate_group_count = 0
    else:
        duplicate_group_count = 0

    return DuplicateReport(
        total_rows=n_rows,
        duplicate_rows=duplicate_rows,
        duplicate_rate=safe_divide(duplicate_rows, n_rows, 0.0),
        examples=examples.reset_index(drop=True),
        duplicate_group_count=duplicate_group_count,
    )


def duplicate_findings(report: DuplicateReport) -> list[Finding]:
    """Explain duplicate records, including why they may be legitimate."""
    if report.duplicate_rows == 0:
        return [
            Finding(
                category="duplicates",
                title="No exact duplicate rows detected",
                detail=f"All {report.total_rows:,} rows are distinct.",
                severity=Severity.INFO,
                evidence={"duplicate_rows": 0, "duplicate_rate": 0.0},
                recommendation=(
                    "Also consider near-duplicates: identical rows after ignoring one "
                    "or two columns are not caught by an exact check."
                ),
            )
        ]

    severity = Severity.INFO
    if report.duplicate_rate >= config.DUPLICATE_HIGH:
        severity = Severity.HIGH
    elif report.duplicate_rate >= config.DUPLICATE_MODERATE:
        severity = Severity.MODERATE
    elif report.duplicate_rate >= config.DUPLICATE_LOW:
        severity = Severity.LOW

    return [
        Finding(
            category="duplicates",
            title=(
                f"{report.duplicate_rows:,} exact duplicate row(s) "
                f"({format_percent(report.duplicate_rate)})"
            ),
            detail=(
                f"{report.duplicate_rows:,} of {report.total_rows:,} rows repeat an "
                f"earlier row across all columns, forming {report.duplicate_group_count:,} "
                "duplicate group(s)."
            ),
            severity=severity,
            evidence={
                "duplicate_rows": report.duplicate_rows,
                "duplicate_rate": report.duplicate_rate,
                "duplicate_groups": report.duplicate_group_count,
                "unique_rows": report.unique_rows,
            },
            recommendation=(
                "Decide deliberately: genuine repeat observations (e.g. two identical "
                "transactions) must stay, while export or join artefacts should be "
                "de-duplicated. Duplicates inflate row counts and distort statistics."
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# Constant / near-constant features
# --------------------------------------------------------------------------- #
def analyze_variance(frame: pd.DataFrame, column_types: ColumnTypes | None = None) -> list[VarianceColumn]:
    """Detect constant and near-constant columns.

    A column that never (or almost never) changes cannot separate observations,
    which makes it useless for prediction and often signals a broken export.
    """
    results: list[VarianceColumn] = []
    n_rows = len(frame)
    if n_rows == 0:
        return results

    candidates = column_types.all_columns if column_types else [str(c) for c in frame.columns]

    for column in candidates:
        if column not in frame.columns:
            continue
        series = frame[column]
        non_null = series.dropna()
        if non_null.empty:
            results.append(
                VarianceColumn(
                    column=str(column),
                    n_unique=0,
                    dominant_value=None,
                    dominant_rate=1.0,
                    is_constant=True,
                    is_near_constant=True,
                    note="Column is entirely empty (no non-missing values).",
                )
            )
            continue

        counts = non_null.value_counts(dropna=True)
        n_unique = int(counts.size)
        dominant_value = counts.index[0]
        dominant_count = int(counts.iloc[0])
        dominant_rate = safe_divide(dominant_count, len(non_null), 0.0)

        is_constant = n_unique == 1
        is_near_constant = (not is_constant) and dominant_rate >= config.NEAR_CONSTANT_DOMINANCE

        note = ""
        missing_rate = safe_divide(series.isna().sum(), n_rows, 0.0)
        if is_constant:
            note = (
                "Every observed value is identical; the column carries no "
                "discriminative information."
            )
        elif is_near_constant:
            note = (
                f"{format_percent(dominant_rate)} of non-missing values equal "
                f"'{dominant_value}'; the minority values are too rare to support "
                "a stable split."
            )
        if missing_rate >= config.MISSING_HIGH:
            note = (note + " " if note else "") + (
                f"Additionally {format_percent(missing_rate)} of the column is missing, "
                "which reduces its information content further."
            )

        if is_constant or is_near_constant:
            results.append(
                VarianceColumn(
                    column=str(column),
                    n_unique=n_unique,
                    dominant_value=dominant_value,
                    dominant_rate=dominant_rate,
                    is_constant=is_constant,
                    is_near_constant=is_near_constant,
                    note=note,
                )
            )

    results.sort(key=lambda r: (not r.is_constant, -r.dominant_rate))
    return results


def variance_findings(results: list[VarianceColumn], n_columns: int) -> list[Finding]:
    """Explain the impact of constant / near-constant columns."""
    if not results:
        return [
            Finding(
                category="low_variance",
                title="No constant or near-constant columns",
                detail=(
                    "Every column shows meaningful variation across observations "
                    f"(threshold: a single value covering "
                    f"{format_percent(config.NEAR_CONSTANT_DOMINANCE)}+ of non-missing rows)."
                ),
                severity=Severity.INFO,
                evidence={"columns_flagged": 0},
            )
        ]

    constant = [r for r in results if r.is_constant]
    near_constant = [r for r in results if r.is_near_constant]
    findings: list[Finding] = []

    if constant:
        findings.append(
            Finding(
                category="low_variance",
                title=f"{len(constant)} constant column(s) detected",
                detail=(
                    f"{join_names([r.column for r in constant])} have a single distinct "
                    "value. They cannot separate observations and only add noise and "
                    "dimensionality to a model."
                ),
                severity=Severity.MODERATE,
                evidence={
                    "columns": [r.column for r in constant],
                    "values": {r.column: str(r.dominant_value) for r in constant},
                },
                recommendation=(
                    "Drop constant columns before modelling; keep them only if the "
                    "constant value itself is a documented fact about the extract "
                    "(e.g. a single site or a fixed date range)."
                ),
            )
        )

    if near_constant:
        findings.append(
            Finding(
                category="low_variance",
                title=f"{len(near_constant)} near-constant column(s) detected",
                detail=(
                    f"{join_names([r.column for r in near_constant])} are dominated by a "
                    f"single value (>= {format_percent(config.NEAR_CONSTANT_DOMINANCE)} of "
                    "non-missing rows). Their variance is close to zero, so models can "
                    "rarely learn a stable relationship from them."
                ),
                severity=Severity.LOW,
                evidence={
                    "columns": [r.column for r in near_constant],
                    "dominant_rates": {r.column: r.dominant_rate for r in near_constant},
                },
                recommendation=(
                    "Consider dropping these features or merging the rare categories "
                    "into an 'other' group. Limited predictive usefulness is not the "
                    "same as being wrong — verify with domain knowledge first."
                ),
            )
        )

    flagged = len(constant) + len(near_constant)
    if n_columns and flagged / n_columns >= 0.25:
        findings.append(
            Finding(
                category="low_variance",
                title=f"{format_percent(flagged / n_columns)} of columns have almost no variance",
                detail=(
                    f"{flagged} of {n_columns} columns are constant or near-constant, which "
                    "is a strong sign that the export was filtered too narrowly, that "
                    "several columns were derived from the same field, or that the wrong "
                    "table was extracted."
                ),
                severity=Severity.HIGH,
                evidence={"flagged": flagged, "total_columns": n_columns},
                recommendation=(
                    "Review how this dataset was extracted before investing in "
                    "modelling: a quarter of the feature space carries no signal."
                ),
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Data-type / consistency issues
# --------------------------------------------------------------------------- #
def dtype_issue_findings(issues: list[DtypeIssue], n_columns: int) -> list[Finding]:
    """Turn detected dtype problems into findings and recommendations."""
    if not issues:
        return [
            Finding(
                category="dtypes",
                title="Stored data types match column content",
                detail=(
                    "No numeric-in-text, datetime-in-text or mixed-format columns were "
                    "detected."
                ),
                severity=Severity.INFO,
                evidence={"issues": 0},
            )
        ]

    findings: list[Finding] = []
    by_kind: dict[str, list[DtypeIssue]] = {}
    for issue in issues:
        by_kind.setdefault(issue.issue, []).append(issue)

    titles = {
        "numeric_stored_as_text": "Numbers stored as text — excluded from all numerical analysis",
        "datetime_stored_as_text": "Dates stored as text — time-series analysis is disabled",
        "mixed_types": "Columns mixing numeric and text values",
    }
    recommendations = {
        "numeric_stored_as_text": (
            "Convert these columns with str.replace(',', '').astype(float) (or the "
            "equivalent cleanup for your locale) so they enter the statistical analysis."
        ),
        "datetime_stored_as_text": (
            "Parse these columns with pd.to_datetime so trends, seasonality and time "
            "splits become available."
        ),
        "mixed_types": (
            "Standardise the value format at the source: mixed entry formats usually "
            "indicate a keying or integration problem upstream."
        ),
    }
    severities = {
        "numeric_stored_as_text": Severity.HIGH,
        "datetime_stored_as_text": Severity.MODERATE,
        "mixed_types": Severity.MODERATE,
    }

    for kind, group in by_kind.items():
        findings.append(
            Finding(
                category="dtypes",
                title=titles.get(kind, f"Data-type issue: {kind}"),
                detail=" ".join(issue.detail for issue in group[:3])
                + (" ..." if len(group) > 3 else ""),
                severity=severities.get(kind, Severity.LOW),
                evidence={
                    "issue": kind,
                    "columns": [i.column for i in group],
                    "examples": {i.column: i.example_values for i in group[:5]},
                },
                recommendation=recommendations.get(kind, "Review the column encoding."),
            )
        )

    return findings


def looks_like_number_text(series: pd.Series) -> bool:
    """Public wrapper used by tests and the UI for the numeric-text heuristic."""
    return is_probably_numeric_text(series)
