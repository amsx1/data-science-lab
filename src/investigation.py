"""Investigation orchestration — the single entry point of the analysis engine.

``run_investigation`` chains every analysis module in a fixed, reproducible order
and assembles one :class:`~src.schema.InvestigationReport`. The dashboard, the CLI
demo and the tests all call this same function, so what you see in the UI is
exactly what the tests exercise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from src import (
    anomalies as anomaly_module,
    categorical as categorical_module,
    correlations as correlation_module,
    data_loader,
    health as health_module,
    leakage as leakage_module,
    outliers as outlier_module,
    quality,
    statistics as statistics_module,
)
from src import config
from src.schema import (
    AnomalyResult,
    CorrelationReport,
    DuplicateReport,
    DtypeIssue,
    Finding,
    InvestigationReport,
    LeakageReport,
    MissingReport,
    OutlierSummary,
    Severity,
    VarianceColumn,
    sort_findings,
)

ProgressCallback = Callable[[str, float], None]


@dataclass
class InvestigationOptions:
    """User-configurable knobs of an investigation."""

    contamination: float = config.ANOMALY_DEFAULT_CONTAMINATION
    correlation_method: str = "pearson"
    include_modified_zscore: bool = True
    leakage_target: str | None = None
    enable_anomaly_detection: bool = True
    enable_leakage_screening: bool = True
    n_estimators: int = config.ANOMALY_DEFAULT_ESTIMATORS

    def to_dict(self) -> dict[str, object]:
        """Plain-dictionary view, used in the exported report header."""
        return {
            "contamination": self.contamination,
            "correlation_method": self.correlation_method,
            "include_modified_zscore": self.include_modified_zscore,
            "leakage_target": self.leakage_target,
            "enable_anomaly_detection": self.enable_anomaly_detection,
            "enable_leakage_screening": self.enable_leakage_screening,
            "n_estimators": self.n_estimators,
        }


def _notify(progress: ProgressCallback | None, label: str, fraction: float) -> None:
    """Report progress without letting a UI callback break the analysis."""
    if progress is None:
        return
    try:
        progress(label, fraction)
    except Exception:  # noqa: BLE001 - the UI must never break the engine
        pass


def run_investigation(
    frame: pd.DataFrame,
    filename: str = "dataset.csv",
    options: InvestigationOptions | None = None,
    load_warnings: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> InvestigationReport:
    """Run the complete DATA AUTOPSY pipeline over ``frame``.

    Args:
        frame: The dataset to investigate (already loaded and cleaned).
        filename: Display name of the dataset.
        options: Analysis configuration; defaults are used when omitted.
        load_warnings: Messages produced by the loader, surfaced in the report.
        progress: Optional ``(label, fraction)`` callback for progress reporting.

    Returns:
        A fully populated :class:`InvestigationReport`.
    """
    options = options or InvestigationOptions()
    started = time.perf_counter()

    # --- 1. structure -------------------------------------------------------
    _notify(progress, "Profiling dataset structure", 0.05)
    overview = data_loader.build_overview(frame, filename, load_warnings)
    column_types = overview.column_types

    # --- 2. quality ---------------------------------------------------------
    _notify(progress, "Analysing missing values", 0.12)
    missing: MissingReport = quality.analyze_missing(frame)

    _notify(progress, "Detecting duplicate records", 0.20)
    duplicates: DuplicateReport = quality.analyze_duplicates(frame)

    _notify(progress, "Checking constant and low-variance columns", 0.28)
    variance: list[VarianceColumn] = quality.analyze_variance(frame, column_types)

    _notify(progress, "Inspecting stored data types", 0.34)
    dtype_issues: list[DtypeIssue] = data_loader.detect_dtype_issues(frame, column_types)

    # --- 3. statistics ------------------------------------------------------
    _notify(progress, "Computing descriptive statistics", 0.45)
    numeric_stats = statistics_module.analyze_numeric(frame, column_types.numeric)
    categorical_stats = categorical_module.analyze_categorical(
        frame, column_types.categorical + column_types.boolean
    )

    # --- 4. outliers --------------------------------------------------------
    _notify(progress, "Testing for outliers (IQR, z-score, MAD)", 0.58)
    outliers: OutlierSummary = outlier_module.analyze_outliers(
        frame, column_types.numeric,
        include_modified_zscore=options.include_modified_zscore,
    )

    # --- 5. relationships ---------------------------------------------------
    _notify(progress, "Correlating numerical variables", 0.70)
    correlations: CorrelationReport = correlation_module.analyze_correlations(
        frame, column_types.numeric, method=options.correlation_method  # type: ignore[arg-type]
    )
    associations = correlation_module.categorical_associations(
        frame, column_types.categorical + column_types.boolean
    )

    # --- 6. anomalies -------------------------------------------------------
    if options.enable_anomaly_detection:
        _notify(progress, "Screening for multivariate anomalies", 0.82)
        anomaly_result: AnomalyResult = anomaly_module.detect_anomalies(
            frame,
            column_types.numeric,
            contamination=options.contamination,
            n_estimators=options.n_estimators,
        )
    else:
        anomaly_result = AnomalyResult(
            available=False,
            reason="Anomaly detection was disabled in the sidebar.",
            contamination=options.contamination,
        )

    # --- 7. leakage ---------------------------------------------------------
    if options.enable_leakage_screening:
        _notify(progress, "Assessing potential data leakage", 0.90)
        leakage: LeakageReport = leakage_module.analyze_leakage(
            frame, column_types, target=options.leakage_target,
            correlation_method="spearman",
        )
    else:
        leakage = LeakageReport(
            signals=[],
            target=None,
            columns_flagged=[],
            candidate_targets=[],
        )

    # --- 8. health score ----------------------------------------------------
    _notify(progress, "Scoring dataset health", 0.96)
    numeric_cells = sum(s.count for s in numeric_stats)
    health = health_module.compute_health_score(
        missing=missing,
        duplicates=duplicates,
        variance=variance,
        dtype_issues=dtype_issues,
        outliers=outliers,
        n_columns=overview.n_columns,
        numeric_cells=numeric_cells,
    )

    # --- 9. findings and recommendations ------------------------------------
    findings: list[Finding] = []
    findings += quality.missing_findings(missing, overview.n_columns, overview.n_rows)
    findings += quality.duplicate_findings(duplicates)
    findings += quality.variance_findings(variance, overview.n_columns)
    findings += quality.dtype_issue_findings(dtype_issues, overview.n_columns)
    findings += statistics_module.numeric_findings(numeric_stats)
    findings += categorical_module.categorical_findings(
        categorical_stats, overview.n_columns
    )
    findings += outlier_module.outlier_findings(outliers, column_types.numeric)
    findings += correlation_module.correlation_findings(correlations)
    findings += correlation_module.association_findings(associations)
    findings += anomaly_module.anomaly_findings(anomaly_result)
    if options.enable_leakage_screening:
        findings += leakage_module.leakage_findings(leakage)

    findings = sort_findings(findings)
    recommendations = build_recommendations(findings, overview.n_rows, overview.n_columns)

    _notify(progress, "Finalising report", 1.0)

    return InvestigationReport(
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        overview=overview,
        missing=missing,
        duplicates=duplicates,
        variance=variance,
        dtype_issues=dtype_issues,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        outliers=outliers,
        correlations=correlations,
        anomalies=anomaly_result,
        leakage=leakage,
        health=health,
        findings=findings,
        recommendations=recommendations,
        analysis_seconds=round(time.perf_counter() - started, 2),
    )


def build_recommendations(
    findings: list[Finding], n_rows: int, n_columns: int, limit: int = 8
) -> list[str]:
    """Derive an ordered, de-duplicated action list from the findings.

    Recommendations are taken verbatim from the findings that produced them, so
    every line of advice is traceable to a measurement. Highest severity first.
    """
    recommendations: list[str] = []
    seen: set[str] = set()

    for finding in findings:
        if finding.severity.rank < Severity.LOW.rank or not finding.recommendation:
            continue
        text = f"{finding.title}: {finding.recommendation}"
        fingerprint = finding.recommendation.strip().lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        recommendations.append(text)
        if len(recommendations) >= limit:
            break

    if not recommendations:
        recommendations.append(
            "No corrective action is required by the automated checks. Document the "
            "dataset's provenance, keep the analysis reproducible, and validate the "
            "data against its source system before modelling."
        )

    if n_rows < 50:
        recommendations.append(
            f"The dataset has only {n_rows:,} rows: statistical estimates (correlation, "
            "skewness, anomaly scores) are noisy at this size, so treat them as "
            "indicative rather than conclusive."
        )
    if n_columns < 3:
        recommendations.append(
            f"With only {n_columns} column(s), multivariate techniques (anomaly "
            "detection, feature-relationship screening) have very little to work with."
        )
    return recommendations
