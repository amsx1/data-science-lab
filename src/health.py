"""Dataset health score.

The score is a transparent, fully documented weighted average of five measurable
dimensions. It contains no machine learning, no randomness and no hidden
heuristics: given the same dataset it always produces the same number, and every
point that was deducted can be traced back to a measurement shown in the UI.

Dimensions and weights (see :mod:`src.config`):

===============  ======  =====================================================
Dimension        Weight  Measurement
===============  ======  =====================================================
completeness     0.30    share of missing cells + penalty for >50% missing cols
integrity        0.20    share of numeric cells flagged by the IQR outlier rule
consistency      0.20    share of columns whose stored dtype hides their content
uniqueness       0.15    share of exact duplicate rows
variability      0.15    share of constant / near-constant columns
===============  ======  =====================================================

Sub-scores are linear in the measured rate and saturate at a fixed threshold, so
the score is monotone: making a dataset cleaner can never lower it.

Potential-leakage signals are deliberately **excluded** from the score because
they are name- and heuristic-based; they are reported separately as risks.
"""

from __future__ import annotations

from src import config
from src.schema import (
    DtypeIssue,
    HealthComponent,
    HealthScore,
    MissingReport,
    DuplicateReport,
    OutlierSummary,
    VarianceColumn,
)
from src.utils import clamp, rescale


def _score_from_rate(rate: float, saturation: float) -> float:
    """Linear 0-100 sub-score: 100 when rate is 0, 0 once rate reaches saturation."""
    return 100.0 * (1.0 - rescale(rate, 0.0, saturation))


def completeness_component(missing: MissingReport, n_columns: int) -> HealthComponent:
    """Sub-score for missing data."""
    base = _score_from_rate(missing.missing_rate, config.COMPLETENESS_SATURATION_RATE)
    severe = [c for c in missing.affected_columns if c.missing_rate >= config.MISSING_HIGH]
    column_penalty = 5.0 * len(severe)
    score = clamp(base - column_penalty, 0.0, 100.0)

    detail = (
        f"{missing.missing_rate:.2%} of cells missing "
        f"(sub-score reaches 0 at {config.COMPLETENESS_SATURATION_RATE:.0%})"
    )
    reason = ""
    if severe:
        reason = (
            f"-{column_penalty:.0f} points: {len(severe)} column(s) are "
            f">{config.MISSING_HIGH:.0%} missing ({', '.join(c.column for c in severe[:4])})"
        )
    return HealthComponent(
        key="completeness",
        label="Completeness",
        score=round(score, 1),
        weight=config.HEALTH_WEIGHTS["completeness"],
        weighted_score=round(score * config.HEALTH_WEIGHTS["completeness"], 2),
        measurement=detail,
        penalty_reason=reason,
    )


def integrity_component(outliers: OutlierSummary, numeric_cells: int) -> HealthComponent:
    """Sub-score for extreme values, measured with the IQR rule (robust to masking)."""
    weight = config.HEALTH_WEIGHTS["integrity"]
    if numeric_cells <= 0:
        return HealthComponent(
            key="integrity",
            label="Integrity",
            score=100.0,
            weight=weight,
            weighted_score=100.0 * weight,
            measurement="Not applicable: the dataset contains no numeric cells.",
        )

    iqr_cells = sum(r.count for r in outliers.results if r.method == "iqr")
    cell_rate = iqr_cells / numeric_cells
    score = _score_from_rate(cell_rate, config.INTEGRITY_SATURATION_RATE)

    return HealthComponent(
        key="integrity",
        label="Integrity",
        score=round(score, 1),
        weight=weight,
        weighted_score=round(score * weight, 2),
        measurement=(
            f"{cell_rate:.2%} of numeric cells fall outside 1.5xIQR "
            f"(sub-score reaches 0 at {config.INTEGRITY_SATURATION_RATE:.0%}); "
            f"{outliers.total_columns_with_outliers} column(s) affected"
        ),
        penalty_reason=(
            "Outliers are not automatically errors — they are extreme but valid "
            "observations that inflate spread and distort means."
            if iqr_cells else ""
        ),
    )


def consistency_component(dtype_issues: list[DtypeIssue], n_columns: int) -> HealthComponent:
    """Sub-score for stored-type problems (numbers/dates hidden inside text)."""
    weight = config.HEALTH_WEIGHTS["consistency"]
    if n_columns <= 0:
        return HealthComponent(
            key="consistency", label="Consistency", score=100.0, weight=weight,
            weighted_score=100.0 * weight,
            measurement="Not applicable: no columns to check.",
        )

    flagged = len({issue.column for issue in dtype_issues})
    rate = flagged / n_columns
    score = 100.0 * (1.0 - clamp(rate, 0.0, 1.0))
    return HealthComponent(
        key="consistency",
        label="Consistency",
        score=round(score, 1),
        weight=weight,
        weighted_score=round(score * weight, 2),
        measurement=(
            f"{flagged} of {n_columns} column(s) store values in a type that hides "
            "their content (numeric text, datetime text, mixed formats)"
        ),
        penalty_reason=(
            "-{:.0f} points: {}".format(
                score and (100.0 - score) or 0.0,
                ", ".join(sorted({i.column for i in dtype_issues})[:4]),
            )
            if flagged else ""
        ),
    )


def uniqueness_component(duplicates: DuplicateReport) -> HealthComponent:
    """Sub-score for duplicate rows."""
    weight = config.HEALTH_WEIGHTS["uniqueness"]
    score = _score_from_rate(duplicates.duplicate_rate, config.UNIQUENESS_SATURATION_RATE)
    return HealthComponent(
        key="uniqueness",
        label="Uniqueness",
        score=round(score, 1),
        weight=weight,
        weighted_score=round(score * weight, 2),
        measurement=(
            f"{duplicates.duplicate_rate:.2%} of rows are exact duplicates "
            f"(sub-score reaches 0 at {config.UNIQUENESS_SATURATION_RATE:.0%})"
        ),
        penalty_reason=(
            f"-{100.0 - score:.1f} points: {duplicates.duplicate_rows:,} duplicate row(s)"
            if duplicates.duplicate_rows else ""
        ),
    )


def variability_component(variance: list[VarianceColumn], n_columns: int) -> HealthComponent:
    """Sub-score for constant / near-constant columns."""
    weight = config.HEALTH_WEIGHTS["variability"]
    if n_columns <= 0:
        return HealthComponent(
            key="variability", label="Variability", score=100.0, weight=weight,
            weighted_score=100.0 * weight,
            measurement="Not applicable: no columns to check.",
        )

    constant = [v for v in variance if v.is_constant]
    near = [v for v in variance if v.is_near_constant and not v.is_constant]
    rate = (len(constant) + 0.5 * len(near)) / n_columns
    score = 100.0 * (1.0 - clamp(rate, 0.0, 1.0))

    reason_parts = []
    if constant:
        reason_parts.append(f"{len(constant)} constant")
    if near:
        reason_parts.append(f"{len(near)} near-constant")
    return HealthComponent(
        key="variability",
        label="Variability",
        score=round(score, 1),
        weight=weight,
        weighted_score=round(score * weight, 2),
        measurement=(
            f"{len(constant)} constant and {len(near)} near-constant column(s) out of "
            f"{n_columns} (near-constant counts half a penalty)"
        ),
        penalty_reason=("-{:.1f} points: {}".format(100.0 - score, " + ".join(reason_parts))
                        if reason_parts else ""),
    )


def _grade_for(score: float) -> str:
    """Map a 0-100 score onto a qualitative grade band."""
    for threshold, label in config.HEALTH_GRADES:
        if score >= threshold:
            return label
    return config.HEALTH_GRADES[-1][1]


def compute_health_score(
    missing: MissingReport,
    duplicates: DuplicateReport,
    variance: list[VarianceColumn],
    dtype_issues: list[DtypeIssue],
    outliers: OutlierSummary,
    n_columns: int,
    numeric_cells: int,
) -> HealthScore:
    """Combine the five measurable dimensions into one auditable score."""
    components = [
        completeness_component(missing, n_columns),
        integrity_component(outliers, numeric_cells),
        consistency_component(dtype_issues, n_columns),
        uniqueness_component(duplicates),
        variability_component(variance, n_columns),
    ]
    total = sum(component.weighted_score for component in components)
    total = round(clamp(total, 0.0, 100.0), 1)

    notes = [
        "The score is a weighted average of five measured dimensions — no machine "
        "learning, no randomness, no hidden judgement.",
        "Potential-leakage signals are excluded from the score because they are "
        "name-based heuristics and cannot be verified automatically.",
        "A high score means the dataset is *clean*, not that it is *useful*: a "
        "dataset with excellent hygiene can still lack predictive signal.",
    ]

    return HealthScore(
        score=total,
        grade=_grade_for(total),
        components=components,
        notes=notes,
    )
