"""Correlation and association analysis.

Two complementary screens are implemented:

* **Pearson / Spearman correlation** between numerical variables, with pairwise
  complete observations, p-values for the strongest pairs, and an explicit
  no-self-correlation guarantee (the diagonal is never reported as a "finding").
* **Cramer's V** (bias-corrected) with a chi-square test between categorical
  variables, which answers the same question the correlation matrix answers for
  numbers: which features move together?

Correlation is *association*, not causation, and every recommendation in this
module says so.
"""

from __future__ import annotations

import itertools
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src import config
from src.schema import CorrelationPair, CorrelationReport, Finding, Severity
from src.utils import join_names

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


# --------------------------------------------------------------------------- #
# Numerical correlation
# --------------------------------------------------------------------------- #
def correlation_matrix(
    frame: pd.DataFrame, columns: list[str], method: CorrelationMethod = "pearson"
) -> pd.DataFrame:
    """Correlation matrix for ``columns``, tolerating missing values pairwise.

    Returns an empty frame when fewer than two usable columns are available, so
    callers never have to special-case degenerate input.
    """
    usable = [
        c for c in columns
        if c in frame.columns and frame[c].nunique(dropna=True) >= config.MIN_DISTINCT_VALUES
    ]
    if len(usable) < 2:
        return pd.DataFrame()

    numeric = frame[usable].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    try:
        matrix = numeric.corr(method=method)
    except Exception:  # noqa: BLE001 - any scipy failure degrades to "unavailable"
        return pd.DataFrame()
    return matrix.dropna(how="all", axis=0).dropna(how="all", axis=1)


def _pairwise_count(frame: pd.DataFrame, column_a: str, column_b: str) -> int:
    """Number of rows where both columns have a finite value."""
    pair = frame[[column_a, column_b]].apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan)
    return int(pair.dropna().shape[0])


def _pair_p_value(
    frame: pd.DataFrame, column_a: str, column_b: str, method: CorrelationMethod
) -> float | None:
    """Two-sided p-value for a single pair using pairwise complete observations."""
    pair = frame[[column_a, column_b]].apply(pd.to_numeric, errors="coerce")
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if pair.shape[0] < 3:
        return None
    if pair[column_a].nunique() < 2 or pair[column_b].nunique() < 2:
        return None
    try:
        if method == "pearson":
            return float(scipy_stats.pearsonr(pair[column_a], pair[column_b]).pvalue)
        if method == "spearman":
            return float(scipy_stats.spearmanr(pair[column_a], pair[column_b]).pvalue)
        return float(scipy_stats.kendalltau(pair[column_a], pair[column_b]).pvalue)
    except Exception:  # noqa: BLE001
        return None


def strength_label(abs_r: float) -> str:
    """Human label for the magnitude of a correlation coefficient."""
    if abs_r >= config.CORR_NEAR_PERFECT:
        return "near-perfect"
    if abs_r >= config.CORR_STRONG:
        return "strong"
    if abs_r >= config.CORR_MODERATE:
        return "moderate"
    if abs_r >= 0.30:
        return "weak"
    return "very weak"


def analyze_correlations(
    frame: pd.DataFrame,
    columns: list[str],
    method: CorrelationMethod = "pearson",
    max_pairs_for_pvalue: int = config.MAX_PAIRS_FOR_PVALUE,
) -> CorrelationReport:
    """Build the full correlation report for the numeric columns of a frame.

    Self-correlations (``column_a == column_b``) are never emitted: they carry no
    information and would otherwise dominate any "strongest relationship" list.
    """
    matrix = correlation_matrix(frame, columns, method)
    if matrix.empty:
        reason = (
            "At least two numerical columns with more than one distinct value each "
            "are required for correlation analysis."
        )
        return CorrelationReport(method=method, unavailable_reason=reason)

    analysed = [str(c) for c in matrix.columns]
    pairs: list[CorrelationPair] = []

    for column_a, column_b in itertools.combinations(analysed, 2):
        r = matrix.loc[column_a, column_b]
        if pd.isna(r):
            continue
        r = float(r)
        n_obs = _pairwise_count(frame, column_a, column_b)
        pairs.append(
            CorrelationPair(
                column_a=column_a,
                column_b=column_b,
                r=r,
                abs_r=abs(r),
                n_observations=n_obs,
                reliable=n_obs >= config.MIN_PAIRWISE_OBSERVATIONS,
                strength=strength_label(abs(r)),
                direction="positive" if r >= 0 else "negative",
            )
        )

    pairs.sort(key=lambda p: p.abs_r, reverse=True)

    # p-values are expensive, so only the most interesting pairs get them.
    budget = max_pairs_for_pvalue
    for pair in pairs:
        if pair.abs_r < config.CORR_MODERATE or budget <= 0:
            break
        pair.p_value = _pair_p_value(frame, pair.column_a, pair.column_b, method)
        budget -= 1

    strong_positive = [p for p in pairs if p.r >= config.CORR_STRONG]
    strong_negative = [p for p in pairs if p.r <= -config.CORR_STRONG]
    multicollinear = [p for p in pairs if p.abs_r >= config.MULTICOLLINEARITY_FLAG]

    return CorrelationReport(
        method=method,
        matrix=matrix,
        pairs=pairs,
        strong_positive=sorted(strong_positive, key=lambda p: -p.r),
        strong_negative=sorted(strong_negative, key=lambda p: p.r),
        multicollinear=sorted(multicollinear, key=lambda p: -p.abs_r),
        columns_analysed=analysed,
    )


def correlation_findings(report: CorrelationReport) -> list[Finding]:
    """Interpret the correlation report: strongest pairs, redundancy, causation caveat."""
    if report.unavailable_reason:
        return [
            Finding(
                category="correlations",
                title="Correlation analysis not available",
                detail=report.unavailable_reason,
                severity=Severity.INFO,
                evidence={"method": report.method},
            )
        ]

    findings: list[Finding] = []

    if report.strong_positive:
        top = report.strong_positive[:5]
        findings.append(
            Finding(
                category="correlations",
                title=f"{len(report.strong_positive)} strong positive relationship(s) found",
                detail=(
                    f"Using {report.method} correlation across {len(report.columns_analysed)} "
                    "numerical columns: "
                    + "; ".join(
                        f"`{p.column_a}` and `{p.column_b}` move together "
                        f"(r = {p.r:+.3f}, n = {p.n_observations:,})"
                        for p in top
                    )
                    + ". Correlation measures association only — it does not imply that "
                    "one variable causes the other."
                ),
                severity=Severity.MODERATE,
                evidence={
                    "pairs": [
                        {"a": p.column_a, "b": p.column_b, "r": p.r,
                         "n": p.n_observations, "p_value": p.p_value}
                        for p in top
                    ]
                },
                recommendation=(
                    "If both variables are features, one of them may be redundant; if one "
                    "was recorded after the other, consider the direction of the "
                    "relationship before interpreting it causally."
                ),
            )
        )

    if report.strong_negative:
        top = report.strong_negative[:5]
        findings.append(
            Finding(
                category="correlations",
                title=f"{len(report.strong_negative)} strong negative relationship(s) found",
                detail=(
                    "; ".join(
                        f"`{p.column_a}` and `{p.column_b}` move in opposite directions "
                        f"(r = {p.r:+.3f}, n = {p.n_observations:,})"
                        for p in top
                    )
                    + ". A negative coefficient means that higher values in one column "
                    "tend to accompany lower values in the other."
                ),
                severity=Severity.MODERATE,
                evidence={
                    "pairs": [
                        {"a": p.column_a, "b": p.column_b, "r": p.r, "n": p.n_observations}
                        for p in top
                    ]
                },
                recommendation=(
                    "Verify that a monotone relationship is a sensible explanation (e.g. a "
                    "score versus a rank) rather than an artefact of how the data was "
                    "collected."
                ),
            )
        )

    near_duplicates = [p for p in report.pairs if p.abs_r >= config.DUPLICATE_FEATURE_CORRELATION]
    if near_duplicates:
        findings.append(
            Finding(
                category="correlations",
                title=f"{len(near_duplicates)} pair(s) of columns are almost identical",
                detail=(
                    "These pairs correlate at |r| >= "
                    f"{config.DUPLICATE_FEATURE_CORRELATION}: "
                    + "; ".join(
                        f"`{p.column_a}` vs `{p.column_b}` (r = {p.r:+.4f})"
                        for p in near_duplicates[:5]
                    )
                    + ". They are likely the same measurement stored twice, in two units, "
                    "or a rescaled copy."
                ),
                severity=Severity.HIGH,
                evidence={
                    "pairs": [{"a": p.column_a, "b": p.column_b, "r": p.r}
                              for p in near_duplicates]
                },
                recommendation=(
                    "Keep one of each pair. Redundant features split importance between "
                    "them, destabilise linear-model coefficients and duplicate any "
                    "leakage they carry."
                ),
            )
        )

    if not report.pairs:
        findings.append(
            Finding(
                category="correlations",
                title="No comparable numerical pairs",
                detail=(
                    "Fewer than two numerical columns with sufficient variation were "
                    "available, so no pair could be compared."
                ),
                severity=Severity.INFO,
                evidence={"columns_analysed": report.columns_analysed},
            )
        )
    elif max(p.abs_r for p in report.pairs) < 0.30:
        findings.append(
            Finding(
                category="correlations",
                title="No meaningful linear relationships detected",
                detail=(
                    "Every numerical pair has |r| < 0.30. Either the variables are "
                    "genuinely independent, or the relationships are non-linear and "
                    "invisible to a correlation coefficient."
                ),
                severity=Severity.INFO,
                evidence={"max_abs_r": max(p.abs_r for p in report.pairs)},
                recommendation=(
                    "Before concluding independence, inspect scatter plots and consider "
                    "Spearman correlation, mutual information or nonlinear models."
                ),
            )
        )

    weak_evidence = [p for p in report.pairs if p.abs_r >= config.CORR_MODERATE and not p.reliable]
    if weak_evidence:
        findings.append(
            Finding(
                category="correlations",
                title="Some correlations rest on few observations",
                detail=(
                    "Pairwise complete observations fall below "
                    f"{config.MIN_PAIRWISE_OBSERVATIONS} for "
                    + join_names(sorted({p.column_a for p in weak_evidence} | {p.column_b for p in weak_evidence}))
                    + ", so their coefficients are unstable."
                ),
                severity=Severity.LOW,
                evidence={
                    "pairs": [{"a": p.column_a, "b": p.column_b, "n": p.n_observations}
                              for p in weak_evidence]
                },
                recommendation=(
                    "Treat these coefficients as provisional until more complete "
                    "observations are available."
                ),
            )
        )

    return findings


# --------------------------------------------------------------------------- #
# Categorical association (Cramer's V)
# --------------------------------------------------------------------------- #
def cramers_v(table: pd.DataFrame) -> tuple[float, float, int]:
    """Bias-corrected Cramer's V for a contingency table, with chi-square p-value.

    Returns ``(v, p_value, dof)``. ``v`` lies in [0, 1]: 0 means independence,
    1 means one variable is perfectly determined by the other. The bias
    correction (Bergsma) matters for tables with many levels.
    """
    if table.shape[0] < 2 or table.shape[1] < 2:
        return (0.0, 1.0, 0)
    try:
        chi2, p_value, dof, _ = scipy_stats.chi2_contingency(table.to_numpy(), correction=False)
    except Exception:  # noqa: BLE001 - e.g. zero-margin tables
        return (0.0, 1.0, 0)

    n = float(table.to_numpy().sum())
    if n <= 0:
        return (0.0, 1.0, 0)
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corrected = max(
        0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1)
    )
    r_corrected = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corrected = k - ((k - 1) ** 2) / max(n - 1, 1)
    denominator = min(k_corrected - 1, r_corrected - 1)
    if denominator <= 0:
        return (0.0, float(p_value), int(dof))
    return (float(np.sqrt(phi2_corrected / denominator)), float(p_value), int(dof))


def categorical_associations(
    frame: pd.DataFrame,
    categorical_columns: list[str],
    max_columns: int = config.MAX_ASSOCIATION_COLUMNS,
    max_levels: int = config.MAX_ASSOCIATION_LEVELS,
) -> pd.DataFrame:
    """Screen categorical column pairs with bias-corrected Cramer's V."""
    candidates = [
        c for c in categorical_columns
        if c in frame.columns
        and config.MIN_DISTINCT_VALUES <= frame[c].nunique(dropna=True) <= max_levels
    ][:max_columns]

    rows: list[dict[str, object]] = []
    for column_a, column_b in itertools.combinations(candidates, 2):
        working = frame[[column_a, column_b]].dropna()
        if working.shape[0] < config.MIN_PAIRWISE_OBSERVATIONS:
            continue
        table = pd.crosstab(working[column_a], working[column_b])
        v, p_value, dof = cramers_v(table)
        rows.append(
            {
                "column_a": str(column_a),
                "column_b": str(column_b),
                "cramers_v": round(v, 4),
                "p_value": p_value,
                "n_observations": int(working.shape[0]),
                "levels_a": int(table.shape[0]),
                "levels_b": int(table.shape[1]),
                "dof": dof,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["column_a", "column_b", "cramers_v", "p_value", "n_observations"]
        )
    return pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)


def association_findings(associations: pd.DataFrame) -> list[Finding]:
    """Interpret strong categorical associations between features."""
    if associations.empty:
        return []

    strong = associations[associations["cramers_v"] >= config.CRAMERS_V_STRONG]
    if strong.empty:
        return []

    top = strong.head(5)
    pairs_text = "; ".join(
        f"`{row.column_a}` vs `{row.column_b}` (V = {row.cramers_v:.3f})"
        for row in top.itertuples()
    )
    very_strong = strong[strong["cramers_v"] >= config.CRAMERS_V_VERY_STRONG]
    return [
        Finding(
            category="associations",
            title=f"{len(strong)} strongly associated categorical pair(s)",
            detail=(
                f"Cramer's V (bias-corrected) identifies dependence between categorical "
                f"variables: {pairs_text}. V = 0 means independent, V = 1 means one "
                "variable fully determines the other. "
                + (
                    f"{len(very_strong)} pair(s) exceed V = "
                    f"{config.CRAMERS_V_VERY_STRONG}, which usually means one column is "
                    "derived from the other."
                    if not very_strong.empty else ""
                )
            ),
            severity=(
                Severity.MODERATE if not very_strong.empty else Severity.LOW
            ),
            evidence={
                "pairs": [
                    {"a": row.column_a, "b": row.column_b, "v": row.cramers_v,
                     "p_value": row.p_value}
                    for row in top.itertuples()
                ]
            },
            recommendation=(
                "Treat strongly associated categorical pairs as potentially redundant, "
                "and check whether one was derived from the other before using both as "
                "features."
            ),
        )
    ]


def correlation_pairs_table(report: CorrelationReport, limit: int = 25) -> pd.DataFrame:
    """Flat table of the strongest correlations, ready for display or export."""
    if not report.pairs:
        return pd.DataFrame(
            columns=["column_a", "column_b", "r", "abs_r", "strength",
                     "direction", "n_observations", "p_value"]
        )
    rows = []
    for pair in report.pairs[:limit]:
        rows.append(
            {
                "column_a": pair.column_a,
                "column_b": pair.column_b,
                "r": round(pair.r, 4),
                "abs_r": round(pair.abs_r, 4),
                "strength": pair.strength,
                "direction": pair.direction,
                "n_observations": pair.n_observations,
                "p_value": None if pair.p_value is None else float(pair.p_value),
            }
        )
    return pd.DataFrame(rows)
