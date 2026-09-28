"""Descriptive statistics for numerical columns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schema import Finding, NumericColumnStats, Severity
from src.utils import finite_values, format_number, is_integer_valued, join_names, safe_divide


def describe_numeric_column(series: pd.Series, column: str) -> NumericColumnStats:
    """Compute descriptive statistics for a single numeric column.

    Non-finite values (``inf``, ``-inf``) are excluded and reported through the
    ``count``/``missing`` fields so totals stay reconcilable with the row count.
    """
    total = len(series)
    values = finite_values(series)
    n = int(values.size)
    missing = int(total - n)

    if n == 0:
        return NumericColumnStats(
            column=column, count=0, missing=missing, mean=None, median=None, std=None,
            minimum=None, q1=None, q3=None, maximum=None, iqr=None, skewness=None,
            kurtosis=None, n_unique=0, zero_share=0.0, coefficient_of_variation=None,
        )

    q1, q3 = np.percentile(values, [25, 75])
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(values))

    return NumericColumnStats(
        column=column,
        count=n,
        missing=missing,
        mean=mean,
        median=float(np.median(values)),
        std=std,
        minimum=float(np.min(values)),
        q1=float(q1),
        q3=float(q3),
        maximum=float(np.max(values)),
        iqr=float(q3 - q1),
        # Skew/kurtosis need at least 3 / 4 observations to be defined.
        skewness=float(pd.Series(values).skew()) if n >= 3 else None,
        kurtosis=float(pd.Series(values).kurtosis()) if n >= 4 else None,
        n_unique=int(np.unique(values).size),
        zero_share=safe_divide(int(np.count_nonzero(values == 0)), n, 0.0),
        coefficient_of_variation=(
            float(std / abs(mean)) if mean not in (0.0,) and np.isfinite(std) else None
        ),
    )


def analyze_numeric(frame: pd.DataFrame, numeric_columns: list[str]) -> list[NumericColumnStats]:
    """Describe every numeric column, skipping ones that cannot be analysed."""
    stats: list[NumericColumnStats] = []
    for column in numeric_columns:
        if column not in frame.columns:
            continue
        try:
            stats.append(describe_numeric_column(frame[column], str(column)))
        except Exception:  # noqa: BLE001 - never let one odd column kill the run
            continue
    return stats


def numeric_findings(stats: list[NumericColumnStats]) -> list[Finding]:
    """Report distribution shape (skew, kurtosis), zero-inflation and scaling needs."""
    if not stats:
        return [
            Finding(
                category="numerical",
                title="No numerical columns available",
                detail=(
                    "The dataset contains no numeric columns, so distribution, outlier "
                    "and correlation analyses were skipped. Check whether numeric "
                    "values were stored as text."
                ),
                severity=Severity.INFO,
                evidence={"numeric_columns": 0},
            )
        ]

    findings: list[Finding] = []

    skewed = [s for s in stats if s.skewness is not None and abs(s.skewness) >= config.SKEW_THRESHOLD]
    if skewed:
        skewed_sorted = sorted(skewed, key=lambda s: abs(s.skewness or 0), reverse=True)
        findings.append(
            Finding(
                category="numerical",
                title=f"{len(skewed)} numerical variable(s) are clearly skewed",
                detail=(
                    "Skewness above "
                    f"{config.SKEW_THRESHOLD:.1f} means the mean is pulled away from the "
                    "median by a long tail. Most affected: "
                    + ", ".join(
                        f"`{s.column}` (skew {s.skewness:+.2f})" for s in skewed_sorted[:4]
                    )
                    + "."
                ),
                severity=Severity.MODERATE if len(skewed) > len(stats) / 3 else Severity.LOW,
                evidence={
                    "columns": [s.column for s in skewed],
                    "skewness": {s.column: s.skewness for s in skewed},
                },
                recommendation=(
                    "For linear models and distance-based algorithms consider a log1p "
                    "or Box-Cox/Yeo-Johnson transform; tree-based models are insensitive "
                    "to monotone transforms."
                ),
            )
        )

    heavy_tailed = [
        s for s in stats if s.kurtosis is not None and s.kurtosis >= config.KURTOSIS_THRESHOLD
    ]
    if heavy_tailed:
        findings.append(
            Finding(
                category="numerical",
                title=f"{len(heavy_tailed)} variable(s) have heavy tails (high kurtosis)",
                detail=(
                    "Excess kurtosis above "
                    f"{config.KURTOSIS_THRESHOLD:.1f} indicates frequent extreme values "
                    "relative to a normal distribution, which makes the mean and standard "
                    "deviation poor summaries. Affected: "
                    + join_names([s.column for s in heavy_tailed]) + "."
                ),
                severity=Severity.LOW,
                evidence={"columns": [s.column for s in heavy_tailed],
                          "kurtosis": {s.column: s.kurtosis for s in heavy_tailed}},
                recommendation=(
                    "Prefer median and IQR when describing these columns, and review the "
                    "extreme observations with the outlier and anomaly sections."
                ),
            )
        )

    zero_inflated = [s for s in stats if s.zero_share >= 0.30 and s.n_unique > 2]
    if zero_inflated:
        findings.append(
            Finding(
                category="numerical",
                title=f"{len(zero_inflated)} variable(s) are zero-inflated",
                detail=(
                    "At least 30% of the non-missing values are exactly zero in "
                    + join_names([s.column for s in zero_inflated])
                    + ". Zero-inflation often reflects a two-part process (something "
                    "happened / did not happen) rather than a continuous measurement."
                ),
                severity=Severity.LOW,
                evidence={"columns": [s.column for s in zero_inflated],
                          "zero_share": {s.column: s.zero_share for s in zero_inflated}},
                recommendation=(
                    "Consider modelling the occurrence and the magnitude separately, and "
                    "verify that zero means 'none' rather than 'unknown'."
                ),
            )
        )

    scale_spread = _scaling_spread(stats)
    if scale_spread:
        findings.append(scale_spread)

    flags = binary_flag_columns(stats)
    if flags:
        findings.append(
            Finding(
                category="numerical",
                title=f"{len(flags)} binary indicator column(s) present",
                detail=(
                    join_names([s.column for s in flags])
                    + " take exactly two distinct numeric values. They are treated as "
                    "flags: interval statistics such as skewness and kurtosis are not "
                    "meaningful for them, and they are excluded from outlier testing "
                    "because the minority class would be flagged as 'extreme'."
                ),
                severity=Severity.INFO,
                evidence={"columns": [s.column for s in flags]},
            )
        )

    integer_ids = [
        s for s in stats
        if s.count > 10 and s.n_unique == s.count and s.minimum is not None and s.minimum >= 0
        and s.column.lower().endswith(("id", "index", "number", "_no"))
    ]
    if integer_ids:
        findings.append(
            Finding(
                category="numerical",
                title=f"{len(integer_ids)} numerical column(s) look like identifiers",
                detail=(
                    join_names([s.column for s in integer_ids])
                    + " are unique, non-negative integers with no repeated values — the "
                    "signature of a record identifier rather than a measurement."
                ),
                severity=Severity.LOW,
                evidence={"columns": [s.column for s in integer_ids]},
                recommendation=(
                    "Exclude identifier columns from correlation, scaling and modelling; "
                    "they add noise and can create spurious relationships."
                ),
            )
        )

    return findings


def _scaling_spread(stats: list[NumericColumnStats]) -> Finding | None:
    """Flag a very large spread of column magnitudes, which affects scaling."""
    magnitudes = [
        max(abs(s.minimum or 0.0), abs(s.maximum or 0.0))
        for s in stats
        if s.maximum is not None and abs(s.maximum) > 0
    ]
    if len(magnitudes) < 2:
        return None
    magnitudes.sort()
    smallest, largest = magnitudes[0], magnitudes[-1]
    if smallest <= 0 or largest / smallest < 1e4:
        return None
    return Finding(
        category="numerical",
        title="Numerical columns span very different magnitudes",
        detail=(
            f"Column ranges differ by roughly {largest / smallest:,.0f}x (from about "
            f"{format_number(smallest)} to {format_number(largest)}). Unscaled features "
            "of very different size dominate distance-based and regularised models."
        ),
        severity=Severity.LOW,
        evidence={"smallest_range": smallest, "largest_range": largest,
                  "ratio": largest / smallest},
        recommendation=(
            "Standardise (z-score) or min-max scale features before using linear "
            "models, k-NN, SVMs, PCA or neural networks."
        ),
    )


def binary_flag_columns(stats: list[NumericColumnStats]) -> list[NumericColumnStats]:
    """Numeric columns holding exactly two distinct values (0/1 style indicators)."""
    return [s for s in stats if s.is_binary_flag]


def outlier_candidate_columns(frame: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Numeric columns eligible for outlier testing (at least 4 distinct values).

    Binary indicators are excluded on purpose: every IQR or z-score rule would flag
    the minority class, which is a property of the encoding rather than an anomaly.
    """
    return [
        c for c in numeric_columns
        if c in frame.columns
        and frame[c].nunique(dropna=True) >= config.MIN_UNIQUE_FOR_OUTLIER_TESTS
    ]


def correlation_candidate_columns(frame: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """Numeric columns suitable for correlation analysis (>= 2 distinct values)."""
    return [
        c for c in numeric_columns
        if c in frame.columns and frame[c].nunique(dropna=True) >= config.MIN_DISTINCT_VALUES
    ]


def is_identifier_like_numeric(stats: NumericColumnStats) -> bool:
    """True when a numeric column is unique-valued and therefore ID-like."""
    return (
        stats.count > 0
        and stats.n_unique == stats.count
        and is_integer_valued(pd.Series([stats.minimum or 0.0, stats.maximum or 0.0]))
    )
