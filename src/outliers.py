"""Outlier detection using three complementary, explainable methods.

* **IQR (Tukey) rule** — flags values below ``Q1 - k*IQR`` or above ``Q3 + k*IQR``.
  Distribution-free and robust, because quartiles are barely affected by extremes.
* **Z-score** — flags values whose distance from the mean exceeds ``k`` standard
  deviations. Sensitive to skew: a single extreme value inflates the standard
  deviation and can hide other outliers (the masking effect).
* **Modified Z-score (MAD)** — uses the median and the median absolute deviation,
  so it behaves like the z-score but resists masking.

An outlier is a *statistical* observation, never automatically an error. All
findings from this module say so explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src import config
from src.schema import Finding, OutlierResult, OutlierSummary, Severity
from src.utils import finite_values, format_number, format_percent, join_names, safe_divide

#: Numeric values frequently used as "missing" or sentinel codes in real systems.
SENTINEL_CANDIDATES: tuple[float, ...] = (
    -1.0, -9.0, -99.0, -999.0, -9999.0,
    999.0, 9999.0, 99999.0, 999999.0,
)


def _values_and_positions(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(row_positions, finite_values)`` for the finite entries of ``series``.

    Positions refer to the original row index of the frame, which lets the UI
    display the actual offending records.
    """
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    mask = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
    positions = np.flatnonzero(mask)
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)[mask]
    return positions, values


def _empty_result(column: str, method: str, reason: str) -> OutlierResult:
    """Result object used when a method cannot be applied to a column."""
    return OutlierResult(
        column=column, method=method, count=0, rate=0.0, available=False, reason=reason
    )


def _build_result(
    column: str,
    method: str,
    positions: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    **extra: float | None,
) -> OutlierResult:
    """Assemble an :class:`OutlierResult` from a boolean flag mask."""
    flagged_positions = positions[mask]
    flagged_values = values[mask]
    n = int(values.size)
    return OutlierResult(
        column=column,
        method=method,
        count=int(mask.sum()),
        rate=safe_divide(int(mask.sum()), n, 0.0),
        indices=[int(p) for p in flagged_positions],
        values=[float(v) for v in flagged_values],
        **extra,
    )


# --------------------------------------------------------------------------- #
# Individual methods
# --------------------------------------------------------------------------- #
def iqr_outliers(
    series: pd.Series,
    column: str,
    multiplier: float = config.IQR_MULTIPLIER,
    method: str = "iqr",
) -> OutlierResult:
    """Tukey's fence rule. Returns a zero-count result for empty/constant columns."""
    positions, values = _values_and_positions(series)
    if values.size < 4:
        return _empty_result(column, method, "Fewer than 4 finite values.")

    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr == 0:
        # A zero IQR means at least half the values are identical; the fence
        # degenerates, so only exact departures from that plateau are flagged.
        lower = upper = float(q1)
    else:
        lower = float(q1 - multiplier * iqr)
        upper = float(q3 + multiplier * iqr)

    mask = (values < lower) | (values > upper)
    return _build_result(
        column, method, positions, values, mask,
        lower_bound=lower, upper_bound=upper, threshold=float(multiplier),
    )


def zscore_outliers(
    series: pd.Series,
    column: str,
    threshold: float = config.ZSCORE_THRESHOLD,
    method: str = "zscore",
) -> OutlierResult:
    """Classic z-score rule using the sample mean and standard deviation."""
    positions, values = _values_and_positions(series)
    if values.size < config.MIN_ROWS_FOR_ZSCORE:
        return _empty_result(
            column, method,
            f"Fewer than {config.MIN_ROWS_FOR_ZSCORE} finite values: z-scores would be unstable.",
        )

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    if std == 0 or not np.isfinite(std):
        return _empty_result(column, method, "Zero variance: every value is identical.")

    z = np.abs((values - mean) / std)
    mask = z > threshold
    return _build_result(
        column, method, positions, values, mask, threshold=float(threshold),
        lower_bound=float(mean - threshold * std), upper_bound=float(mean + threshold * std),
    )


def modified_zscore_outliers(
    series: pd.Series,
    column: str,
    threshold: float = config.MODIFIED_ZSCORE_THRESHOLD,
    method: str = "modified_zscore",
) -> OutlierResult:
    """Robust z-score based on the median absolute deviation (Iglewicz & Hoaglin)."""
    positions, values = _values_and_positions(series)
    if values.size < config.MIN_ROWS_FOR_ZSCORE:
        return _empty_result(
            column, method,
            f"Fewer than {config.MIN_ROWS_FOR_ZSCORE} finite values: MAD would be unstable.",
        )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return _empty_result(
            column, method,
            "Median absolute deviation is zero: more than half the values are identical.",
        )

    # 0.6745 rescales the MAD so the score is comparable to a standard z-score.
    score = 0.6745 * (values - median) / mad
    mask = np.abs(score) > threshold
    return _build_result(
        column, method, positions, values, mask, threshold=float(threshold),
        lower_bound=float(median - threshold * mad / 0.6745),
        upper_bound=float(median + threshold * mad / 0.6745),
    )


# --------------------------------------------------------------------------- #
# Whole-dataset analysis
# --------------------------------------------------------------------------- #
def analyze_outliers(
    frame: pd.DataFrame, numeric_columns: list[str], include_modified_zscore: bool = True
) -> OutlierSummary:
    """Run all outlier methods over the numeric columns of ``frame``."""
    results: list[OutlierResult] = []
    flagged_columns: list[str] = []
    rows_flagged: set[int] = set()

    for column in numeric_columns:
        if column not in frame.columns:
            continue
        series = frame[column]

        # Binary / near-binary columns are skipped with an explicit reason rather than
        # silently: the fences would just re-detect the minority class.
        if series.nunique(dropna=True) < config.MIN_UNIQUE_FOR_OUTLIER_TESTS:
            reason = (
                f"Only {series.nunique(dropna=True)} distinct value(s): the IQR and "
                "z-score rules would flag the minority class itself rather than a true "
                "extreme observation."
            )
            results.extend(
                _empty_result(str(column), method, reason)
                for method in ("iqr", "iqr_extreme", "zscore", "modified_zscore")
            )
            continue

        column_results = [
            iqr_outliers(series, str(column)),
            iqr_outliers(series, str(column), multiplier=config.IQR_EXTREME_MULTIPLIER,
                         method="iqr_extreme"),
            zscore_outliers(series, str(column)),
        ]
        if include_modified_zscore:
            column_results.append(modified_zscore_outliers(series, str(column)))

        results.extend(column_results)

        for result in column_results:
            if result.count > 0:
                rows_flagged.update(result.indices)
        if any(r.count > 0 for r in column_results if r.method == "iqr"):
            flagged_columns.append(str(column))

    return OutlierSummary(
        results=results,
        columns_flagged=flagged_columns,
        rows_flagged_by_any=rows_flagged,
    )


def outlier_findings(summary: OutlierSummary, numeric_columns: list[str]) -> list[Finding]:
    """Summarise outlier counts per column and explain how to interpret them."""
    if not numeric_columns:
        return [
            Finding(
                category="outliers",
                title="Outlier detection not applicable",
                detail=(
                    "No numeric columns were found, so the IQR, z-score and modified "
                    "z-score rules could not be applied."
                ),
                severity=Severity.INFO,
                evidence={"numeric_columns": 0},
            )
        ]

    ineligible = sorted({
        r.column for r in summary.results if r.method == "iqr" and not r.available
    })
    iqr_results = [r for r in summary.results if r.method == "iqr" and r.available]
    flagged = [r for r in iqr_results if r.count > 0]
    findings: list[Finding] = []

    if not flagged:
        findings.append(
            Finding(
                category="outliers",
                title="No IQR outliers detected in any numeric column",
                detail=(
                    f"Across {len(numeric_columns)} numeric column(s), every value falls "
                    f"inside the 1.5xIQR fences. "
                    + (
                        "Note that the z-score or modified z-score rules may still flag "
                        "values in skewed columns."
                        if any(r.count > 0 for r in summary.results if r.method != "iqr")
                        else ""
                    )
                ),
                severity=Severity.INFO,
                evidence={"columns_checked": len(numeric_columns)},
            )
        )
    else:
        worst = max(flagged, key=lambda r: r.rate)
        findings.append(
            Finding(
                category="outliers",
                title=f"{len(flagged)} numerical variable(s) contain IQR outliers",
                detail=(
                    "Using the 1.5xIQR rule, "
                    + ", ".join(
                        f"`{r.column}` ({r.count:,} values, {format_percent(r.rate)})"
                        for r in sorted(flagged, key=lambda r: r.rate, reverse=True)[:5]
                    )
                    + f". The most affected column is `{worst.column}` with "
                    f"{format_percent(worst.rate)} of its values outside "
                    f"[{format_number(worst.lower_bound)}, {format_number(worst.upper_bound)}]."
                ),
                severity=(
                    Severity.HIGH if worst.rate >= config.OUTLIER_RATE_HIGH
                    else Severity.MODERATE if worst.rate >= config.OUTLIER_RATE_MODERATE
                    else Severity.LOW
                ),
                evidence={
                    "columns": {
                        r.column: {"count": r.count, "rate": r.rate,
                                   "lower": r.lower_bound, "upper": r.upper_bound}
                        for r in sorted(flagged, key=lambda r: r.rate, reverse=True)
                    },
                    "rows_flagged": len(summary.rows_flagged_by_any),
                },
                recommendation=(
                    "Review the extreme observations individually before removing them. "
                    "An outlier is not automatically an error: it may be a legitimate "
                    "extreme case, a rare sub-population, or a data-entry mistake. "
                    "Removing valid extremes biases the model, especially if they are "
                    "the cases you care about."
                ),
            )
        )

    disagreements = _method_disagreements(summary)
    if disagreements:
        findings.append(
            Finding(
                category="outliers",
                title=f"IQR and z-score disagree in {len(disagreements)} column(s)",
                detail=(
                    "In "
                    + join_names([d[0] for d in disagreements])
                    + " the two methods flag different numbers of values "
                    + "; ".join(f"`{col}` {iqr} vs {zs}" for col, iqr, zs in disagreements[:4])
                    + ". Disagreement usually means the distribution is skewed or has a "
                    "heavy tail: the standard deviation is inflated by the extremes, "
                    "which masks moderately extreme values from the z-score rule."
                ),
                severity=Severity.LOW,
                evidence={
                    "columns": [
                        {"column": col, "iqr_count": iqr, "zscore_count": zs}
                        for col, iqr, zs in disagreements
                    ]
                },
                recommendation=(
                    "Prefer the IQR or modified z-score result for skewed columns, and "
                    "consider a transformation before relying on the plain z-score."
                ),
            )
        )

    if ineligible:
        findings.append(
            Finding(
                category="outliers",
                title=f"{len(ineligible)} column(s) were excluded from outlier testing",
                detail=(
                    join_names(ineligible)
                    + f" have fewer than {config.MIN_UNIQUE_FOR_OUTLIER_TESTS} distinct "
                    "values (binary or near-binary indicators). Applying the IQR or "
                    "z-score rule to them would flag the minority class itself, so they "
                    "were skipped deliberately."
                ),
                severity=Severity.INFO,
                evidence={"columns": ineligible},
            )
        )

    sentinel = detect_sentinel_values(summary)
    if sentinel:
        findings.append(sentinel)

    return findings


def _method_disagreements(summary: OutlierSummary) -> list[tuple[str, int, int]]:
    """Columns where the IQR rule and the z-score rule flag very different counts."""
    by_column: dict[str, dict[str, int]] = {}
    for result in summary.results:
        if result.method in {"iqr", "zscore"} and result.available:
            by_column.setdefault(result.column, {"iqr": 0, "zscore": 0})[result.method] = result.count

    disagreements = []
    for column, counts in by_column.items():
        iqr_count, z_count = counts["iqr"], counts["zscore"]
        if iqr_count == z_count == 0:
            continue
        largest = max(iqr_count, z_count)
        if largest and abs(iqr_count - z_count) / largest >= 0.5:
            disagreements.append((column, iqr_count, z_count))
    return disagreements


def detect_sentinel_values(summary: OutlierSummary) -> Finding | None:
    """Look for frequent round values at the extreme edge — classic sentinel codes.

    Values such as ``-999``, ``9999`` or ``-1`` are often used by source systems to
    mean "unknown". This check only *suspects* that; it never asserts it.
    """
    suspects: dict[str, dict[str, float]] = {}
    for result in summary.results:
        if result.method != "iqr" or result.count == 0:
            continue
        values = np.asarray(result.values or [], dtype="float64")
        extreme = np.concatenate(
            [
                values[values <= (result.lower_bound if result.lower_bound is not None else -np.inf)],
                values[values >= (result.upper_bound if result.upper_bound is not None else np.inf)],
            ]
        )
        if extreme.size == 0:
            continue
        for candidate in SENTINEL_CANDIDATES:
            matches = int(np.count_nonzero(np.isclose(extreme, candidate)))
            if matches >= 1 and result.count > 0 and matches / max(result.count, 1) >= 0.25:
                suspects.setdefault(result.column, {})[str(int(candidate))] = matches

    if not suspects:
        return None

    first = next(iter(suspects))
    return Finding(
        category="outliers",
        title="Possible sentinel or placeholder values detected",
        detail=(
            f"Extreme values in `{first}` repeatedly take round values such as "
            f"{', '.join(list(suspects[first])[:4])}, which source systems commonly use "
            "as stand-ins for 'unknown' or 'not applicable' rather than as real "
            "measurements. This is a suspicion, not a verified fact."
        ),
        severity=Severity.MODERATE,
        evidence={"columns": suspects},
        recommendation=(
            "Check the data dictionary for these codes and, if they are placeholders, "
            "recode them as missing before computing statistics."
        ),
    )


def outlier_examples(
    frame: pd.DataFrame, column: str, result: OutlierResult, limit: int = config.MAX_EXAMPLE_ROWS
) -> pd.DataFrame:
    """Return the actual rows flagged as outliers, sorted by absolute value."""
    if column not in frame.columns or not result.indices:
        return pd.DataFrame()
    positions = [p for p in result.indices if 0 <= p < len(frame)]
    example = frame.iloc[positions].copy()
    example.insert(0, "row_index", positions)
    example = example.assign(_abs=example[column].abs()).sort_values(
        "_abs", ascending=False
    ).drop(columns="_abs")
    return example.head(limit).reset_index(drop=True)


def modified_zscore(series: pd.Series) -> pd.Series:
    """Robust z-score (MAD based) as a Series aligned with the input index."""
    values = finite_values(series)
    if values.size == 0:
        return pd.Series(dtype="float64")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return pd.Series(np.zeros(values.size), index=series.index[series.notna()])
    scores = 0.6745 * (values - median) / mad
    return pd.Series(scores, index=series.index[series.notna()])


def shapiro_p_value(series: pd.Series) -> float | None:
    """Shapiro-Wilk normality p-value (used only as a descriptive extra)."""
    values = finite_values(series)
    if values.size < 8:
        return None
    try:
        sample = values if values.size <= 5000 else np.random.default_rng(
            config.RANDOM_STATE
        ).choice(values, 5000, replace=False)
        return float(scipy_stats.shapiro(sample).pvalue)
    except Exception:  # noqa: BLE001 - scipy can raise on degenerate input
        return None
