"""Categorical variable analysis: cardinality, dominance, rarity and entropy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schema import CategoricalColumnStats, Finding, Severity
from src.utils import ellipsize, format_percent, join_names, safe_divide


def frequency_table(frame: pd.DataFrame, column: str, top_n: int = 20) -> pd.DataFrame:
    """Return the ``top_n`` most frequent values of ``column`` with counts and shares.

    Produces an empty frame (instead of raising) when the column is missing or
    entirely null, so the dashboard can render a friendly message.
    """
    columns = ["value", "count", "percent"]
    if column not in frame.columns:
        return pd.DataFrame(columns=columns)

    series = frame[column].dropna()
    if series.empty:
        return pd.DataFrame(columns=columns)

    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    table = pd.DataFrame(
        {
            "value": counts.index.astype(str),
            "count": counts.to_numpy(),
            "percent": (counts.to_numpy() / max(total, 1)) * 100.0,
        }
    ).head(top_n)

    other_count = total - int(table["count"].sum())
    if other_count > 0:
        table = pd.concat(
            [
                table,
                pd.DataFrame(
                    [{
                        "value": f"(other {max(len(counts) - top_n, 0)} categories)",
                        "count": other_count,
                        "percent": other_count / max(total, 1) * 100.0,
                    }]
                ),
            ],
            ignore_index=True,
        )
    table["percent"] = table["percent"].round(3)
    return table.reset_index(drop=True)


def describe_categorical_column(series: pd.Series, column: str) -> CategoricalColumnStats:
    """Compute cardinality, dominance and entropy for one categorical column."""
    total = len(series)
    non_null = series.dropna()
    n = int(non_null.size)
    missing = int(total - n)

    if n == 0:
        return CategoricalColumnStats(
            column=column, count=0, missing=missing, n_unique=0, unique_ratio=0.0,
            top_value=None, top_count=0, top_rate=0.0,
        )

    counts = non_null.value_counts(dropna=True)
    n_unique = int(counts.size)
    top_value = counts.index[0]
    top_count = int(counts.iloc[0])
    top_rate = safe_divide(top_count, n, 0.0)
    second_value = counts.index[1] if n_unique > 1 else None
    second_rate = safe_divide(int(counts.iloc[1]), n, 0.0) if n_unique > 1 else 0.0
    unique_ratio = safe_divide(n_unique, n, 0.0)

    probabilities = counts.to_numpy(dtype="float64") / n
    entropy = float(-np.sum(probabilities * np.log2(probabilities))) if n_unique > 1 else 0.0
    rare_categories = int((counts.to_numpy() / n < config.RARE_CATEGORY_SHARE).sum())

    return CategoricalColumnStats(
        column=column,
        count=n,
        missing=missing,
        n_unique=n_unique,
        unique_ratio=unique_ratio,
        top_value=top_value,
        top_count=top_count,
        top_rate=top_rate,
        second_value=second_value,
        second_rate=second_rate,
        rare_categories=rare_categories,
        entropy=entropy,
        is_dominant=top_rate >= config.DOMINANT_CATEGORY,
        is_high_cardinality=(
            n_unique >= config.HIGH_CARDINALITY_UNIQUE
            or unique_ratio >= config.HIGH_CARDINALITY_RATIO
        ),
    )


def analyze_categorical(
    frame: pd.DataFrame, categorical_columns: list[str]
) -> list[CategoricalColumnStats]:
    """Describe each categorical column; failures on one column are isolated."""
    stats: list[CategoricalColumnStats] = []
    for column in categorical_columns:
        if column not in frame.columns:
            continue
        try:
            stats.append(describe_categorical_column(frame[column], str(column)))
        except Exception:  # noqa: BLE001 - mixed/unhashable values must not crash the run
            continue
    return stats


def categorical_findings(
    stats: list[CategoricalColumnStats], n_columns: int
) -> list[Finding]:
    """Report dominant, imbalanced, high-cardinality and rare-category columns."""
    if not stats:
        return [
            Finding(
                category="categorical",
                title="No categorical columns available",
                detail="The dataset contains no categorical columns to analyse.",
                severity=Severity.INFO,
                evidence={"categorical_columns": 0},
            )
        ]

    findings: list[Finding] = []

    # Constant columns are reported by the low-variance analysis; repeating them here
    # as "100% dominant" would double-count the same problem.
    dominant = [s for s in stats if s.is_dominant and s.n_unique > 1]
    if dominant:
        worst = max(dominant, key=lambda s: s.top_rate)
        findings.append(
            Finding(
                category="categorical",
                title=f"{len(dominant)} categorical variable(s) are highly imbalanced",
                detail=(
                    f"{format_percent(worst.top_rate)} of observations in "
                    f"`{worst.column}` belong to the single category "
                    f"'{ellipsize(str(worst.top_value), 40)}' "
                    f"({worst.top_count:,} of {worst.count:,} non-missing rows). "
                    "Affected columns: "
                    + join_names([s.column for s in dominant]) + "."
                ),
                severity=(
                    Severity.HIGH
                    if worst.top_rate >= config.DOMINANT_CATEGORY_SEVERE
                    else Severity.MODERATE
                ),
                evidence={
                    "columns": {s.column: round(s.top_rate, 4) for s in dominant},
                    "worst_case": worst.column,
                },
                recommendation=(
                    "If such a column is your prediction target, accuracy becomes a "
                    "misleading metric (predicting the majority class alone would score "
                    f"{format_percent(worst.top_rate)}). Use precision/recall, F1, PR-AUC "
                    "or balanced sampling instead. If it is a feature, it cannot "
                    "separate many observations."
                ),
            )
        )

    high_cardinality = [s for s in stats if s.is_high_cardinality and s.n_unique > 1]
    if high_cardinality:
        findings.append(
            Finding(
                category="categorical",
                title=f"{len(high_cardinality)} high-cardinality categorical column(s)",
                detail=(
                    "These columns hold many distinct values relative to the number of "
                    "rows (e.g. "
                    + ", ".join(
                        f"`{s.column}`: {s.n_unique:,} unique values in {s.count:,} rows"
                        for s in high_cardinality[:3]
                    )
                    + "). One-hot encoding them would create thousands of near-empty "
                    "features, and they often behave like free text or identifiers."
                ),
                severity=Severity.MODERATE,
                evidence={
                    "columns": {s.column: {"n_unique": s.n_unique, "ratio": round(s.unique_ratio, 3)}
                                for s in high_cardinality}
                },
                recommendation=(
                    "Use target/frequency encoding, group rare levels into 'other', or "
                    "exclude the column if it is an identifier."
                ),
            )
        )

    rare_heavy = [s for s in stats if s.rare_categories >= max(5, 0.3 * s.n_unique)]
    if rare_heavy:
        findings.append(
            Finding(
                category="categorical",
                title=f"{len(rare_heavy)} column(s) contain many rare categories",
                detail=(
                    "More than 30% of the categories in "
                    + join_names([s.column for s in rare_heavy])
                    + f" each represent less than {format_percent(config.RARE_CATEGORY_SHARE)} "
                    "of observations. Rare levels add little signal and can make a model "
                    "unstable when they are absent from a validation split."
                ),
                severity=Severity.LOW,
                evidence={"columns": {s.column: s.rare_categories for s in rare_heavy}},
                recommendation=(
                    "Group rare levels into an 'other' bucket, or keep only the most "
                    "frequent levels if the column is important."
                ),
            )
        )

    return findings
