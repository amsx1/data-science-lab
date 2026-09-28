"""Anomaly detection with scikit-learn's Isolation Forest.

Isolation Forest isolates observations by repeatedly splitting the feature space
at random; points that are isolated after few splits are unusual. It is
unsupervised, distribution-free and scales linearly, which makes it a good
default for tabular screening.

Preprocessing decisions (all reported in the UI):

* features are imputed with the **column median** so no row is dropped;
* features are **standardised** (zero mean, unit variance) so no single column
  dominates the distance-like reasoning of the algorithm;
* columns with a single distinct value are removed — they cannot contribute;
* rows are capped at ``config.ANOMALY_MAX_ROWS`` with a fixed seed, because the
  algorithm's runtime grows with the number of rows.

An anomaly is *an unusual observation*, not a fraudulent or incorrect one. The
narrative in this module always says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schema import AnomalyResult, Finding, Severity
from src.utils import format_percent, join_names


def _prepare_matrix(
    frame: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Impute, standardise and return the model matrix plus the feature names."""
    usable = [
        c for c in columns
        if c in frame.columns
        and pd.to_numeric(frame[c], errors="coerce").nunique(dropna=True) >= config.MIN_DISTINCT_VALUES
    ]
    if not usable:
        return np.empty((0, 0)), [], frame.iloc[0:0]

    subset = frame[usable].apply(pd.to_numeric, errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan)

    # Median imputation keeps every row in the anomaly screen.
    medians = subset.median(numeric_only=True)
    imputed = subset.fillna(medians).fillna(0.0)

    matrix = imputed.to_numpy(dtype="float64")
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    scaled = (matrix - matrix.mean(axis=0)) / std

    return scaled, [str(c) for c in usable], imputed


def detect_anomalies(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    contamination: float = config.ANOMALY_DEFAULT_CONTAMINATION,
    n_estimators: int = config.ANOMALY_DEFAULT_ESTIMATORS,
    random_state: int = config.RANDOM_STATE,
    max_rows: int = config.ANOMALY_MAX_ROWS,
) -> AnomalyResult:
    """Run Isolation Forest over the numeric columns of ``frame``.

    Returns an explicit ``available=False`` result (with a reason) whenever the
    data cannot support the analysis, instead of raising.
    """
    contamination = float(
        min(max(contamination, config.ANOMALY_CONTAMINATION_MIN),
            config.ANOMALY_CONTAMINATION_MAX)
    )

    if len(frame) < config.ANOMALY_MIN_ROWS:
        return AnomalyResult(
            available=False,
            reason=(
                f"Only {len(frame):,} row(s) available. Isolation Forest needs at least "
                f"{config.ANOMALY_MIN_ROWS} rows to estimate an isolation depth profile."
            ),
            contamination=contamination,
        )

    features = [
        c for c in numeric_columns
        if c in frame.columns
        and frame[c].notna().any()
        and frame[c].nunique(dropna=True) >= config.MIN_UNIQUE_FOR_OUTLIER_TESTS
    ]
    excluded_binary = [
        str(c) for c in numeric_columns
        if c in frame.columns and frame[c].nunique(dropna=True) < config.MIN_UNIQUE_FOR_OUTLIER_TESTS
    ]
    if len(features) < config.ANOMALY_MIN_FEATURES:
        return AnomalyResult(
            available=False,
            reason=(
                f"Only {len(features)} usable numerical column(s) found; anomaly "
                f"detection requires at least {config.ANOMALY_MIN_FEATURES} so that "
                "'unusual' has more than one dimension to be unusual in."
            ),
            contamination=contamination,
        )

    working = frame
    sampled = False
    if len(working) > max_rows:
        working = working.sample(n=max_rows, random_state=random_state).sort_index()
        sampled = True

    matrix, used_features, imputed = _prepare_matrix(working, features)
    if matrix.shape[0] < config.ANOMALY_MIN_ROWS or matrix.shape[1] < config.ANOMALY_MIN_FEATURES:
        return AnomalyResult(
            available=False,
            reason="Not enough usable numeric structure remained after preprocessing.",
            contamination=contamination,
        )

    from sklearn.ensemble import IsolationForest  # imported lazily to keep start-up fast
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)

    model = IsolationForest(
        n_estimators=int(n_estimators),
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(matrix_scaled)          # -1 = anomaly, 1 = normal
    scores = model.score_samples(matrix_scaled)        # lower = more anomalous
    decisions = model.decision_function(matrix_scaled)

    anomaly_mask = labels == -1
    n_anomalies = int(anomaly_mask.sum())

    records = frame.loc[working.index].copy()
    records.insert(0, "row_index", list(working.index))
    records["anomaly_score"] = scores
    records["anomaly_decision"] = decisions
    records["is_anomaly"] = anomaly_mask
    records["anomaly_percentile"] = (
        pd.Series(scores, index=records.index).rank(pct=True) * 100.0
    )
    records = records.sort_values("anomaly_score").reset_index(drop=True)

    deviations = _feature_deviations(records, used_features)

    reason = ""
    if excluded_binary:
        reason = (
            f"Binary/flag column(s) excluded from the feature space: "
            f"{', '.join(excluded_binary)} (they carry no distance information after "
            "standardisation). "
        )
    if sampled:
        reason += (
            f"A random sample of {max_rows:,} rows (seed {random_state}) was scored to "
            f"keep the analysis fast; {len(frame):,} rows were available."
        )

    return AnomalyResult(
        available=True,
        reason=reason,
        contamination=contamination,
        n_rows_scored=int(matrix.shape[0]),
        n_anomalies=n_anomalies,
        anomaly_rate=n_anomalies / max(matrix.shape[0], 1),
        features_used=used_features,
        records=records,
        top_deviating_features=deviations,
    )


def _feature_deviations(records: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Measure which features deviate most inside the flagged anomaly group.

    For each feature the *robust* z-score (median/MAD over all scored rows) is
    averaged over the anomalous rows. A high value means the anomaly group sits
    far from the bulk of the data in that dimension, which is what makes the row
    unusual. This is descriptive, not causal.
    """
    anomalies = records[records["is_anomaly"]]
    if anomalies.empty or not features:
        return pd.DataFrame(columns=["feature", "mean_robust_z", "max_abs_robust_z"])

    rows: list[dict[str, float | str]] = []
    for feature in features:
        values = pd.to_numeric(records[feature], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad == 0:
            continue
        scores = np.abs(0.6745 * (values - median) / mad)
        anomaly_scores = pd.Series(scores, index=records.index)[records["is_anomaly"]]
        rows.append(
            {
                "feature": feature,
                "mean_robust_z": float(anomaly_scores.mean()),
                "max_abs_robust_z": float(anomaly_scores.max()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "mean_robust_z", "max_abs_robust_z"])
    return pd.DataFrame(rows).sort_values("mean_robust_z", ascending=False).reset_index(drop=True)


def anomaly_findings(result: AnomalyResult) -> list[Finding]:
    """Interpret the Isolation Forest output."""
    if not result.available:
        return [
            Finding(
                category="anomalies",
                title="Anomaly detection not available",
                detail=result.reason,
                severity=Severity.INFO,
                evidence={"available": False, "contamination": result.contamination},
            )
        ]

    findings: list[Finding] = []
    severity = Severity.INFO
    if result.anomaly_rate >= 0.10:
        severity = Severity.HIGH
    elif result.anomaly_rate >= 0.05:
        severity = Severity.MODERATE
    elif result.anomaly_rate > 0:
        severity = Severity.LOW

    top_features = result.top_deviating_features.head(5)
    feature_text = ""
    if not top_features.empty:
        feature_text = (
            " The anomalies deviate most in "
            + join_names([str(f) for f in top_features["feature"].tolist()])
            + "."
        )

    findings.append(
        Finding(
            category="anomalies",
            title=(
                f"Isolation Forest flagged {result.n_anomalies:,} anomalous "
                f"record(s) ({format_percent(result.anomaly_rate)})"
            ),
            detail=(
                f"{result.n_rows_scored:,} rows were scored across "
                f"{len(result.features_used)} standardised numeric feature(s) at a "
                f"contamination setting of {result.contamination:.3f}."
                + feature_text
                + (
                    " " + result.reason if result.reason else ""
                )
                + " An anomaly here means 'statistically unusual', not 'fraudulent', "
                "'wrong' or 'impossible': rare but legitimate cases are flagged exactly "
                "because they are rare."
            ),
            severity=severity,
            evidence={
                "n_anomalies": result.n_anomalies,
                "anomaly_rate": result.anomaly_rate,
                "contamination": result.contamination,
                "features_used": result.features_used,
                "scored_rows": result.n_rows_scored,
            },
            recommendation=(
                "Inspect the flagged records with domain experts before acting on them. "
                "If genuine rare cases matter for the task, consider keeping them and "
                "using robust models instead of deleting them."
            ),
        )
    )

    if result.contamination >= 0.25:
        findings.append(
            Finding(
                category="anomalies",
                title="Contamination is set very high",
                detail=(
                    f"At contamination = {result.contamination:.2f} the algorithm is "
                    "forced to label roughly that fraction of rows as anomalies. This "
                    "inflates the anomaly count and its usefulness drops as the setting "
                    "grows."
                ),
                severity=Severity.LOW,
                evidence={"contamination": result.contamination},
                recommendation=(
                    "Try 0.01-0.10 and compare which records are flagged; a defensible "
                    "setting usually comes from an expectation about the data, not from "
                    "maximising the number of flags."
                ),
            )
        )

    return findings


def anomaly_examples(result: AnomalyResult, limit: int = config.MAX_EXAMPLE_ROWS) -> pd.DataFrame:
    """The most anomalous rows, most extreme first."""
    if not result.available or result.records.empty:
        return pd.DataFrame()
    flagged = result.records[result.records["is_anomaly"]]
    if flagged.empty:
        return pd.DataFrame()
    return flagged.head(limit).reset_index(drop=True)
