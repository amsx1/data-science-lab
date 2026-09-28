"""Tests for correlation, association and anomaly detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import anomalies, correlations
from src.schema import Severity


# --------------------------------------------------------------------------- #
# Numerical correlation
# --------------------------------------------------------------------------- #
def test_perfect_positive_and_negative_correlations_are_found() -> None:
    x = np.arange(50, dtype="float64")
    frame = pd.DataFrame({"x": x, "plus": x * 2 + 1, "minus": -x, "noise": np.random.default_rng(0).normal(0, 1, 50)})
    report = correlations.analyze_correlations(frame, ["x", "plus", "minus", "noise"])

    pairs = {frozenset((p.column_a, p.column_b)): p for p in report.pairs}
    assert pairs[frozenset(("plus", "x"))].r == pytest.approx(1.0)
    assert pairs[frozenset(("minus", "x"))].r == pytest.approx(-1.0)
    assert pairs[frozenset(("minus", "x"))].direction == "negative"
    assert pairs[frozenset(("plus", "x"))].strength == "near-perfect"
    assert "x" in report.columns_analysed


def test_self_correlations_are_never_reported() -> None:
    frame = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) * 3})
    report = correlations.analyze_correlations(frame, ["a", "b"])
    assert all(p.column_a != p.column_b for p in report.pairs)
    assert len(report.pairs) == 1


def test_correlation_matrix_diagonal_is_present_but_blanked_in_charts() -> None:
    from src.charts import correlation_heatmap

    frame = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) ** 2, "c": np.arange(10.0) * -1})
    matrix = correlations.correlation_matrix(frame, ["a", "b", "c"])
    assert matrix.loc["a", "a"] == pytest.approx(1.0)

    figure = correlation_heatmap(matrix)
    z = np.array(figure.data[0].z, dtype=float)
    assert np.isnan(np.diag(z)).all()


def test_pairwise_complete_observations_are_used() -> None:
    x = pd.Series(np.arange(30, dtype="float64"))
    y = x * 2
    y[::3] = np.nan          # missing values in one column only
    frame = pd.DataFrame({"x": x, "y": y})

    report = correlations.analyze_correlations(frame, ["x", "y"])
    pair = report.pairs[0]
    assert pair.r == pytest.approx(1.0)
    assert pair.n_observations == 20
    assert pair.reliable is True


def test_strength_labels_and_direction() -> None:
    assert correlations.strength_label(0.99) == "near-perfect"
    assert correlations.strength_label(0.75) == "strong"
    assert correlations.strength_label(0.55) == "moderate"
    assert correlations.strength_label(0.35) == "weak"
    assert correlations.strength_label(0.05) == "very weak"


def test_strong_pairs_are_split_by_sign() -> None:
    x = np.arange(60, dtype="float64")
    frame = pd.DataFrame({"x": x, "pos": x * 1.5, "neg": -2 * x + 5})
    report = correlations.analyze_correlations(frame, ["x", "pos", "neg"])

    assert any(p.r > 0.7 for p in report.strong_positive)
    assert any(p.r < -0.7 for p in report.strong_negative)
    assert report.multicollinear                                # |r| >= 0.8 also recorded


def test_p_values_are_returned_for_strong_pairs_only() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, 200)
    frame = pd.DataFrame({"x": x, "strong": x * 3 + rng.normal(0, 0.1, 200),
                          "weak": rng.normal(0, 1, 200)})
    report = correlations.analyze_correlations(frame, ["x", "strong", "weak"])

    strong_pair = next(p for p in report.pairs if "strong" in (p.column_a, p.column_b))
    assert strong_pair.p_value is not None and strong_pair.p_value < 0.05


def test_correlation_unavailable_with_single_numeric_column() -> None:
    frame = pd.DataFrame({"a": np.arange(10.0), "b": list("abcdefghij")})
    report = correlations.analyze_correlations(frame, ["a"])
    assert report.unavailable_reason
    findings = correlations.correlation_findings(report)
    assert findings[0].severity is Severity.INFO


def test_constant_column_is_excluded_from_correlation() -> None:
    frame = pd.DataFrame({"a": np.arange(10.0), "const": [1.0] * 10})
    report = correlations.analyze_correlations(frame, ["a", "const"])
    assert report.unavailable_reason or all(
        p.column_a != "const" and p.column_b != "const" for p in report.pairs
    )


def test_correlation_findings_mention_causation_caveat() -> None:
    x = np.arange(40, dtype="float64")
    frame = pd.DataFrame({"a": x, "b": x * 2})
    report = correlations.analyze_correlations(frame, ["a", "b"])
    findings = correlations.correlation_findings(report)
    combined = " ".join(f.detail for f in findings)
    assert "does not imply" in combined or "not imply" in combined


def test_no_relationship_message_for_independent_columns() -> None:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
    report = correlations.analyze_correlations(frame, ["a", "b"])
    findings = correlations.correlation_findings(report)
    assert any("no meaningful linear relationships" in f.title.lower() for f in findings)


def test_correlation_pairs_table_shape() -> None:
    x = np.arange(20.0)
    frame = pd.DataFrame({"a": x, "b": x * 2})
    report = correlations.analyze_correlations(frame, ["a", "b"])
    table = correlations.correlation_pairs_table(report)
    assert list(table.columns)[:4] == ["column_a", "column_b", "r", "abs_r"]
    assert len(table) == 1


# --------------------------------------------------------------------------- #
# Categorical association
# --------------------------------------------------------------------------- #
def test_cramers_v_detects_perfect_association() -> None:
    table = pd.DataFrame({"x": [30, 0], "y": [0, 30]})
    v, p_value, dof = correlations.cramers_v(table)
    assert v == pytest.approx(1.0, abs=0.01)
    assert p_value < 0.05
    assert dof == 1


def test_cramers_v_is_near_zero_for_independence() -> None:
    rng = np.random.default_rng(3)
    table = pd.DataFrame(rng.integers(40, 60, size=(3, 3)))
    v, _, _ = correlations.cramers_v(table)
    assert v < 0.2


def test_categorical_associations_screen_is_bounded_and_sorted() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        {
            "a": rng.choice(list("xyz"), 300),
            "b": rng.choice(list("pqr"), 300),
            "c": rng.choice(list("mn"), 300),
        }
    )
    frame["b"] = frame["a"]           # inject a deterministic association
    associations = correlations.categorical_associations(frame, ["a", "b", "c"])

    assert not associations.empty
    assert associations["cramers_v"].is_monotonic_decreasing
    top = associations.iloc[0]
    assert {top["column_a"], top["column_b"]} == {"a", "b"}
    assert top["cramers_v"] > 0.9


def test_association_findings_report_strong_pairs() -> None:
    rng = np.random.default_rng(6)
    frame = pd.DataFrame({"a": rng.choice(list("abc"), 200)})
    frame["b"] = frame["a"]
    associations = correlations.categorical_associations(frame, ["a", "b"])
    findings = correlations.association_findings(associations)
    assert findings and "associated" in findings[0].title.lower()


def test_association_findings_empty_when_no_strong_pairs() -> None:
    associations = pd.DataFrame(
        [{"column_a": "a", "column_b": "b", "cramers_v": 0.1, "p_value": 0.5,
          "n_observations": 100, "levels_a": 3, "levels_b": 3, "dof": 4}]
    )
    assert correlations.association_findings(associations) == []


# --------------------------------------------------------------------------- #
# Anomaly detection
# --------------------------------------------------------------------------- #
def test_anomalies_detects_planted_multivariate_outliers() -> None:
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, 3))
    planted = np.array([[12.0, 12.0, 12.0], [-11.0, -11.0, -11.0]])
    data = np.vstack([normal, planted])
    frame = pd.DataFrame(data, columns=["a", "b", "c"])

    result = anomalies.detect_anomalies(frame, ["a", "b", "c"], contamination=0.01)

    assert result.available is True
    assert result.features_used == ["a", "b", "c"]
    flagged_rows = set(result.records.loc[result.records["is_anomaly"], "row_index"])
    assert {500, 501}.issubset(flagged_rows)
    assert result.anomaly_rate == pytest.approx(0.01, abs=0.02)
    assert result.mean_score is not None


def test_anomaly_detection_is_deterministic() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(0, 1, size=(200, 2)), columns=["a", "b"])
    first = anomalies.detect_anomalies(frame, ["a", "b"], random_state=42)
    second = anomalies.detect_anomalies(frame, ["a", "b"], random_state=42)
    assert first.n_anomalies == second.n_anomalies
    assert first.records["anomaly_score"].tolist() == second.records["anomaly_score"].tolist()


def test_anomaly_detection_unavailable_for_tiny_datasets() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    result = anomalies.detect_anomalies(frame, ["a", "b"])
    assert result.available is False
    assert "at least" in result.reason.lower()

    findings = anomalies.anomaly_findings(result)
    assert findings[0].severity is Severity.INFO
    assert "not available" in findings[0].title.lower()


def test_anomaly_detection_requires_two_features() -> None:
    frame = pd.DataFrame({"a": np.random.default_rng(2).normal(0, 1, 100)})
    result = anomalies.detect_anomalies(frame, ["a"])
    assert result.available is False
    assert "two" in result.reason.lower() or "2" in result.reason


def test_anomaly_detection_handles_missing_values_by_imputation() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(0, 1, size=(200, 2)), columns=["a", "b"])
    frame.loc[::10, "a"] = np.nan
    result = anomalies.detect_anomalies(frame, ["a", "b"])
    assert result.available is True
    assert result.n_rows_scored == 200


def test_anomaly_contamination_is_clamped_to_valid_range() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(0, 1, size=(300, 2)), columns=["a", "b"])
    result = anomalies.detect_anomalies(frame, ["a", "b"], contamination=5.0)
    assert result.contamination <= 0.5


def test_high_contamination_produces_a_cautionary_finding() -> None:
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(0, 1, size=(300, 2)), columns=["a", "b"])
    result = anomalies.detect_anomalies(frame, ["a", "b"], contamination=0.30)
    findings = anomalies.anomaly_findings(result)
    assert any("contamination" in f.title.lower() for f in findings)


def test_anomaly_findings_explain_they_are_not_errors() -> None:
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(rng.normal(0, 1, size=(300, 2)), columns=["a", "b"])
    result = anomalies.detect_anomalies(frame, ["a", "b"])
    finding = next(f for f in anomalies.anomaly_findings(result) if "flagged" in f.title)
    assert "not 'fraudulent'" in finding.detail or "statistically unusual" in finding.detail


def test_anomaly_examples_and_deviations() -> None:
    rng = np.random.default_rng(7)
    data = np.vstack([rng.normal(0, 1, size=(400, 3)), [[14.0, 14.0, 14.0]]])
    frame = pd.DataFrame(data, columns=["a", "b", "c"])
    result = anomalies.detect_anomalies(frame, ["a", "b", "c"], contamination=0.01)

    examples = anomalies.anomaly_examples(result)
    assert not examples.empty
    assert examples["is_anomaly"].all()
    assert not result.top_deviating_features.empty
    assert set(result.top_deviating_features.columns) == {"feature", "mean_robust_z", "max_abs_robust_z"}
