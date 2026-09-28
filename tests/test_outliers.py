"""Tests for the IQR, z-score and modified z-score outlier rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import outliers
from src.schema import Severity


def test_iqr_flags_known_extreme_values() -> None:
    # 20 tightly clustered values plus one obvious extreme.
    series = pd.Series([10.0, 11, 12, 13, 14, 10.5, 11.5, 12.5, 13.5, 14.5,
                        10.2, 11.2, 12.2, 13.2, 14.2, 10.8, 11.8, 12.8, 13.8, 14.8, 500.0])
    result = outliers.iqr_outliers(series, "x")

    assert result.method == "iqr"
    assert result.count == 1
    assert result.values == [500.0]
    assert result.indices == [20]
    assert result.rate == pytest.approx(1 / 21)
    assert result.upper_bound is not None and result.upper_bound < 500


def test_iqr_boundaries_match_manual_computation() -> None:
    series = pd.Series(list(range(1, 101)), dtype="float64")
    result = outliers.iqr_outliers(series, "x")
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    assert result.lower_bound == pytest.approx(q1 - 1.5 * iqr)
    assert result.upper_bound == pytest.approx(q3 + 1.5 * iqr)
    assert result.count == 0


def test_iqr_extreme_multiplier_flags_fewer_points() -> None:
    series = pd.Series([1.0] * 50 + [3.0, 10.0])
    standard = outliers.iqr_outliers(series, "x")
    extreme = outliers.iqr_outliers(series, "x", multiplier=3.0, method="iqr_extreme")
    assert extreme.count <= standard.count


def test_zscore_detects_outlier_and_reports_threshold() -> None:
    rng = np.random.default_rng(42)
    values = np.append(rng.normal(0, 1, 500), 12.0)
    result = outliers.zscore_outliers(pd.Series(values), "x")

    assert result.available is True
    assert result.count == 1
    assert result.values[0] == pytest.approx(12.0)
    assert result.threshold == pytest.approx(3.0)


def test_zscore_unavailable_for_tiny_or_constant_columns() -> None:
    tiny = outliers.zscore_outliers(pd.Series([1.0, 2.0]), "x")
    assert tiny.available is False
    assert tiny.count == 0
    assert "fewer than" in tiny.reason.lower()

    constant = outliers.zscore_outliers(pd.Series([5.0] * 30), "x")
    assert constant.available is False
    assert "zero variance" in constant.reason.lower()


def test_modified_zscore_is_robust_to_masking() -> None:
    # Two extremes inflate the standard deviation, so the plain z-score rule sees only
    # the larger one. The MAD-based rule is not dragged along by them and sees both.
    rng = np.random.default_rng(11)
    values = np.append(rng.normal(0, 1, 28), [6.0, 12.0])
    series = pd.Series(values)

    zscore = outliers.zscore_outliers(series, "x")
    modified = outliers.modified_zscore_outliers(series, "x")

    assert modified.available is True
    assert modified.count == 2
    assert zscore.count < modified.count


def test_modified_zscore_unavailable_when_mad_is_zero() -> None:
    series = pd.Series([1.0] * 60 + [2.0] * 5)
    result = outliers.modified_zscore_outliers(series, "x")
    assert result.available is False
    assert "zero" in result.reason.lower()


def test_analyze_outliers_covers_every_method_and_columns_flagged() -> None:
    frame = pd.DataFrame(
        {
            "a": np.append(np.random.default_rng(1).normal(0, 1, 200), 40.0),
            "b": np.random.default_rng(2).normal(5, 2, 201),
        }
    )
    summary = outliers.analyze_outliers(frame, ["a", "b"])

    methods = {r.method for r in summary.results}
    assert methods == {"iqr", "iqr_extreme", "zscore", "modified_zscore"}
    assert "a" in summary.columns_flagged
    assert len(summary.rows_flagged_by_any) >= 1
    assert summary.total_columns_with_outliers == len(summary.columns_flagged)


def test_binary_columns_are_excluded_from_outlier_testing() -> None:
    frame = pd.DataFrame({"flag": [0] * 190 + [1] * 10, "value": list(range(200))})
    summary = outliers.analyze_outliers(frame, ["flag", "value"])

    flag_iqr = next(r for r in summary.results if r.column == "flag" and r.method == "iqr")
    assert flag_iqr.available is False
    assert "distinct" in flag_iqr.reason
    assert "flag" not in summary.columns_flagged

    findings = outliers.outlier_findings(summary, ["flag", "value"])
    assert any("excluded" in f.title.lower() for f in findings)


def test_outlier_findings_state_that_outliers_are_not_errors() -> None:
    clustered = [10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5] * 5
    frame = pd.DataFrame({"x": clustered + [500.0]})
    summary = outliers.analyze_outliers(frame, ["x"])
    findings = outliers.outlier_findings(summary, ["x"])

    flagged = next(f for f in findings if "IQR outliers" in f.title)
    assert "not automatically an error" in flagged.recommendation
    assert flagged.severity.rank >= Severity.LOW.rank


def test_no_numeric_columns_returns_informational_finding() -> None:
    findings = outliers.outlier_findings(outliers.OutlierSummary(), [])
    assert findings[0].severity is Severity.INFO
    assert "not applicable" in findings[0].title.lower()


def test_outlier_examples_return_the_flagged_rows() -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 99.0], "label": list("abcdef")})
    result = outliers.iqr_outliers(frame["x"], "x")
    examples = outliers.outlier_examples(frame, "x", result)

    assert len(examples) == result.count
    assert examples.iloc[0]["row_index"] == 5
    assert examples.iloc[0]["x"] == 99.0


def test_sentinel_values_are_reported_as_a_suspicion() -> None:
    values = np.append(np.random.default_rng(3).normal(50, 5, 300), [-999.0] * 5)
    frame = pd.DataFrame({"amount": values})
    summary = outliers.analyze_outliers(frame, ["amount"])
    finding = outliers.detect_sentinel_values(summary)

    assert finding is not None
    assert "suspicion" in finding.detail.lower()
    assert finding.severity.rank >= Severity.LOW.rank


def test_disagreement_between_methods_is_explained() -> None:
    # Heavily skewed data: the standard deviation is inflated, so the z-score sees
    # fewer outliers than the IQR rule.
    values = np.append(np.random.default_rng(4).exponential(3.0, 400), [80.0, 95.0, 120.0])
    frame = pd.DataFrame({"skewed": values})
    summary = outliers.analyze_outliers(frame, ["skewed"])
    findings = outliers.outlier_findings(summary, ["skewed"])
    assert any("disagree" in f.title.lower() for f in findings)
