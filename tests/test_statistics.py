"""Tests for descriptive statistics, categorical analysis and the health score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import categorical, health, statistics
from src.schema import DtypeIssue, DuplicateReport, MissingReport, OutlierSummary, VarianceColumn


# --------------------------------------------------------------------------- #
# Numerical statistics
# --------------------------------------------------------------------------- #
def test_numeric_statistics_match_numpy_reference() -> None:
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    stats = statistics.describe_numeric_column(pd.Series(values), "x")

    assert stats.count == 8
    assert stats.missing == 0
    assert stats.mean == pytest.approx(values.mean())
    assert stats.median == pytest.approx(np.median(values))
    assert stats.std == pytest.approx(values.std(ddof=1))
    assert stats.minimum == pytest.approx(values.min())
    assert stats.maximum == pytest.approx(values.max())
    assert stats.q1 == pytest.approx(np.percentile(values, 25))
    assert stats.q3 == pytest.approx(np.percentile(values, 75))
    assert stats.iqr == pytest.approx(np.percentile(values, 75) - np.percentile(values, 25))
    assert stats.skewness == pytest.approx(pd.Series(values).skew())
    assert stats.kurtosis == pytest.approx(pd.Series(values).kurtosis())


def test_numeric_statistics_ignore_non_finite_and_count_missing() -> None:
    series = pd.Series([1.0, np.nan, np.inf, -np.inf, 5.0])
    stats = statistics.describe_numeric_column(series, "x")
    assert stats.count == 2
    assert stats.missing == 3
    assert stats.mean == pytest.approx(3.0)


def test_numeric_statistics_handle_empty_and_single_value_columns() -> None:
    empty = statistics.describe_numeric_column(pd.Series([np.nan, np.nan]), "empty")
    assert empty.count == 0
    assert empty.mean is None
    assert empty.skewness is None

    single = statistics.describe_numeric_column(pd.Series([7.0]), "one")
    assert single.count == 1
    assert single.std == 0.0
    assert single.skewness is None       # not defined for n < 3


def test_zero_share_and_coefficient_of_variation() -> None:
    stats = statistics.describe_numeric_column(pd.Series([0.0, 0.0, 0.0, 10.0]), "z")
    assert stats.zero_share == pytest.approx(0.75)
    assert stats.coefficient_of_variation is not None

    zero_mean = statistics.describe_numeric_column(pd.Series([-5.0, 5.0]), "zm")
    assert zero_mean.coefficient_of_variation is None


def test_skewness_finding_triggers_on_skewed_data() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"skewed": rng.exponential(2.0, 500)})
    stats = statistics.analyze_numeric(frame, ["skewed"])
    findings = statistics.numeric_findings(stats)
    assert any("skewed" in f.title.lower() for f in findings)


def test_numeric_findings_without_numeric_columns() -> None:
    findings = statistics.numeric_findings([])
    assert findings[0].severity.value == "info"
    assert "no numerical columns" in findings[0].title.lower()


def test_binary_flag_and_outlier_eligibility() -> None:
    frame = pd.DataFrame(
        {
            "flag": [0, 1, 0, 1, 0, 1, 0, 1],
            "continuous": [1.0, 2.5, 3.2, 4.8, 5.1, 6.0, 7.4, 8.9],
        }
    )
    stats = statistics.analyze_numeric(frame, ["flag", "continuous"])
    flags = statistics.binary_flag_columns(stats)
    assert [s.column for s in flags] == ["flag"]

    eligible = statistics.outlier_candidate_columns(frame, ["flag", "continuous"])
    assert eligible == ["continuous"]


def test_numeric_identifier_like_columns_are_flagged() -> None:
    frame = pd.DataFrame({"record_id": range(100), "value": np.arange(100) % 7})
    stats = statistics.analyze_numeric(frame, ["record_id", "value"])
    findings = statistics.numeric_findings(stats)
    assert any("identifier" in f.title.lower() for f in findings)


# --------------------------------------------------------------------------- #
# Categorical statistics
# --------------------------------------------------------------------------- #
def test_categorical_statistics_are_correct() -> None:
    series = pd.Series(["a"] * 190 + ["b"] * 9 + ["c"] * 1)      # c is 0.5% of the column
    stats = categorical.describe_categorical_column(series, "cat")

    assert stats.count == 200
    assert stats.n_unique == 3
    assert stats.top_value == "a"
    assert stats.top_count == 190
    assert stats.top_rate == pytest.approx(0.95)
    assert stats.second_value == "b"
    assert stats.second_rate == pytest.approx(0.045)
    assert stats.is_dominant is True
    assert stats.entropy == pytest.approx(
        -(0.95 * np.log2(0.95) + 0.045 * np.log2(0.045) + 0.005 * np.log2(0.005))
    )
    assert stats.rare_categories == 1                             # below the 1% threshold


def test_dominant_category_finding_reports_the_percentage() -> None:
    frame = pd.DataFrame({"imbalanced": ["yes"] * 987 + ["no"] * 13})
    stats = categorical.analyze_categorical(frame, ["imbalanced"])
    findings = categorical.categorical_findings(stats, n_columns=1)

    dominant = next(f for f in findings if "imbalanced" in f.title.lower())
    assert "98.7%" in dominant.detail
    assert dominant.severity.value in {"moderate", "high"}


def test_constant_column_is_not_reported_twice_as_dominant() -> None:
    frame = pd.DataFrame({"constant": ["same"] * 50, "varied": ["a", "b"] * 25})
    stats = categorical.analyze_categorical(frame, ["constant", "varied"])
    findings = categorical.categorical_findings(stats, n_columns=2)
    assert not any("constant" in f.title.lower() for f in findings)


def test_high_cardinality_detection() -> None:
    frame = pd.DataFrame({"ids": [f"key-{i}" for i in range(200)]})
    stats = categorical.analyze_categorical(frame, ["ids"])
    assert stats[0].is_high_cardinality is True
    findings = categorical.categorical_findings(stats, n_columns=1)
    assert any("high-cardinality" in f.title.lower() for f in findings)


def test_frequency_table_includes_percentages_and_other_bucket() -> None:
    frame = pd.DataFrame({"c": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})
    table = categorical.frequency_table(frame, "c", top_n=2)
    assert list(table.columns) == ["value", "count", "percent"]
    assert table.loc[0, "value"] == "a"
    assert table.loc[0, "percent"] == pytest.approx(50.0)
    assert table["value"].iloc[-1].startswith("(other")


def test_frequency_table_on_missing_column_returns_empty_frame() -> None:
    table = categorical.frequency_table(pd.DataFrame({"a": [1]}), "does_not_exist")
    assert table.empty


def test_categorical_statistics_with_all_null_column() -> None:
    stats = categorical.describe_categorical_column(pd.Series([None, None]), "empty")
    assert stats.count == 0
    assert stats.n_unique == 0
    assert stats.top_value is None


# --------------------------------------------------------------------------- #
# Health score
# --------------------------------------------------------------------------- #
def _clean_inputs():
    missing = MissingReport(total_cells=100, missing_cells=0, missing_rate=0.0,
                            complete_rows=10, rows_with_missing=0, columns_with_missing=0)
    duplicates = DuplicateReport(total_rows=10, duplicate_rows=0, duplicate_rate=0.0)
    return missing, duplicates


def test_perfect_dataset_scores_100() -> None:
    missing, duplicates = _clean_inputs()
    score = health.compute_health_score(
        missing=missing, duplicates=duplicates, variance=[], dtype_issues=[],
        outliers=OutlierSummary(), n_columns=5, numeric_cells=50,
    )
    assert score.score == pytest.approx(100.0)
    assert score.grade == "Excellent"
    assert sum(c.weight for c in score.components) == pytest.approx(1.0)


def test_health_score_weights_sum_and_are_configured() -> None:
    missing, duplicates = _clean_inputs()
    score = health.compute_health_score(missing, duplicates, [], [], OutlierSummary(), 4, 40)
    weighted = sum(c.weighted_score for c in score.components)
    assert weighted == pytest.approx(score.score, abs=0.05)


def test_worse_data_can_never_score_higher() -> None:
    """Monotonicity: adding a problem must not increase the health score."""
    good_missing = MissingReport(total_cells=1000, missing_cells=10, missing_rate=0.01)
    bad_missing = MissingReport(total_cells=1000, missing_cells=200, missing_rate=0.20,
                                affected_columns=[])
    duplicates = DuplicateReport(total_rows=1000, duplicate_rows=0, duplicate_rate=0.0)

    higher = health.compute_health_score(good_missing, duplicates, [], [], OutlierSummary(), 10, 100)
    lower = health.compute_health_score(bad_missing, duplicates, [], [], OutlierSummary(), 10, 100)
    assert lower.score < higher.score


def test_duplicates_and_constant_columns_reduce_the_score() -> None:
    missing, clean_duplicates = _clean_inputs()
    dirty_duplicates = DuplicateReport(total_rows=100, duplicate_rows=10, duplicate_rate=0.10)
    constants = [VarianceColumn("c", 1, "A", 1.0, True, True, "")]
    dtype_issues = [
        DtypeIssue("t", "numeric_stored_as_text", "object", "float64", "detail", 0.95)
    ]

    baseline = health.compute_health_score(missing, clean_duplicates, [], [], OutlierSummary(), 4, 40)
    degraded = health.compute_health_score(
        missing, dirty_duplicates, constants, dtype_issues, OutlierSummary(), 4, 40
    )
    assert degraded.score < baseline.score


def test_health_score_notes_state_that_leakage_is_excluded() -> None:
    missing, duplicates = _clean_inputs()
    score = health.compute_health_score(missing, duplicates, [], [], OutlierSummary(), 1, 10)
    assert any("leakage" in note.lower() for note in score.notes)
    assert any("weighted average" in note.lower() for note in score.notes)


def test_health_grade_bands() -> None:
    assert health._grade_for(95) == "Excellent"
    assert health._grade_for(80) == "Good"
    assert health._grade_for(65) == "Fair"
    assert health._grade_for(45) == "Poor"
    assert health._grade_for(10) == "Critical"


def test_components_explain_their_deductions() -> None:
    missing = MissingReport(
        total_cells=1000, missing_cells=600, missing_rate=0.60,
        affected_columns=[], columns_with_missing=3,
    )
    component = health.completeness_component(missing, n_columns=5)
    assert component.score < 50
    assert component.measurement
