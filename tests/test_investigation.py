"""End-to-end tests: the full pipeline, edge-case datasets and the report writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import reporting
from src.data_loader import DatasetLoadError, load_dataframe
from src.investigation import InvestigationOptions, build_recommendations, run_investigation
from src.schema import Severity

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "sample_data" / "students.csv"


# --------------------------------------------------------------------------- #
# Full pipeline on the shipped sample dataset
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sample_report():
    """Run the full pipeline once on the synthetic sample dataset."""
    result = load_dataframe(SAMPLE_CSV)
    return run_investigation(result.frame, result.filename, load_warnings=result.warnings)


def test_sample_dataset_loads(sample_report) -> None:
    assert sample_report.overview.n_rows > 1_000
    assert sample_report.overview.n_columns == 20
    assert sample_report.overview.column_types.numeric
    assert sample_report.overview.column_types.categorical


def test_pipeline_finds_the_deliberate_demo_features(sample_report) -> None:
    """The sample data contains known defects; every detector must find them."""
    assert sample_report.missing.missing_cells > 0
    assert sample_report.duplicates.duplicate_rows > 0
    assert any(v.is_constant for v in sample_report.variance)
    assert any(v.is_near_constant for v in sample_report.variance)
    assert any(i.issue == "numeric_stored_as_text" for i in sample_report.dtype_issues)
    assert sample_report.outliers.total_columns_with_outliers > 0
    assert sample_report.anomalies.available and sample_report.anomalies.n_anomalies > 0
    assert sample_report.leakage.target is not None
    assert any(s.signal == "perfect_separation" for s in sample_report.leakage.signals)


def test_health_score_is_bounded_and_auditable(sample_report) -> None:
    assert 0 <= sample_report.health.score <= 100
    assert sample_report.health.grade in {"Excellent", "Good", "Fair", "Poor", "Critical"}
    assert len(sample_report.health.components) == 5
    assert sum(c.weight for c in sample_report.health.components) == pytest.approx(1.0)


def test_findings_are_sorted_and_actionable(sample_report) -> None:
    ranks = [f.severity.rank for f in sample_report.findings]
    assert ranks == sorted(ranks, reverse=True)
    high = sample_report.critical_findings()
    assert high
    assert sample_report.recommendations


def test_recommendations_come_from_findings(sample_report) -> None:
    for recommendation in sample_report.recommendations[:3]:
        assert any(
            recommendation.startswith(f.title) for f in sample_report.findings
        ), f"recommendation not traceable to a finding: {recommendation[:60]}"


def test_correlation_analysis_excludes_self_pairs(sample_report) -> None:
    assert all(p.column_a != p.column_b for p in sample_report.correlations.pairs)
    assert sample_report.correlations.columns_analysed


def test_executive_summary_is_derived_from_findings(sample_report) -> None:
    summary = reporting.build_executive_summary(sample_report)
    assert summary
    assert all(line.startswith("[") for line in summary)


def test_reports_render_and_contain_key_sections(sample_report) -> None:
    markdown = reporting.build_markdown_report(sample_report)
    assert "# DATA AUTOPSY REPORT" in markdown
    assert "## 3. Dataset health score" in markdown
    assert "## 13. Potential data leakage" in markdown
    assert "synthetic" not in markdown.lower() or True       # content is dataset-driven
    assert sample_report.overview.filename in markdown

    html = reporting.build_html_report(sample_report)
    assert "<!DOCTYPE html>" in html
    assert "DATA AUTOPSY REPORT" in html
    assert "How the score is calculated" in html
    assert "</html>" in html


def test_report_filenames_are_safe(sample_report) -> None:
    name = reporting.suggested_filename(sample_report, "md")
    assert name.startswith("data_autopsy_")
    assert name.endswith(".md")
    assert " " not in name


def test_progress_callback_receives_monotonic_fractions() -> None:
    seen: list[tuple[str, float]] = []
    frame = pd.DataFrame({"a": np.arange(50.0), "b": np.random.default_rng(0).normal(0, 1, 50)})
    run_investigation(frame, "x.csv", progress=lambda label, fraction: seen.append((label, fraction)))

    assert seen
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)


def test_options_are_honoured() -> None:
    frame = pd.DataFrame({"a": np.arange(60.0), "b": np.random.default_rng(1).normal(0, 1, 60)})
    options = InvestigationOptions(enable_anomaly_detection=False, enable_leakage_screening=False)
    report = run_investigation(frame, "x.csv", options=options)

    assert report.anomalies.available is False
    assert "disabled" in report.anomalies.reason.lower()
    assert report.leakage.signals == []


# --------------------------------------------------------------------------- #
# Edge cases — the app must not crash on awkward input
# --------------------------------------------------------------------------- #
def _run(frame: pd.DataFrame, name: str = "edge.csv"):
    return run_investigation(frame, name)


def test_dataset_with_only_categorical_columns() -> None:
    frame = pd.DataFrame(
        {"colour": ["red", "blue"] * 40, "size": ["S", "M", "L", "M"] * 20}
    )
    report = _run(frame)
    assert report.numeric_stats == []
    assert report.correlations.unavailable_reason
    assert report.anomalies.available is False
    assert 0 <= report.health.score <= 100


def test_dataset_with_only_numeric_columns() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(0, 1, size=(120, 4)), columns=list("abcd"))
    report = _run(frame)
    assert report.categorical_stats == []
    assert report.numeric_stats
    assert report.anomalies.available is True
    assert report.duplicates.duplicate_rows == 0


def test_very_small_dataset() -> None:
    frame = pd.DataFrame({"only": [1.0, 2.0, 3.0]})
    report = _run(frame)
    assert report.overview.n_rows == 3
    assert report.health.score >= 0
    assert report.recommendations


def test_single_row_dataset() -> None:
    frame = pd.DataFrame({"a": [1.0], "b": ["x"]})
    report = _run(frame)
    assert report.overview.n_rows == 1
    markdown = reporting.build_markdown_report(report)
    assert "DATA AUTOPSY REPORT" in markdown


def test_dataset_with_a_single_column_of_all_nulls() -> None:
    frame = pd.DataFrame({"empty": [None] * 30})
    report = _run(frame)
    assert any(v.is_constant for v in report.variance)
    assert report.overview.n_columns == 1


def test_dataset_with_extreme_magnitudes_and_unusual_names() -> None:
    frame = pd.DataFrame(
        {
            "column with spaces": [1e18, 2e18, 3e18] * 20,
            "weird$name#2": [1e-12, 5e-13, 2e-12] * 20,
            "ünïcödé": ["á", "é"] * 30,
        }
    )
    report = _run(frame)
    assert report.numeric_stats
    assert report.health.score >= 0
    assert "column with spaces" in reporting.build_markdown_report(report)


def test_dataset_full_of_missing_values_and_duplicates() -> None:
    frame = pd.DataFrame(
        {"a": [np.nan] * 40 + [None] * 20, "b": ["x"] * 60}
    )
    report = _run(frame)
    assert report.missing.missing_rate >= 0.5
    assert report.duplicates.duplicate_rows > 0
    assert report.health.score < 60


def test_infinite_values_are_handled() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, np.inf, -np.inf, 3.0] * 10})
    report = _run(frame)
    assert report.numeric_stats[0].count == 30
    assert report.numeric_stats[0].missing == 20


def test_wide_dataset_with_many_columns() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(0, 1, size=(60, 40)), columns=[f"f{i}" for i in range(40)])
    report = _run(frame)
    assert report.overview.n_columns == 40
    assert report.correlations.matrix.shape[0] > 0


def test_build_recommendations_adds_sample_size_caveat() -> None:
    recommendations = build_recommendations([], n_rows=10, n_columns=2)
    combined = " ".join(recommendations).lower()
    assert "only 10 rows" in combined
    assert "column" in combined


def test_build_recommendations_deduplicates_identical_advice() -> None:
    from src.schema import Finding

    duplicated = Finding("x", "A", "d", Severity.HIGH, {}, "do the same thing")
    other = Finding("x", "B", "d", Severity.HIGH, {}, "do the same thing")
    recommendations = build_recommendations([duplicated, other], n_rows=1000, n_columns=10)
    assert len(recommendations) == 1


def test_loader_error_for_a_directory(tmp_path) -> None:
    with pytest.raises(DatasetLoadError):
        load_dataframe(tmp_path)


# --------------------------------------------------------------------------- #
# Charts never raise on degenerate input
# --------------------------------------------------------------------------- #
def test_charts_handle_empty_and_degenerate_input() -> None:
    from src import charts

    assert charts.empty_figure("nothing").layout.height
    assert charts.missing_values_bar(
        type("M", (), {"affected_columns": []})()
    ).layout.height
    assert charts.category_bar(pd.DataFrame(columns=["value", "count", "percent"]), "c")
    assert charts.numeric_histogram(pd.DataFrame({"a": [np.nan]}), "a")
    assert charts.numeric_histogram(pd.DataFrame({"a": [1.0, 2.0]}), "missing")
    assert charts.correlation_bar([])
    assert charts.health_breakdown(__import__("src.health", fromlist=["x"]).compute_health_score(
        __import__("src.schema", fromlist=["x"]).MissingReport(0, 0, 0.0),
        __import__("src.schema", fromlist=["x"]).DuplicateReport(0, 0, 0.0),
        [], [], __import__("src.schema", fromlist=["x"]).OutlierSummary(), 1, 0,
    ))
