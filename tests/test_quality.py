"""Tests for data loading, missing values, duplicates, variance and dtypes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import quality
from src.data_loader import (
    DatasetLoadError,
    build_overview,
    classify_columns,
    detect_dtype_issues,
    load_dataframe,
)
from src.schema import Severity
from src.utils import (
    format_percent,
    human_bytes,
    is_probably_datetime_text,
    is_probably_numeric_text,
    safe_divide,
    tokenize_name,
)


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def test_missing_counts_and_rates_are_exact() -> None:
    frame = pd.DataFrame(
        {
            "a": [1, 2, np.nan, 4],
            "b": [np.nan, np.nan, np.nan, np.nan],
            "c": [1, 2, 3, 4],
        }
    )
    report = quality.analyze_missing(frame)

    assert report.total_cells == 12
    assert report.missing_cells == 5
    assert report.missing_rate == pytest.approx(5 / 12)
    assert report.columns_with_missing == 2
    # Column "b" is empty in every row, so no row is complete.
    assert report.rows_with_missing == 4
    assert report.complete_rows == 0
    assert report.complete_row_rate == pytest.approx(0.0)

    by_name = {c.column: c for c in report.affected_columns}
    assert by_name["b"].missing_rate == 1.0
    assert by_name["a"].missing == 1


def test_missing_findings_flag_severe_columns_and_recommend_investigation() -> None:
    frame = pd.DataFrame({"x": [1, np.nan, np.nan, np.nan, np.nan, 6], "y": range(6)})
    report = quality.analyze_missing(frame)
    findings = quality.missing_findings(report, n_columns=2, n_rows=6)

    titles = " ".join(f.title for f in findings)
    assert "missing values across" in titles
    column_finding = next(f for f in findings if "x" in f.title)
    assert column_finding.severity.rank >= Severity.MODERATE.rank
    assert "missing" in column_finding.recommendation.lower()


def test_no_missing_values_mentions_sentinel_possibility() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    findings = quality.missing_findings(quality.analyze_missing(frame), 2, 3)
    assert len(findings) == 1
    assert findings[0].severity is Severity.INFO
    assert "sentinel" in findings[0].detail.lower()


def test_missing_rate_boundaries_map_to_expected_severities() -> None:
    assert quality._missing_severity(0.01) is Severity.INFO
    assert quality._missing_severity(0.10) is Severity.LOW
    assert quality._missing_severity(0.30) is Severity.MODERATE
    assert quality._missing_severity(0.60) is Severity.HIGH
    assert quality._missing_severity(0.95) is Severity.CRITICAL


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #
def test_duplicate_rows_counted_once_per_repeat() -> None:
    frame = pd.DataFrame({"a": [1, 1, 1, 2], "b": ["x", "x", "x", "y"]})
    report = quality.analyze_duplicates(frame)

    assert report.total_rows == 4
    # Rows 1 and 2 duplicate row 0 -> two extra rows beyond the first occurrence.
    assert report.duplicate_rows == 2
    assert report.duplicate_rate == pytest.approx(0.5)
    assert report.unique_rows == 2
    assert report.duplicate_group_count == 1
    assert len(report.examples) == 3          # every member of the duplicated group


def test_no_duplicates_reports_clean_state() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    report = quality.analyze_duplicates(frame)
    assert report.duplicate_rows == 0
    findings = quality.duplicate_findings(report)
    assert findings[0].severity is Severity.INFO
    assert report.examples.empty


def test_duplicate_subset_analysis() -> None:
    frame = pd.DataFrame({"id": [1, 1, 2], "value": [10.0, 20.0, 30.0]})
    report = quality.analyze_duplicates(frame, subset=["id"])
    assert report.duplicate_rows == 1


# --------------------------------------------------------------------------- #
# Constant / near-constant
# --------------------------------------------------------------------------- #
def test_constant_and_near_constant_detection() -> None:
    frame = pd.DataFrame(
        {
            "constant": ["A"] * 200,
            "near_constant": ["A"] * 199 + ["B"],
            "varied": list(range(200)),
        }
    )
    results = {r.column: r for r in quality.analyze_variance(frame)}

    assert results["constant"].is_constant is True
    assert results["near_constant"].is_near_constant is True
    assert results["near_constant"].is_constant is False
    assert results["near_constant"].dominant_rate == pytest.approx(0.995)
    assert "varied" not in results


def test_variance_findings_explain_limited_usefulness() -> None:
    frame = pd.DataFrame({"c": [1] * 100, "n": [1] * 99 + [2], "v": range(100)})
    findings = quality.variance_findings(quality.analyze_variance(frame), n_columns=3)
    text = " ".join(f.detail + (f.recommendation or "") for f in findings).lower()
    assert "constant" in text
    assert "predictive" in text or "useful" in text or "signal" in text


def test_all_empty_column_is_reported_as_constant() -> None:
    frame = pd.DataFrame({"empty": [np.nan] * 10, "x": range(10)})
    results = {r.column: r for r in quality.analyze_variance(frame)}
    assert results["empty"].is_constant is True
    assert "empty" in results["empty"].note.lower()


# --------------------------------------------------------------------------- #
# Data-type drift
# --------------------------------------------------------------------------- #
def test_numeric_text_detection_and_dtype_issue() -> None:
    frame = pd.DataFrame(
        {
            "income": ["1,200", "3,400", "5,600", "7,800"],
            "name": ["a", "b", "c", "d"],
        }
    )
    assert is_probably_numeric_text(frame["income"]) is True
    assert is_probably_numeric_text(frame["name"]) is False

    types = classify_columns(frame)
    issues = detect_dtype_issues(frame, types)
    kinds = {issue.issue for issue in issues}
    assert "numeric_stored_as_text" in kinds

    findings = quality.dtype_issue_findings(issues, n_columns=2)
    assert any("text" in f.title.lower() for f in findings)
    assert findings[0].severity.rank >= Severity.MODERATE.rank


def test_datetime_text_detection() -> None:
    dates = pd.Series(["2024-01-31", "2024-02-29", "2024-03-15"])
    assert is_probably_datetime_text(dates) is True
    assert is_probably_datetime_text(pd.Series(["20240131", "20240229"])) is False


def test_mixed_type_column_is_flagged() -> None:
    frame = pd.DataFrame({"score": ["72", "68", "not available", "81", "absent", "75", "90", "60"]})
    issues = detect_dtype_issues(frame, classify_columns(frame))
    assert any(issue.issue == "mixed_types" for issue in issues)


def test_clean_frame_reports_no_dtype_issues() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    findings = quality.dtype_issue_findings(detect_dtype_issues(frame, classify_columns(frame)), 2)
    assert findings[0].severity is Severity.INFO


# --------------------------------------------------------------------------- #
# Loading robustness
# --------------------------------------------------------------------------- #
def test_load_csv_with_semicolon_delimiter(tmp_path) -> None:
    path = tmp_path / "semi.csv"
    path.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    result = load_dataframe(path)
    assert result.delimiter == ";"
    assert list(result.frame.columns) == ["a", "b"]
    assert len(result.frame) == 2


def test_load_csv_with_messy_headers_and_blank_rows(tmp_path) -> None:
    path = tmp_path / "messy.csv"
    path.write_text(" age , age ,name\n18,18,A\n\n19,19,B\n", encoding="utf-8")
    result = load_dataframe(path)
    assert list(result.frame.columns)[:3] == ["age", "age_duplicate_1", "name"]
    assert len(result.frame) == 2
    assert any("normalised" in w for w in result.warnings)


def test_load_csv_with_unnamed_index_column(tmp_path) -> None:
    path = tmp_path / "indexed.csv"
    path.write_text(",value\n0,10\n1,20\n", encoding="utf-8")
    result = load_dataframe(path)
    assert list(result.frame.columns) == ["value"]
    assert any("index" in w.lower() for w in result.warnings)


def test_empty_file_raises_dataset_load_error(tmp_path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetLoadError):
        load_dataframe(path)


def test_binary_file_raises_dataset_load_error(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    with pytest.raises(DatasetLoadError):
        load_dataframe(path)


def test_missing_file_raises_dataset_load_error() -> None:
    with pytest.raises(DatasetLoadError):
        load_dataframe("/nonexistent/path/file.csv")


def test_load_from_bytes_and_odd_encoding() -> None:
    payload = "name,city\nJosé,Nairobi\nAna,Mombasa\n".encode("cp1252")
    result = load_dataframe(payload, filename="latin.csv")
    assert len(result.frame) == 2
    assert result.frame["name"].tolist() == ["José", "Ana"]


def test_single_column_file_still_loads(tmp_path) -> None:
    path = tmp_path / "single.csv"
    path.write_text("value\n1\n2\n3\n", encoding="utf-8")
    result = load_dataframe(path)
    assert result.frame.shape == (3, 1)


def test_overview_reports_structure_and_roles() -> None:
    frame = pd.DataFrame(
        {
            "n": [1.0, 2.0, 3.0],
            "t": ["a", "b", "a"],
            "flag": [True, False, True],
            "when": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        }
    )
    overview = build_overview(frame, "tiny.csv")

    assert overview.n_rows == 3
    assert overview.n_columns == 4
    assert overview.memory_bytes > 0
    assert overview.column_types.numeric == ["n"]
    assert overview.column_types.categorical == ["t"]
    assert overview.column_types.boolean == ["flag"]
    assert overview.column_types.datetime == ["when"]
    assert set(overview.dtype_table["role"]) == {"numeric", "categorical", "boolean", "datetime"}


# --------------------------------------------------------------------------- #
# Small utilities (used by every module above)
# --------------------------------------------------------------------------- #
def test_safe_divide_handles_degenerate_input() -> None:
    assert safe_divide(1, 2) == 0.5
    assert safe_divide(1, 0) == 0.0
    assert safe_divide(1, 0, default=-1) == -1
    assert safe_divide(float("nan"), 2) == 0.0
    assert safe_divide(None, 2) == 0.0


def test_tokenize_name_matches_whole_words_only() -> None:
    assert tokenize_name("exam_score") == ["exam", "score"]
    assert tokenize_name("customerID") == ["customer", "id"]
    assert tokenize_name("gender") == ["gender"]
    assert "end" not in tokenize_name("gender")
    assert "end" not in tokenize_name("attendance_rate")


def test_formatting_helpers() -> None:
    assert format_percent(0.1234) == "12.3%"
    assert human_bytes(2048) == "2.0 KiB"
    assert human_bytes(None) == "n/a"
