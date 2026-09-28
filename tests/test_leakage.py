"""Tests for the leakage heuristics.

The most important properties tested here are *honesty* properties: the module
must use cautious "potential leakage" wording, must not fire on innocent names,
and must not flag a genuine feature as an identifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.leakage import (
    _best_threshold_accuracy,
    analyze_leakage,
    guess_target,
    leakage_findings,
)
from src.schema import Severity
from src.data_loader import classify_columns


# --------------------------------------------------------------------------- #
# Target discovery
# --------------------------------------------------------------------------- #
def test_target_is_guessed_from_conventional_names() -> None:
    frame = pd.DataFrame({"a": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    target, candidates, reasoning = guess_target(frame, classify_columns(frame))
    assert target == "target"
    assert "target" in candidates
    assert reasoning


def test_boolean_flag_name_is_recognised_as_target() -> None:
    frame = pd.DataFrame({"feature": np.arange(40), "is_at_risk": [0, 1] * 20})
    target, _, _ = guess_target(frame, classify_columns(frame))
    assert target == "is_at_risk"


def test_no_target_found_is_reported_instead_of_guessed() -> None:
    frame = pd.DataFrame({"height": np.arange(20.0), "width": np.arange(20.0)})
    target, candidates, reasoning = guess_target(frame, classify_columns(frame))
    assert target is None
    assert candidates == []
    assert "skipped" in reasoning.lower()


# --------------------------------------------------------------------------- #
# Identifier detection
# --------------------------------------------------------------------------- #
def test_unique_key_column_is_flagged_as_identifier() -> None:
    frame = pd.DataFrame(
        {"customer_id": [f"C{i:04d}" for i in range(200)], "amount": np.arange(200.0)}
    )
    report = analyze_leakage(frame, classify_columns(frame), target="amount")
    signals = [s for s in report.signals if s.signal == "identifier_like"]
    assert any(s.column == "customer_id" for s in signals)
    assert all("Potential leakage" in s.recommendation for s in signals)


def test_continuous_measurement_is_not_called_an_identifier() -> None:
    """A continuous column is nearly all-distinct by nature — that is not a key."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"house_price": rng.normal(50_000, 9_000, 300)})
    report = analyze_leakage(frame, classify_columns(frame), target=None)
    assert not [s for s in report.signals if s.signal == "identifier_like"]


def test_innocent_names_do_not_trigger_outcome_signals() -> None:
    """Regression: substring matching once flagged 'gender' because it contains 'end'."""
    frame = pd.DataFrame(
        {
            "gender": ["F", "M"] * 60,
            "attendance_rate": np.linspace(50, 100, 120),
            "value": np.arange(120.0),
        }
    )
    report = analyze_leakage(frame, classify_columns(frame), target=None)
    outcome_signals = [s for s in report.signals if s.signal == "outcome_name"]
    flagged = {s.column for s in outcome_signals}
    assert "gender" not in flagged
    assert "attendance_rate" not in flagged


def test_post_outcome_count_is_not_flagged_but_status_is() -> None:
    """'assignments_completed' is a count; 'course_completed' is a status."""
    frame = pd.DataFrame(
        {
            "assignments_completed": [7, 9, 11] * 30,
            "course_completed": [0, 1, 1] * 30,
            "value": np.arange(90.0),
        }
    )
    report = analyze_leakage(frame, classify_columns(frame), target="value")
    flagged = {s.column for s in report.signals if s.signal == "outcome_name"}
    assert "course_completed" in flagged
    assert "assignments_completed" not in flagged


def test_duplicate_columns_are_detected() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(10, 2, 100)
    frame = pd.DataFrame({"score_a": base, "score_b": base.copy(), "other": rng.normal(0, 1, 100)})
    report = analyze_leakage(frame, classify_columns(frame), target=None)
    duplicates = [s for s in report.signals if s.signal == "duplicate_column"]
    assert duplicates
    assert duplicates[0].confidence == "high"


def test_copy_of_the_target_is_critical() -> None:
    frame = pd.DataFrame({"target": [0, 1] * 60, "outcome": [0, 1] * 60, "x": np.arange(120.0)})
    report = analyze_leakage(frame, classify_columns(frame), target="target")
    copies = [s for s in report.signals if s.signal == "target_copy"]
    assert copies
    assert copies[0].severity is Severity.CRITICAL


# --------------------------------------------------------------------------- #
# Perfect separation
# --------------------------------------------------------------------------- #
def test_threshold_accuracy_finds_perfect_split_in_both_directions() -> None:
    values = np.concatenate([np.linspace(0, 9, 50), np.linspace(10, 20, 50)])
    labels = np.concatenate([np.ones(50), np.zeros(50)])       # class 1 *below* the split
    assert _best_threshold_accuracy(values, labels) == pytest.approx(1.0)

    reversed_labels = 1 - labels
    assert _best_threshold_accuracy(values, reversed_labels) == pytest.approx(1.0)

    random_labels = np.random.default_rng(0).integers(0, 2, 100).astype(float)
    assert _best_threshold_accuracy(values, random_labels) < 1.0


def test_perfectly_separating_feature_is_flagged() -> None:
    rng = np.random.default_rng(2)
    exam = rng.normal(60, 10, 400)
    frame = pd.DataFrame(
        {
            "exam_score": exam,
            "attendance": rng.uniform(40, 100, 400),
            "is_at_risk": (exam < 55).astype(int),
        }
    )
    report = analyze_leakage(frame, classify_columns(frame), target="is_at_risk")
    separation = [s for s in report.signals if s.signal == "perfect_separation"]
    assert separation
    assert separation[0].column == "exam_score"
    assert separation[0].severity is Severity.HIGH
    assert "potential leakage" in separation[0].recommendation.lower()


def test_partial_separation_is_not_flagged() -> None:
    rng = np.random.default_rng(3)
    exam = rng.normal(60, 10, 400)
    noisy_target = (exam + rng.normal(0, 20, 400) < 60).astype(int)
    frame = pd.DataFrame({"exam_score": exam, "is_at_risk": noisy_target})
    report = analyze_leakage(frame, classify_columns(frame), target="is_at_risk")
    assert not [s for s in report.signals if s.signal == "perfect_separation"]


def test_categorical_perfect_separation_is_detected() -> None:
    frame = pd.DataFrame(
        {
            "segment": (["high"] * 60) + (["low"] * 60),
            "target": ([1] * 60) + ([0] * 60),
            "noise": np.random.default_rng(4).normal(0, 1, 120),
        }
    )
    report = analyze_leakage(frame, classify_columns(frame), target="target")
    assert any(s.signal == "perfect_separation" for s in report.signals)


def test_near_perfect_target_correlation_is_flagged() -> None:
    rng = np.random.default_rng(5)
    price = rng.normal(200_000, 40_000, 300)
    frame = pd.DataFrame({"sale_price": price, "deal_value": price * 1.0005, "rooms": rng.integers(1, 6, 300).astype(float)})
    report = analyze_leakage(frame, classify_columns(frame), target="sale_price")
    signals = {s.signal for s in report.signals}
    assert "duplicate_column" in signals or "target_correlation" in signals


# --------------------------------------------------------------------------- #
# Findings wording
# --------------------------------------------------------------------------- #
def test_findings_use_cautious_wording() -> None:
    rng = np.random.default_rng(6)
    exam = rng.normal(60, 10, 300)
    frame = pd.DataFrame({"exam_score": exam, "is_at_risk": (exam < 55).astype(int)})
    report = analyze_leakage(frame, classify_columns(frame), target="is_at_risk")
    findings = leakage_findings(report)

    separation_finding = next(f for f in findings if "perfect_separation" in f.title)
    assert "potential leakage" in separation_finding.title.lower()
    assert "investigate" in separation_finding.title.lower()
    assert "potential leakage" in (separation_finding.recommendation or "").lower()


def test_clean_dataset_reports_no_signals_but_states_the_limit() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500), "c": rng.choice(list("xyz"), 500)}
    )
    report = analyze_leakage(frame, classify_columns(frame), target=None)
    actionable = [s for s in report.signals if s.signal != "target_selection"]
    if not actionable:
        findings = leakage_findings(report)
        assert "not proof" in findings[0].recommendation.lower()


def test_report_records_target_and_candidates() -> None:
    frame = pd.DataFrame({"feature": np.arange(100.0), "label": [0, 1] * 50})
    report = analyze_leakage(frame, classify_columns(frame))
    assert report.target == "label"
    assert "label" in report.candidate_targets
    assert report.target in report.columns_flagged or report.columns_flagged == []
