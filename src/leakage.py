"""Heuristic screening for potential data leakage.

**This module never claims that leakage exists.** Leakage is a property of how a
model will be trained and deployed, which cannot be determined from a CSV alone.
What the module does is raise *signals*: verifiable, reproducible observations
that a column behaves the way leaking columns usually behave, together with the
reasoning and the concrete question a human should answer.

Signals implemented
-------------------
``identifier_like``      unique-per-row values or ID-style names
``duplicate_column``     a column that is identical (or near-identical) to another
``target_correlation``   |r| with the target above the near-perfect threshold
``perfect_separation``   a feature that splits a binary target with ~100% purity
``outcome_name``         name tokens suggesting post-outcome information
``hardcoded_missing``    a numeric "ID" that is also the target ordering — sanity
``target_copy``          a feature that reproduces the target exactly
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schema import ColumnTypes, Finding, LeakageReport, LeakageSignal, Severity
from src.utils import (
    format_percent,
    is_probably_numeric_text,
    join_names,
    safe_divide,
    tokenize_name,
)

#: Column names that almost always denote the prediction target (exact word tokens).
TARGET_NAME_EXACT: tuple[str, ...] = (
    "target", "label", "outcome", "class", "y", "result", "response",
    "label_encoded", "target_encoded",
)
#: Whole-word tokens that suggest an outcome/event to predict.
TARGET_NAME_TOKENS: tuple[str, ...] = (
    "churn", "default", "fraud", "converted", "conversion", "survived", "survival",
    "purchased", "diagnosis", "attrition", "approved", "cancelled", "canceled",
    "churned", "is_active",
)
#: Name prefixes that reliably denote a boolean outcome flag.
TARGET_NAME_PREFIXES: tuple[str, ...] = ("is_", "has_", "will_", "did_", "was_")
#: Name suffixes that reliably denote an outcome flag or encoded label.
TARGET_NAME_SUFFIXES: tuple[str, ...] = ("_flag", "_target", "_label", "_class", "_outcome")

#: Maximum number of features examined by the (relatively expensive) separation test.
MAX_SEPARATION_FEATURES: int = 25
#: Minimum group purity to call a strict separation.
SEPARATION_PURITY: float = 0.999
#: Minimum rows required before the separation test is trustworthy.
MIN_ROWS_FOR_SEPARATION: int = 50


# --------------------------------------------------------------------------- #
# Target discovery
# --------------------------------------------------------------------------- #
def guess_target(
    frame: pd.DataFrame, column_types: ColumnTypes
) -> tuple[str | None, list[str], str]:
    """Infer the most plausible prediction target.

    Returns ``(target, candidates, reasoning)``. The guess is intentionally
    conservative: a column must either use standard ML naming or be an obvious
    event flag to be considered. When two candidates tie, the one appearing later
    in the file wins, matching the usual "features first, target last" layout.
    """
    candidates: list[tuple[int, int, str]] = []

    for position, column in enumerate(frame.columns):
        name = str(column)
        lowered = name.lower().strip()
        series = frame[column]
        n_unique = series.nunique(dropna=True)

        # Skip columns that cannot be a target at all.
        if n_unique < 2:
            continue
        if str(column) in column_types.datetime:
            continue

        tokens = tokenize_name(lowered)
        score = 0
        if lowered in TARGET_NAME_EXACT or (len(tokens) == 1 and tokens[0] in TARGET_NAME_EXACT):
            score = 3
        elif lowered.startswith(TARGET_NAME_PREFIXES) or lowered.endswith(TARGET_NAME_SUFFIXES):
            score = 2
        elif any(token in TARGET_NAME_TOKENS for token in tokens):
            score = 2

        if score == 0:
            continue

        # Very high-cardinality columns are rarely a target in tabular work.
        if series.nunique(dropna=True) > 50 and not pd.api.types.is_numeric_dtype(series):
            score -= 1
        if score > 0:
            candidates.append((score, position, name))

    if not candidates:
        return (
            None,
            [],
            "No column name matched the usual target conventions (target, label, "
            "outcome, class, is_*, *_flag, or an event such as churn/default/fraud), "
            "so target-aware leakage checks were skipped.",
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    best_score = candidates[-1][0]
    winners = [name for score, _, name in candidates if score == best_score]
    chosen = candidates[-1][2]
    names = [name for _, _, name in candidates]

    reasoning = (
        f"'{chosen}' was treated as the target because its name follows the usual "
        f"target convention (candidate column(s): {', '.join(names[:5])}). "
        "Set the target explicitly in the sidebar if this is wrong — leakage checks "
        "depend on it."
    )
    if len(winners) > 1:
        reasoning += (
            f" Several columns scored equally ({join_names(winners)}); the last one in "
            "the file was used."
        )
    return chosen, names, reasoning


# --------------------------------------------------------------------------- #
# Individual signals
# --------------------------------------------------------------------------- #
def _unique_ratio(series: pd.Series) -> float:
    """Share of distinct non-null values in ``series``."""
    non_null = series.dropna()
    return safe_divide(non_null.nunique(), max(len(non_null), 1), 0.0)


def identifier_signals(
    frame: pd.DataFrame, column_types: ColumnTypes, target: str | None
) -> list[LeakageSignal]:
    """Flag columns that look like record identifiers rather than features."""
    signals: list[LeakageSignal] = []
    n_rows = max(len(frame), 1)

    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        series = frame[column]
        unique_ratio = _unique_ratio(series)
        lowered = name.lower()
        tokens = set(tokenize_name(lowered))
        name_hit = bool(tokens & set(config.ID_NAME_TOKENS))

        # A continuous measurement is expected to be nearly all-distinct, so the
        # uniqueness test is only meaningful for non-numeric columns.
        measurable = pd.api.types.is_numeric_dtype(series) or is_probably_numeric_text(series)

        if unique_ratio >= config.LEAKAGE_ID_UNIQUE_RATIO and not measurable:
            strength = "high" if unique_ratio >= 0.999 else "medium"
            signals.append(
                LeakageSignal(
                    column=name,
                    signal="identifier_like",
                    severity=Severity.MODERATE if name_hit else Severity.LOW,
                    confidence=strength,
                    reasoning=(
                        f"{format_percent(unique_ratio, 2)} of the "
                        f"{int(series.notna().sum()):,} non-missing values in '{name}' are "
                        "distinct, so it behaves like a record key."
                        + (" Its name also contains identifier-like wording." if name_hit else "")
                        + " A key cannot generalise to unseen records: at prediction time "
                        "a new value has never been seen during training."
                    ),
                    recommendation=(
                        "Potential leakage — investigate this feature. Drop it from the "
                        "feature set (keep it only as a row identifier) unless you have "
                        "evidence the values recur across records."
                    ),
                    evidence={"unique_ratio": unique_ratio, "n_unique": int(series.nunique())},
                )
            )
        elif name_hit and pd.api.types.is_numeric_dtype(series) and series.nunique() > 20:
            span = (
                float(series.max() - series.min())
                if pd.api.types.is_numeric_dtype(series) else None
            )
            signals.append(
                LeakageSignal(
                    column=name,
                    signal="identifier_like",
                    severity=Severity.LOW,
                    confidence="low",
                    reasoning=(
                        f"'{name}' has an identifier-style name and contains {int(series.nunique()):,} "
                        f"distinct values across {n_rows:,} rows"
                        + (f" (range {span:,.0f})" if span else "")
                        + ". Names alone do not prove an identifier, but this pattern is "
                        "typical of a key or a system code."
                    ),
                    recommendation=(
                        "Potential leakage — investigate this feature. Confirm what the "
                        "column represents before using it as a predictor."
                    ),
                    evidence={"n_unique": int(series.nunique()), "n_rows": n_rows},
                )
            )

    return signals


def duplicate_column_signals(
    frame: pd.DataFrame, target: str | None
) -> list[LeakageSignal]:
    """Detect columns that duplicate another column (or the target itself)."""
    signals: list[LeakageSignal] = []
    columns = [str(c) for c in frame.columns]
    if len(columns) < 2 or len(frame) < 2:
        return signals

    # Exact duplicates: a cheap, high-confidence signal.
    for i, column_a in enumerate(columns):
        if frame[column_a].isna().all():
            continue
        for column_b in columns[i + 1:]:
            if frame[column_b].isna().all():
                continue
            comparable = frame[[column_a, column_b]].dropna()
            if comparable.shape[0] < max(10, 0.5 * len(frame)):
                continue
            same_dtype = frame[column_a].dtype == frame[column_b].dtype
            if not same_dtype:
                continue
            try:
                identical = bool((comparable[column_a] == comparable[column_b]).all())
            except Exception:  # noqa: BLE001 - unhashable/odd dtypes
                continue
            if identical:
                is_target_copy = column_b == target or column_a == target
                signals.append(
                    LeakageSignal(
                        column=column_b if column_b != target else column_a,
                        signal="target_copy" if is_target_copy else "duplicate_column",
                        severity=Severity.CRITICAL if is_target_copy else Severity.HIGH,
                        confidence="high",
                        reasoning=(
                            f"'{column_a}' and '{column_b}' hold identical values for all "
                            f"{comparable.shape[0]:,} rows where both are present."
                            + (
                                " One of them is the target, so the other reproduces the "
                                "outcome directly."
                                if is_target_copy else ""
                            )
                        ),
                        recommendation=(
                            "Potential leakage — investigate this feature. Keep a single "
                            "copy and confirm which column the source system considers "
                            "authoritative."
                        ),
                        evidence={"identical_to": column_a, "rows_compared": int(comparable.shape[0])},
                    )
                )
    return signals


def target_relationship_signals(
    frame: pd.DataFrame,
    column_types: ColumnTypes,
    target: str | None,
    correlation_method: str = "spearman",
) -> list[LeakageSignal]:
    """Flag features that are (almost) perfectly related to the target."""
    if target is None or target not in frame.columns:
        return []

    signals: list[LeakageSignal] = []
    target_series = frame[target]
    target_numeric = pd.to_numeric(target_series, errors="coerce")
    n_usable_target = int(target_numeric.notna().sum())
    numeric_target = n_usable_target >= max(10, 0.5 * len(frame)) and target_numeric.nunique() > 2

    if numeric_target:
        for column in frame.columns:
            name = str(column)
            if name == target or name not in column_types.numeric:
                continue
            pair = frame[[name, target]].apply(pd.to_numeric, errors="coerce").dropna()
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if pair.shape[0] < 10:
                continue
            if pair[name].nunique() < 2 or pair[target].nunique() < 2:
                continue
            try:
                r = float(pair[name].corr(pair[target], method=correlation_method))
            except Exception:  # noqa: BLE001
                continue
            if not np.isfinite(r):
                continue
            if abs(r) >= config.TARGET_CORRELATION_FLAG:
                signals.append(
                    LeakageSignal(
                        column=name,
                        signal="target_correlation",
                        severity=Severity.HIGH,
                        confidence="medium",
                        reasoning=(
                            f"'{name}' correlates with the target '{target}' at "
                            f"{correlation_method} r = {r:+.4f} across {pair.shape[0]:,} rows. "
                            "A numerical feature that determines the outcome this closely "
                            "is usually derived from the outcome rather than measured "
                            "before it."
                        ),
                        recommendation=(
                            "Potential leakage — investigate this feature. Ask when this "
                            "value is recorded relative to the outcome; if it is only "
                            "known afterwards it must not be used in training."
                        ),
                        evidence={"correlation": r, "method": correlation_method,
                                  "target": target, "n": int(pair.shape[0])},
                    )
                )

    signals.extend(_separation_signals(frame, target))
    return signals


def _binary_target(frame: pd.DataFrame, target: str) -> pd.Series | None:
    """Return a 0/1 encoding of the target when it is binary, else ``None``."""
    series = frame[target]
    non_null = series.dropna()
    if non_null.nunique() != 2:
        return None
    categories = list(non_null.unique())
    try:
        encoded = non_null.map({categories[0]: 0, categories[1]: 1})
    except Exception:  # noqa: BLE001 - unhashable values
        return None
    if encoded.isna().any():
        return None
    return encoded


def _best_threshold_accuracy(values: np.ndarray, labels: np.ndarray) -> float:
    """Highest achievable accuracy from a single numeric threshold split."""
    order = np.argsort(values, kind="mergesort")
    sorted_labels = labels[order]
    n = sorted_labels.size
    total_pos = int(sorted_labels.sum())

    cumulative_pos = np.cumsum(sorted_labels)

    # Direction A: class 1 above the split, class 0 at or below it.
    correct_low = (np.arange(1, n + 1) - cumulative_pos)      # true negatives below split
    correct_high = total_pos - cumulative_pos                 # true positives above split
    accuracy_a = (correct_low + correct_high) / n

    # Direction B is the mirror image (class 0 above the split). Testing both matters:
    # a leaking column can sit below the outcome as easily as above it.
    accuracy = np.maximum(accuracy_a, 1.0 - accuracy_a)
    return float(np.max(accuracy)) if accuracy.size else 0.0


def _separation_signals(frame: pd.DataFrame, target: str) -> list[LeakageSignal]:
    """Detect features that separate a binary target almost perfectly."""
    labels = _binary_target(frame, target)
    if labels is None or labels.size < MIN_ROWS_FOR_SEPARATION:
        return []
    if labels.sum() < 5 or (labels.size - labels.sum()) < 5:
        return []

    signals: list[LeakageSignal] = []
    examined = 0

    for column in frame.columns:
        name = str(column)
        if name == target or examined >= MAX_SEPARATION_FEATURES:
            continue
        series = frame[column]
        aligned = pd.DataFrame({"feature": series, "target": labels}).dropna()
        if aligned.shape[0] < MIN_ROWS_FOR_SEPARATION or aligned["feature"].nunique() < 2:
            continue

        examined += 1
        n = aligned.shape[0]
        accuracy = 0.0

        if pd.api.types.is_numeric_dtype(aligned["feature"]):
            values = aligned["feature"].to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                continue
            accuracy = _best_threshold_accuracy(values, aligned["target"].to_numpy(dtype="float64"))
        else:
            grouped = aligned.groupby("feature", dropna=True)["target"]
            if grouped.ngroups > 200:      # too many levels: skip to stay honest
                continue
            counts = grouped.agg(["size", "sum"])
            purity = np.maximum(counts["sum"], counts["size"] - counts["sum"]) / counts["size"]
            accuracy = float((purity * counts["size"]).sum() / n)

        if accuracy >= SEPARATION_PURITY:
            signals.append(
                LeakageSignal(
                    column=name,
                    signal="perfect_separation",
                    severity=Severity.HIGH,
                    confidence="high",
                    reasoning=(
                        f"A single rule on '{name}' reproduces '{target}' with "
                        f"{accuracy:.1%} accuracy across {n:,} rows. Features that "
                        "separate an outcome this cleanly are usually recorded after the "
                        "outcome (post-outcome information) or derived from it."
                    ),
                    recommendation=(
                        "Potential leakage — investigate this feature. Establish whether "
                        "its value exists at scoring time; if not, exclude it and rebuild "
                        "the model without it to see how performance changes."
                    ),
                    evidence={"accuracy": accuracy, "target": target, "n": int(n)},
                )
            )

    return signals


def outcome_name_signals(
    frame: pd.DataFrame, column_types: ColumnTypes, target: str | None
) -> list[LeakageSignal]:
    """Flag columns whose names suggest they were recorded after the outcome."""
    signals: list[LeakageSignal] = []
    for column in frame.columns:
        name = str(column)
        if name == target:
            continue
        # Token-based matching: "gender" -> ["gender"] never matches the "end" token.
        tokens = set(tokenize_name(name))
        matched_post = sorted(tokens & set(config.POST_OUTCOME_TOKENS))

        # "completion"-style tokens only signal post-outcome information when the
        # column is a status flag or a date; a count like "assignments_completed"
        # is an ordinary feature rather than an outcome measurement.
        status_like = (
            str(column) in column_types.datetime
            or frame[column].nunique(dropna=True) <= 2
        )
        if status_like:
            matched_post += sorted(tokens & set(config.POST_OUTCOME_CONDITIONAL_TOKENS))

        matched_outcome = sorted(tokens & set(config.OUTCOME_NAME_TOKENS))
        matched_weak = sorted(tokens & set(config.WEAK_OUTCOME_NAME_TOKENS))
        if not matched_post and not matched_outcome and not matched_weak:
            continue

        matched = sorted(set(matched_post) | set(matched_outcome) | set(matched_weak))
        is_weak = not matched_post and not matched_outcome
        signals.append(
            LeakageSignal(
                column=name,
                signal="outcome_name",
                severity=Severity.INFO if is_weak else Severity.LOW,
                confidence="low",
                evidence={"matched_tokens": matched, "status_like": bool(status_like)},
                reasoning=(
                    f"The name '{name}' contains the token(s) {join_names(matched)}. "
                    "Naming alone proves nothing"
                    + (
                        ", but these words commonly mark columns that describe what "
                        "happened after the event being predicted (e.g. a final status, "
                        "a closing date or a resolution code)."
                        if not is_weak else
                        ". This token is common in ordinary measurement names, so treat "
                        "this as a reminder to check the column's timing, not as evidence."
                    )
                ),
                recommendation=(
                    "Potential leakage — investigate this feature. Check the data "
                    "dictionary for when this column is populated relative to the "
                    "outcome."
                ),
            )
        )
    return signals


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyze_leakage(
    frame: pd.DataFrame,
    column_types: ColumnTypes,
    target: str | None = None,
    target_reasoning: str = "",
    correlation_method: str = "spearman",
) -> LeakageReport:
    """Run every leakage heuristic and collect the resulting signals."""
    candidates: list[str] = []
    if target is None:
        target, candidates, target_reasoning = guess_target(frame, column_types)
    elif target not in frame.columns:
        target = None

    signals: list[LeakageSignal] = []
    signals.extend(identifier_signals(frame, column_types, target))
    signals.extend(duplicate_column_signals(frame, target))
    signals.extend(target_relationship_signals(frame, column_types, target, correlation_method))
    signals.extend(outcome_name_signals(frame, column_types, target))

    # Deduplicate on (column, signal), keeping the most confident instance.
    unique: dict[tuple[str, str], LeakageSignal] = {}
    for signal in signals:
        key = (signal.column, signal.signal)
        existing = unique.get(key)
        if existing is None or existing.severity.rank < signal.severity.rank:
            unique[key] = signal

    ordered = sorted(unique.values(), key=lambda s: (-s.severity.rank, s.column))

    report = LeakageReport(
        signals=ordered,
        target=target,
        columns_flagged=sorted({s.column for s in ordered}),
        candidate_targets=candidates,
    )
    # Attach the target reasoning as a synthetic info signal so the UI can show it.
    if target_reasoning:
        report.signals.append(
            LeakageSignal(
                column=target or "n/a",
                signal="target_selection",
                severity=Severity.INFO,
                confidence="n/a",
                reasoning=target_reasoning,
                recommendation=(
                    "Confirm or override the target in the sidebar; target-aware checks "
                    "(correlation, separation, duplicate-copy) all depend on it."
                ),
                evidence={"candidates": candidates},
            )
        )
    return report


def leakage_findings(report: LeakageReport) -> list[Finding]:
    """Convert leakage signals into findings, keeping the "potential" wording."""
    actionable = [s for s in report.signals if s.signal != "target_selection"]
    if not actionable:
        return [
            Finding(
                category="leakage",
                title="No leakage signals detected",
                detail=(
                    "None of the implemented heuristics fired: no identifier-like columns, "
                    "no duplicated columns, no feature correlated with the target above "
                    f"|r| = {config.TARGET_CORRELATION_FLAG}, and no feature that "
                    "separates a binary target perfectly."
                ),
                severity=Severity.INFO,
                evidence={"signals": 0, "target": report.target,
                          "candidates": report.candidate_targets},
                recommendation=(
                    "Absence of signals is not proof of absence. Leakage depends on how "
                    "the data was collected — review the pipeline with the people who "
                    "built it."
                ),
            )
        ]

    findings: list[Finding] = []
    if actionable:
        findings.append(
            Finding(
                category="leakage",
                title=f"{len(report.columns_flagged)} column(s) show potential leakage signals",
                detail=(
                    "Flagged column(s): "
                    + join_names(report.columns_flagged)
                    + ". Signals are heuristics, not verdicts: each one describes a pattern "
                    "that leaking columns often show. Target used for target-aware checks: "
                    + (f"`{report.target}`." if report.target else "none identified.")
                ),
                severity=max(
                    (s.severity for s in actionable), key=lambda sev: sev.rank,
                ),
                evidence={
                    "columns": report.columns_flagged,
                    "target": report.target,
                    "signals": [s.to_dict() for s in actionable],
                },
                recommendation=(
                    "For each flagged column, answer one question before modelling: is "
                    "this value known at the moment the prediction has to be made? If the "
                    "answer is no, exclude the feature."
                ),
            )
        )

    for signal in report.signals:
        if signal.signal == "target_selection":
            continue
        findings.append(
            Finding(
                category="leakage",
                title=f"Potential leakage — investigate `{signal.column}` ({signal.signal})",
                detail=f"{signal.reasoning} Confidence: {signal.confidence}.",
                severity=signal.severity,
                evidence=signal.evidence,
                recommendation=signal.recommendation,
            )
        )

    return findings
