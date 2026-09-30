"""Streamlit rendering functions for every dashboard section.

The analysis itself lives in the Streamlit-free modules; this file only turns
:class:`~src.schema.InvestigationReport` objects into widgets, charts and tables.
Every renderer is defensive: missing columns, empty results and unavailable
analyses produce an explanatory message instead of an exception.
"""

from __future__ import annotations

import html
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import charts, config, reporting, ui_theme
from src.cleaning import clean_dataframe, cleaned_filename, numeric_text_columns
from src.schema import InvestigationReport, Severity
from src.utils import format_number, format_percent, human_bytes


# --------------------------------------------------------------------------- #
# Generic widgets
# --------------------------------------------------------------------------- #
def button(label: str, key: str | None = None, primary: bool = False,
           help_text: str | None = None) -> bool:
    """Full-width button that works across Streamlit versions."""
    kwargs: dict[str, Any] = {"key": key, "help": help_text}
    if primary:
        kwargs["type"] = "primary"
    try:
        return bool(st.button(label, width="stretch", **kwargs))
    except TypeError:  # pragma: no cover - older Streamlit releases
        kwargs.pop("type", None)
        return bool(st.button(label, use_container_width=True, **kwargs))


def plot(figure: go.Figure, key: str | None = None) -> None:
    """Render a Plotly figure, staying compatible across Streamlit versions."""
    try:
        st.plotly_chart(figure, key=key, width="stretch", config={"displayModeBar": False})
    except TypeError:  # pragma: no cover - older Streamlit releases
        st.plotly_chart(figure, key=key, use_container_width=True,
                        config={"displayModeBar": False})


def dataframe(frame: pd.DataFrame, height: int | None = None, key: str | None = None) -> None:
    """Render a DataFrame, staying compatible across Streamlit versions."""
    if frame is None or frame.empty:
        st.info("Nothing to display for this table.")
        return
    kwargs: dict[str, Any] = {"hide_index": True, "key": key}
    if height:
        kwargs["height"] = height
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:  # pragma: no cover - older Streamlit releases
        st.dataframe(frame, use_container_width=True, **kwargs)


def metric_cards(items: Sequence[tuple[str, str, str]], per_row: int = 4,
                 accents: Sequence[str | None] | None = None) -> None:
    """Lay out metric cards in rows."""
    for start in range(0, len(items), per_row):
        row = items[start:start + per_row]
        columns = st.columns(len(row))
        for offset, (label, value, note) in enumerate(row):
            accent = accents[start + offset] if accents and start + offset < len(accents) else None
            with columns[offset]:
                st.markdown(ui_theme.card(label, value, note, accent), unsafe_allow_html=True)


def callout(text: str, severity: Severity | None = None) -> None:
    """Render an informative callout box."""
    st.markdown(ui_theme.note(text), unsafe_allow_html=True)


def show_findings(findings: Sequence[Any], limit: int = 10, empty_message: str = "No findings in this section.") -> None:
    """Render a list of findings as cards."""
    if not findings:
        st.info(empty_message)
        return
    st.markdown(ui_theme.findings_html(findings, limit=limit), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header and overview
# --------------------------------------------------------------------------- #
def render_overview(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Dataset structure, column groups and the raw data preview."""
    overview = report.overview
    types = overview.column_types

    metric_cards(
        [
            ("File", overview.filename, f"{overview.n_columns:,} columns"),
            ("Rows", f"{overview.n_rows:,}", f"{overview.total_cells:,} cells"),
            ("Memory", human_bytes(overview.memory_bytes), "in-memory footprint"),
            ("Missing", format_percent(overview.missing_rate, 2),
             f"{overview.missing_cells:,} empty cells"),
            ("Duplicates", f"{overview.duplicate_rows:,}",
             format_percent(report.duplicates.duplicate_rate, 2)),
            ("Numerical", f"{len(types.numeric):,}", "columns"),
            ("Categorical", f"{len(types.categorical):,}", "columns"),
            ("Datetime", f"{len(types.datetime):,}", "columns"),
        ],
        per_row=4,
    )

    if overview.load_warnings:
        with st.expander(f"Loader notes ({len(overview.load_warnings)})", expanded=False):
            for warning in overview.load_warnings:
                st.warning(warning)

    st.markdown(ui_theme.section("Column profile", "How each column was classified and what it contains"),
                unsafe_allow_html=True)
    profile = overview.dtype_table.copy()
    if not profile.empty:
        profile = profile.rename(columns={
            "column": "Column", "role": "Role", "dtype": "Stored type",
            "non_null": "Non-null", "missing": "Missing", "missing_pct": "Missing %",
            "unique": "Unique", "example": "Example value",
        })
        profile["Missing %"] = profile["Missing %"].round(2)
        dataframe(profile, height=min(560, 40 + 28 * len(profile)))

    with st.expander("Column groups", expanded=False):
        st.markdown(
            f"**Numerical ({len(types.numeric)})** — "
            f"{', '.join(f'`{c}`' for c in types.numeric) or 'none'}\n\n"
            f"**Categorical ({len(types.categorical)})** — "
            f"{', '.join(f'`{c}`' for c in types.categorical) or 'none'}\n\n"
            f"**Datetime ({len(types.datetime)})** — "
            f"{', '.join(f'`{c}`' for c in types.datetime) or 'none'}\n\n"
            f"**Boolean ({len(types.boolean)})** — "
            f"{', '.join(f'`{c}`' for c in types.boolean) or 'none'}",
            unsafe_allow_html=False,
        )
        if types.text_like:
            st.caption(
                "Text-like columns (high uniqueness, possibly free text or identifiers): "
                + ", ".join(types.text_like)
            )

    with st.expander("Raw data preview (first 100 rows)", expanded=False):
        dataframe(frame.head(100))


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def render_health(report: InvestigationReport) -> None:
    """Health score, its components and the full calculation trail."""
    health = report.health
    left, right = st.columns([1, 2])
    with left:
        plot(charts.health_gauge(health), key="health_gauge")
    with right:
        plot(charts.health_breakdown(health), key="health_breakdown")

    st.markdown(ui_theme.health_summary_html(health), unsafe_allow_html=True)

    st.markdown(ui_theme.section("Audit trail", "Every dimension, its measurement and its weight"),
                unsafe_allow_html=True)
    component_rows = pd.DataFrame(
        [
            {
                "Dimension": c.label,
                "What was measured": c.measurement,
                "Sub-score": round(c.score, 1),
                "Weight": f"{c.weight:.0%}",
                "Contribution": round(c.weighted_score, 2),
                "Deduction explanation": c.penalty_reason or "—",
            }
            for c in health.components
        ]
    )
    dataframe(component_rows, height=250)

    with st.expander("How the score is calculated", expanded=False):
        st.markdown(
            "```\n"
            "score = Σ (sub_score_i × weight_i)\n\n"
            "completeness  30%  ← share of missing cells, minus 5 points per column >50% missing\n"
            "integrity     20%  ← share of numeric cells outside the 1.5×IQR fences\n"
            "consistency   20%  ← share of columns whose stored type hides their content\n"
            "uniqueness    15%  ← share of exact duplicate rows\n"
            "variability   15%  ← share of constant / near-constant columns\n"
            "```\n"
            "Sub-scores are linear and saturate at fixed thresholds (missing cells 25%, "
            "outlier cells 5%, duplicate rows 10%), so cleaning the data can never lower "
            "the score. The formula is deterministic: the same file always produces the "
            "same number."
        )
        for item in health.notes:
            st.markdown(f"- {item}")

    st.markdown(ui_theme.section("Findings by severity", "How many issues were raised in total"),
                unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        plot(charts.findings_by_severity(report.findings), key="sev_donut")
    with right:
        counts = pd.Series(
            [f.severity.label for f in report.findings], name="Severity"
        ).value_counts().rename_axis("Severity").reset_index(name="Findings")
        dataframe(counts)


# --------------------------------------------------------------------------- #
# Missing data and duplicates
# --------------------------------------------------------------------------- #
def render_missing(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Missing-value analysis with a map of where the gaps are."""
    missing = report.missing
    metric_cards(
        [
            ("Missing cells", f"{missing.missing_cells:,}",
             f"of {missing.total_cells:,}"),
            ("Missing rate", format_percent(missing.missing_rate, 2), "of all cells"),
            ("Columns affected", f"{missing.columns_with_missing:,}",
             f"of {report.overview.n_columns:,}"),
            ("Complete rows", format_percent(missing.complete_row_rate, 1),
             f"{missing.complete_rows:,} rows"),
        ],
        per_row=4,
    )

    plot(charts.missing_values_bar(missing), key="missing_bar")

    with st.expander("Where are the gaps?", expanded=False):
        st.caption(
            "Red cells are missing values. Rows are capped for rendering; the row index "
            "follows the file order."
        )
        plot(charts.missingness_matrix(frame), key="missing_map")

    st.markdown(ui_theme.section("Per-column detail"), unsafe_allow_html=True)
    if missing.affected_columns:
        table = pd.DataFrame(
            [
                {
                    "Column": c.column,
                    "Missing": c.missing,
                    "Missing %": round(c.missing_rate * 100, 2),
                    "Dtype": c.dtype,
                }
                for c in missing.affected_columns
            ]
        )
        dataframe(table)
    else:
        st.success("No column contains missing values.")

    show_findings(report.findings_by_category("missing_data"))


def render_duplicates(report: InvestigationReport) -> None:
    """Exact duplicate rows with examples."""
    duplicates = report.duplicates
    metric_cards(
        [
            ("Duplicate rows", f"{duplicates.duplicate_rows:,}", "additional copies"),
            ("Duplicate rate", format_percent(duplicates.duplicate_rate, 2), "of all rows"),
            ("Duplicate groups", f"{duplicates.duplicate_group_count:,}",
             "distinct repeated records"),
            ("Distinct rows", f"{duplicates.unique_rows:,}",
             f"of {duplicates.total_rows:,}"),
        ],
        per_row=4,
    )

    if not duplicates.examples.empty:
        st.markdown(ui_theme.section("Example duplicate records"),
                    unsafe_allow_html=True)
        st.caption("Every member of a duplicated group is shown, in file order.")
        dataframe(duplicates.examples.head(config.MAX_EXAMPLE_ROWS))

    show_findings(report.findings_by_category("duplicates"))


# --------------------------------------------------------------------------- #
# Constant columns and dtypes
# --------------------------------------------------------------------------- #
def render_variance_and_types(report: InvestigationReport) -> None:
    """Constant / near-constant columns plus stored-type problems."""
    st.markdown(ui_theme.section(
        "Constant and low-variance features",
        "Columns that barely change cannot separate observations",
    ), unsafe_allow_html=True)

    if report.variance:
        table = pd.DataFrame(
            [
                {
                    "Column": v.column,
                    "Distinct values": v.n_unique,
                    "Dominant value": str(v.dominant_value)[:40],
                    "Dominance": format_percent(v.dominant_rate, 2),
                    "Kind": "constant" if v.is_constant else "near-constant",
                    "Note": v.note,
                }
                for v in report.variance
            ]
        )
        dataframe(table)
    else:
        st.success("Every column shows meaningful variation.")

    show_findings(report.findings_by_category("low_variance"), limit=5)

    st.markdown(ui_theme.section(
        "Data-type issues",
        "Values stored in a type that hides their content",
    ), unsafe_allow_html=True)

    if report.dtype_issues:
        table = pd.DataFrame(
            [
                {
                    "Column": i.column,
                    "Issue": i.issue,
                    "Stored type": i.stored_dtype,
                    "Suggested type": i.suggested_dtype,
                    "Affected": format_percent(i.affected_rate),
                    "Detail": i.detail,
                    "Example values": ", ".join(str(v) for v in i.example_values[:3]),
                }
                for i in report.dtype_issues
            ]
        )
        dataframe(table)
    else:
        st.success("Stored data types match column content.")

    show_findings(report.findings_by_category("dtypes"), limit=4)


# --------------------------------------------------------------------------- #
# Numerical statistics
# --------------------------------------------------------------------------- #
def render_numerical(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Descriptive statistics and per-column distributions."""
    if not report.numeric_stats:
        st.info(
            "No numerical columns were detected. If this is unexpected, check the "
            "Data-type issues tab: numbers stored as text are excluded from numerical "
            "analysis by design."
        )
        return

    table = pd.DataFrame(
        [
            {
                "Column": s.column,
                "N": s.count,
                "Missing": s.missing,
                "Mean": format_number(s.mean),
                "Median": format_number(s.median),
                "Std": format_number(s.std),
                "Min": format_number(s.minimum),
                "Q1": format_number(s.q1),
                "Q3": format_number(s.q3),
                "Max": format_number(s.maximum),
                "IQR": format_number(s.iqr),
                "Skew": format_number(s.skewness),
                "Kurtosis": format_number(s.kurtosis),
                "Zeros": format_percent(s.zero_share),
            }
            for s in report.numeric_stats
        ]
    )
    dataframe(table, height=min(520, 40 + 30 * len(table)))

    st.markdown(ui_theme.section("Distributions", "Histogram per numeric column"),
                unsafe_allow_html=True)
    selected = st.selectbox(
        "Column", [s.column for s in report.numeric_stats], index=0, key="numeric_column_choice"
    )
    outlier_indices: list[int] = []
    for result in report.outliers.results:
        if result.column == selected and result.method == "iqr":
            outlier_indices = result.indices
            break

    left, right = st.columns([3, 2])
    with left:
        plot(charts.numeric_histogram(frame, selected, outlier_indices), key="hist")
    with right:
        plot(charts.numeric_box(frame, selected), key="box")

    show_findings(report.findings_by_category("numerical"), limit=6)


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def render_outliers(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Outlier results for all three rules, with examples."""
    st.markdown(ui_theme.note(
        "<strong>An outlier is not automatically an error.</strong> The IQR rule is "
        "distribution-free, the z-score assumes approximate normality, and the modified "
        "z-score (MAD based) resists skew and masking — that is why all three are "
        "reported side by side. Disagreement between methods is informative, not a bug."
    ), unsafe_allow_html=True)
    st.write("")

    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for result in report.outliers.results:
        entry = grouped.setdefault(result.column, {"Column": result.column})
        entry[result.method] = (
            f"{result.count:,} ({format_percent(result.rate, 2)})" if result.available
            else "not applicable"
        )
        entry.setdefault("_reasons", {})[result.method] = result.reason
    method_columns = [
        ("iqr", "IQR (1.5)"), ("iqr_extreme", "IQR (3.0)"),
        ("zscore", "Z-score (3.0)"), ("modified_zscore", "Modified z (3.5)"),
    ]
    for entry in grouped.values():
        row = {"Column": entry["Column"]}
        for key, label in method_columns:
            row[label] = entry.get(key, "—")
        rows.append(row)

    if rows:
        dataframe(pd.DataFrame(rows))
    else:
        st.info("No numerical columns were available for outlier testing.")

    plot(charts.outlier_bars(report.outliers.results, method="iqr"), key="outlier_bars")

    eligible = [r for r in report.outliers.results if r.method == "iqr" and r.available and r.count]
    if eligible:
        st.markdown(ui_theme.section("Inspect flagged records"), unsafe_allow_html=True)
        choice = st.selectbox(
            "Column with outliers", [r.column for r in eligible], key="outlier_column_choice"
        )
        result = next(r for r in eligible if r.column == choice)
        examples = _outlier_examples(frame, result)
        st.caption(
            f"{result.count:,} of {len(frame):,} values fall outside "
            f"[{format_number(result.lower_bound)}, {format_number(result.upper_bound)}]."
        )
        dataframe(examples)

    show_findings(report.findings_by_category("outliers"), limit=6)


def _outlier_examples(frame: pd.DataFrame, result: Any) -> pd.DataFrame:
    """Rows flagged for one column, most extreme first."""
    from src.outliers import outlier_examples

    return outlier_examples(frame, result.column, result, limit=config.MAX_EXAMPLE_ROWS)


# --------------------------------------------------------------------------- #
# Correlations
# --------------------------------------------------------------------------- #
def render_correlations(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Correlation matrix, strongest pairs and categorical associations."""
    correlation = report.correlations
    if correlation.unavailable_reason:
        st.info(correlation.unavailable_reason)
    else:
        plot(charts.correlation_heatmap(correlation.matrix, correlation.method), key="corr_heat")
        callout(
            "<strong>Correlation is not causation.</strong> A high coefficient can come "
            "from a shared driver, from selection in the data, or from how a column was "
            "computed. Inspect a scatter plot before interpreting any relationship "
            "causally. Self-correlations are always excluded."
        )
        plot(charts.correlation_bar(correlation.pairs), key="corr_bar")

        st.markdown(ui_theme.section("Strongest relationships"),
                    unsafe_allow_html=True)
        pairs_table = pd.DataFrame(
            [
                {
                    "Column A": p.column_a,
                    "Column B": p.column_b,
                    "r": round(p.r, 4),
                    "Strength": p.strength,
                    "Direction": p.direction,
                    "Pairs (n)": p.n_observations,
                    "p-value": "—" if p.p_value is None else f"{p.p_value:.3g}",
                }
                for p in correlation.pairs[:25]
            ]
        )
        dataframe(pairs_table)

        with st.expander("Explore a scatter plot", expanded=False):
            columns = correlation.columns_analysed
            if len(columns) >= 2:
                left, right = st.columns(2)
                with left:
                    x_axis = st.selectbox("X axis", columns, index=0, key="scatter_x")
                with right:
                    y_axis = st.selectbox("Y axis", columns, index=min(1, len(columns) - 1),
                                          key="scatter_y")
                plot(charts.explorer_scatter(frame, x_axis, y_axis), key="corr_scatter")

    show_findings(report.findings_by_category("correlations"), limit=6)

    st.markdown(ui_theme.section(
        "Categorical associations (Cramér's V)",
        "Which categorical columns carry overlapping information?",
    ), unsafe_allow_html=True)
    associations = _associations_for(frame, report)
    if associations.empty:
        st.caption(
            "No pair of categorical columns with a manageable number of levels was "
            "available for association testing."
        )
    else:
        dataframe(associations.head(15))
    show_findings(report.findings_by_category("associations"), limit=3)


def _associations_for(frame: pd.DataFrame, report: InvestigationReport) -> pd.DataFrame:
    """Recompute the association table for display (cheap and always consistent)."""
    from src.correlations import categorical_associations

    return categorical_associations(
        frame,
        report.overview.column_types.categorical + report.overview.column_types.boolean,
    )


# --------------------------------------------------------------------------- #
# Categorical
# --------------------------------------------------------------------------- #
def render_categorical(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Cardinality, dominance and frequency distributions."""
    if not report.categorical_stats:
        st.info("No categorical columns were detected in this dataset.")
        return

    table = pd.DataFrame(
        [
            {
                "Column": s.column,
                "N": s.count,
                "Missing": s.missing,
                "Unique": s.n_unique,
                "Top value": str(s.top_value)[:40],
                "Top count": s.top_count,
                "Top %": round(s.top_rate * 100, 2),
                "2nd %": round(s.second_rate * 100, 2),
                "Rare levels": s.rare_categories,
                "Entropy (bits)": None if s.entropy is None else round(s.entropy, 2),
                "Flags": ", ".join(filter(None, [
                    "dominant" if s.is_dominant and s.n_unique > 1 else "",
                    "high-cardinality" if s.is_high_cardinality else "",
                ])) or "—",
            }
            for s in report.categorical_stats
        ]
    )
    dataframe(table, height=min(520, 40 + 30 * len(table)))

    st.markdown(ui_theme.section("Frequency distribution"), unsafe_allow_html=True)
    choice = st.selectbox(
        "Column", [s.column for s in report.categorical_stats], key="categorical_column_choice"
    )
    stats = next(s for s in report.categorical_stats if s.column == choice)
    frequency = frequency_for(frame, choice)

    left, right = st.columns([3, 2])
    with left:
        plot(charts.category_bar(frequency, choice), key="cat_bar")
    with right:
        plot(charts.category_pie(frequency, choice), key="cat_pie")

    st.caption(
        f"{stats.n_unique:,} distinct values, "
        f"most frequent is '{str(stats.top_value)[:40]}' "
        f"({format_percent(stats.top_rate, 2)} of {stats.count:,} non-missing rows), "
        f"entropy {stats.entropy:.2f} bits."
        if stats.entropy is not None else ""
    )
    dataframe(frequency.head(30))

    show_findings(report.findings_by_category("categorical"), limit=6)


def frequency_for(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Frequency table for one column (thin wrapper kept for readability above)."""
    from src.categorical import frequency_table

    return frequency_table(frame, column, top_n=15)


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
def render_anomalies(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Isolation Forest results with score distribution and examples."""
    anomalies = report.anomalies
    if not anomalies.available:
        st.info(f"Anomaly detection is not available for this dataset. {anomalies.reason}")
        show_findings(report.findings_by_category("anomalies"))
        return

    metric_cards(
        [
            ("Rows scored", f"{anomalies.n_rows_scored:,}", "median-imputed, standardised"),
            ("Anomalies", f"{anomalies.n_anomalies:,}",
             format_percent(anomalies.anomaly_rate, 2)),
            ("Features used", f"{len(anomalies.features_used):,}",
             "numeric, ≥ 4 distinct values"),
            ("Contamination", f"{anomalies.contamination:.3f}", "configured in the sidebar"),
        ],
        per_row=4,
    )

    st.markdown(ui_theme.note(
        "<strong>Anomaly detection identifies unusual observations, not necessarily "
        "fraudulent or incorrect ones.</strong> Isolation Forest isolates records that "
        "sit far from the bulk of the data in the standardised feature space. A rare but "
        "perfectly valid record is flagged as readily as a corrupted one."
    ), unsafe_allow_html=True)
    st.write("")
    if anomalies.reason:
        st.caption(anomalies.reason)

    left, right = st.columns([1, 1])
    with left:
        plot(charts.anomaly_score_distribution(anomalies), key="anom_scores")
    with right:
        if len(anomalies.features_used) >= 2:
            columns = anomalies.features_used
            x_axis = st.selectbox("X axis", columns, index=0, key="anom_x")
            y_axis = st.selectbox("Y axis", columns, index=1, key="anom_y")
            plot(charts.anomaly_scatter(anomalies, x_axis, y_axis), key="anom_scatter")
        else:
            st.info("At least two features are required for the anomaly scatter plot.")

    if not anomalies.top_deviating_features.empty:
        st.markdown(ui_theme.section(
            "Where the anomalies deviate",
            "Mean robust |z| (median/MAD based) across the flagged records",
        ), unsafe_allow_html=True)
        dataframe(anomalies.top_deviating_features.head(12))

    flagged = anomalies.records[anomalies.records["is_anomaly"]] if anomalies.available else pd.DataFrame()
    if not flagged.empty:
        st.markdown(ui_theme.section("Most anomalous records"),
                    unsafe_allow_html=True)
        display = flagged.drop(columns=["is_anomaly", "anomaly_decision"], errors="ignore").head(
            config.MAX_EXAMPLE_ROWS
        )
        st.caption("Lower anomaly score = more unusual. The percentile ranks the record "
                   "against the rest of the dataset.")
        dataframe(display)

    show_findings(report.findings_by_category("anomalies"), limit=4)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def render_leakage(report: InvestigationReport) -> None:
    """Potential-leakage signals with reasoning and recommended checks."""
    leakage = report.leakage
    st.markdown(ui_theme.note(
        "<strong>These are signals, not verdicts.</strong> Leakage depends on how the "
        "data was collected and when each value becomes available — something a CSV "
        "cannot tell us. Each row below names the pattern that was detected, the "
        "reasoning behind it and the question a human should answer."
    ), unsafe_allow_html=True)
    st.write("")

    metric_cards(
        [
            ("Target used", str(leakage.target) if leakage.target else "none identified",
             "drives the target-aware checks"),
            ("Columns flagged", f"{leakage.n_columns_flagged:,}",
             "each needs a human decision"),
            ("Signals raised", f"{len([s for s in leakage.signals if s.signal != 'target_selection']):,}",
             "heuristic patterns matched"),
        ],
        per_row=3,
    )

    target_signal = next((s for s in leakage.signals if s.signal == "target_selection"), None)
    if target_signal is not None:
        with st.expander("How the target was chosen (override it in the sidebar)",
                         expanded=False):
            st.write(target_signal.reasoning)

    actionable = [s for s in leakage.signals if s.signal != "target_selection"]
    if actionable:
        table = pd.DataFrame(
            [
                {
                    "Column": s.column,
                    "Signal": s.signal,
                    "Severity": s.severity.label,
                    "Confidence": s.confidence,
                    "Reasoning": s.reasoning,
                    "Action": s.recommendation,
                }
                for s in actionable
            ]
        )
        dataframe(table, height=min(600, 60 + 34 * len(table)))
    else:
        st.success(
            "No leakage signals fired. This is not proof of absence: review the data "
            "collection process with the people who built the pipeline."
        )

    show_findings(report.findings_by_category("leakage"), limit=8)


# --------------------------------------------------------------------------- #
# Report and downloads
# --------------------------------------------------------------------------- #
def render_clean_data(frame: pd.DataFrame, filename: str) -> None:
    """Preview selected cleaning operations and download a separate CSV copy."""
    st.markdown(
        ui_theme.section(
            "Create a cleaned copy",
            "Choose the changes, review their impact, then download the result",
        ),
        unsafe_allow_html=True,
    )
    st.info(
        "These edits apply only to the preview and downloaded copy. The uploaded "
        "dataset and its investigation report stay unchanged."
    )

    clean_key = st.session_state.get("data_hash") or "dataset"
    left, right = st.columns(2)
    with left:
        normalize_missing = st.checkbox(
            "Treat common text markers as missing",
            value=False,
            key=f"clean_normalize_missing_{clean_key}",
            help=(
                "Converts blank text, NA, N/A, NULL, None, and NaN to missing values. "
                "Leave this off if any of these are real categories in your data."
            ),
        )
        remove_duplicates = st.checkbox(
            "Remove exact duplicate rows",
            value=False,
            key=f"clean_remove_duplicates_{clean_key}",
            help="Keeps the first copy of each repeated row.",
        )

    candidates = numeric_text_columns(frame)
    with right:
        numeric_columns = st.multiselect(
            "Convert numeric-looking text columns",
            options=candidates,
            default=[],
            key=f"clean_numeric_columns_{clean_key}",
            help=(
                "Only columns where at least 90% of sampled values look numeric are "
                "suggested. Currency and separator symbols are removed; unparseable "
                "values become missing. A trailing percent sign is removed without "
                "rescaling the value. Review values with leading zeros carefully; "
                "identifier-like column names are not suggested."
            ),
        )
        if not candidates:
            st.caption("No numeric-looking text columns were detected.")

    missing_choice = st.selectbox(
        "How should remaining missing values be handled?",
        [
            "Leave them missing",
            "Drop rows containing any missing value",
            "Fill numeric columns with the median and other columns with the most common value",
        ],
        index=0,
        key=f"clean_missing_strategy_{clean_key}",
        help=(
            "Dropping or filling values changes your data. Select an option only when "
            "it makes sense for your analysis."
        ),
    )
    missing_strategy = {
        "Leave them missing": "keep",
        "Drop rows containing any missing value": "drop_rows",
        "Fill numeric columns with the median and other columns with the most common value": "fill",
    }[missing_choice]

    result = clean_dataframe(
        frame,
        normalize_missing_markers=normalize_missing,
        numeric_columns=numeric_columns,
        remove_duplicate_rows=remove_duplicates,
        missing_strategy=missing_strategy,
    )
    remaining_missing = int(result.frame.isna().sum().sum())
    metric_cards(
        [
            ("Rows in cleaned copy", f"{len(result.frame):,}",
             f"{result.rows_removed:,} removed"),
            ("Values changed", f"{result.cells_changed:,}", "converted, normalized, or filled"),
            ("Missing cells left", f"{remaining_missing:,}", "in the cleaned copy"),
        ],
        per_row=3,
    )

    st.markdown(ui_theme.section("Cleaning preview"), unsafe_allow_html=True)
    if result.changes:
        dataframe(
            pd.DataFrame(
                [
                    {
                        "Operation": change.operation,
                        "Rows removed": change.rows_removed,
                        "Values changed": change.cells_changed,
                        "Details": change.detail,
                    }
                    for change in result.changes
                ]
            )
        )
        if result.frame.empty:
            st.warning("These choices remove every row. Review the options before downloading.")
        else:
            st.caption("First 20 rows of the cleaned copy:")
            dataframe(result.frame.head(20), key="cleaned_preview")
        st.download_button(
            "Download cleaned CSV",
            data=result.frame.to_csv(index=False).encode("utf-8-sig"),
            file_name=cleaned_filename(filename),
            mime="text/csv",
            key="download_cleaned_csv",
        )
    else:
        st.info(
            "No changes are selected or needed yet. Choose an operation above to preview it."
        )

    st.caption(
        "Potential outliers are not removed automatically: unusual values can be valid, "
        "so inspect them before deciding whether to change or exclude them."
    )


def render_report(report: InvestigationReport, frame: pd.DataFrame) -> None:
    """Automated investigation summary plus downloadable reports."""
    overview = report.overview

    st.markdown(
        f"""
        <div class="da-hero">
          <h1 style="font-size:24px">DATA AUTOPSY REPORT</h1>
          <p><strong>{html.escape(overview.filename)}</strong> &middot;
             {overview.n_rows:,} rows &times; {overview.n_columns:,} columns &middot;
             health {report.health.score:.1f}/100 ({html.escape(report.health.grade)}) &middot;
             analysed in {report.analysis_seconds:.2f}s</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(ui_theme.section("Major findings"), unsafe_allow_html=True)
    summary = reporting.build_executive_summary(report)
    for line in summary:
        severity_name = line[1:line.index("]")].lower() if line.startswith("[") else "info"
        text = line[line.index("]") + 1:].strip() if line.startswith("[") else line
        st.markdown(
            f"<div class='da-reco'>{ui_theme.badge(severity_name)}{html.escape(text)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(ui_theme.section("Recommendations"), unsafe_allow_html=True)
    for index, recommendation in enumerate(report.recommendations, start=1):
        st.markdown(ui_theme.recommendation(recommendation, index), unsafe_allow_html=True)

    st.markdown(ui_theme.section("Download the report"), unsafe_allow_html=True)
    markdown_report = reporting.build_markdown_report(report)
    html_report = reporting.build_html_report(report, figures=_report_figures(report, frame))

    left, middle = st.columns(2)
    with left:
        st.download_button(
            "Download Markdown report",
            data=markdown_report,
            file_name=reporting.suggested_filename(report, "md"),
            mime="text/markdown",
            width="stretch",
        )
    with middle:
        st.download_button(
            "Download HTML report (interactive charts)",
            data=html_report,
            file_name=reporting.suggested_filename(report, "html"),
            mime="text/html",
            width="stretch",
        )
    st.caption(
        "The HTML report embeds the same charts as this dashboard and loads Plotly from "
        "a CDN, so it needs an internet connection to render the interactive figures. "
        "Every number also appears as plain text, so the report stays readable offline."
    )

    with st.expander("Preview the Markdown report", expanded=False):
        st.code(markdown_report[:6000] + ("\n\n... (truncated in the preview)" if len(markdown_report) > 6000 else ""),
                language="markdown")

    st.markdown(ui_theme.section("All findings"), unsafe_allow_html=True)
    show_findings(report.findings, limit=40, empty_message="No findings were raised.")


def _report_figures(report: InvestigationReport, frame: pd.DataFrame) -> dict[str, str]:
    """Figures embedded in the exported HTML report."""
    figures: dict[str, str] = {}
    try:
        figures["Findings by severity"] = charts.figure_to_html(
            charts.findings_by_severity(report.findings)
        )
        figures["Health score components"] = charts.figure_to_html(
            charts.health_breakdown(report.health)
        )
        if report.missing.affected_columns:
            figures["Missing values by column"] = charts.figure_to_html(
                charts.missing_values_bar(report.missing)
            )
        if report.outliers.columns_flagged:
            figures["Outlier rate per column"] = charts.figure_to_html(
                charts.outlier_bars(report.outliers.results, method="iqr")
            )
        if not report.correlations.matrix.empty:
            figures["Correlation heatmap"] = charts.figure_to_html(
                charts.correlation_heatmap(report.correlations.matrix, report.correlations.method)
            )
        if report.anomalies.available and not report.anomalies.records.empty:
            figures["Anomaly scores"] = charts.figure_to_html(
                charts.anomaly_score_distribution(report.anomalies)
            )
    except Exception:  # noqa: BLE001 - a failing chart must not block the download
        pass
    return figures


def render_footer() -> None:
    """Dashboard footer."""
    from src import __version__

    st.markdown(ui_theme.footer_html(__version__), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Landing page (no dataset loaded)
# --------------------------------------------------------------------------- #
def render_landing() -> bool:
    """Empty state shown before any dataset is loaded.

    Returns ``True`` when the "load demo dataset" button was pressed.
    """
    st.markdown(
        ui_theme.hero(
            "DATA AUTOPSY",
            "A statistical and data-quality investigation tool for CSV datasets.",
        ),
        unsafe_allow_html=True,
    )
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "Upload a CSV file and DATA AUTOPSY runs a full, deterministic investigation:\n\n"
            "1. **Structure** — rows, columns, memory, column roles, dtype drift\n"
            "2. **Health score** — weighted, fully auditable, no black box\n"
            "3. **Missing data** — counts, rates, gaps map, imputation warnings\n"
            "4. **Duplicates** — exact duplicate rows with example records\n"
            "5. **Numerical statistics** — mean, median, quartiles, skewness, kurtosis\n"
            "6. **Outliers** — IQR, z-score and robust MAD rules\n"
            "7. **Correlations** — matrix, strongest positive/negative pairs, Cramér's V\n"
            "8. **Categorical analysis** — cardinality, dominance, rare levels, entropy\n"
            "9. **Low-variance features** — constant and near-constant columns\n"
            "10. **Anomalies** — Isolation Forest with a configurable contamination\n"
            "11. **Potential data leakage** — heuristic signals with reasoning\n"
            "12. **Automated report** — Markdown and HTML downloads\n"
        )
        st.info(
            "Start in the sidebar: upload a CSV, adjust the investigation settings if you "
            "want to, then press **Run investigation**."
        )
    with right:
        st.markdown(ui_theme.section("Try it without data"), unsafe_allow_html=True)
        st.write(
            "The repository ships a synthetic student dataset that deliberately contains "
            "missing values, duplicates, outliers, a constant column, numbers stored as "
            "text and a leaking target."
        )
        clicked = button("Load the synthetic demo dataset", key="landing_demo")
        st.caption(
            "synthetic — generated by sample_data/make_sample_data.py with a fixed seed; "
            "it describes fictional students only."
        )
        st.markdown(ui_theme.section("Supported input"), unsafe_allow_html=True)
        st.write(
            "Delimited text files: `.csv`, `.tsv`, `.txt` (comma, semicolon, tab or pipe "
            "separated). Excel, Parquet and JSON are not read directly — export them to "
            "CSV first."
        )
    return bool(clicked)

