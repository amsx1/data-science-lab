"""Report generation.

Two export formats are produced from the *same* :class:`InvestigationReport`
object, so the downloadable files can never drift from what the dashboard shows:

* **Markdown** — portable, diff-friendly, ideal for a repository or a notebook.
* **HTML** — self-contained, dark-themed document with the Plotly charts embedded.

Nothing is invented at report time: every sentence is derived from a measured
value, and the methodology appendix states how each number was computed.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import pandas as pd

from src import config
from src.schema import Finding, HealthScore, InvestigationReport, Severity
from src.utils import (
    ellipsize,
    format_number,
    format_percent,
    human_bytes,
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _escape_md(value: Any) -> str:
    """Escape characters that would break a Markdown table cell."""
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _as_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Convert a DataFrame to plain records, tolerating ``None`` and empty frames."""
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return frame.to_dict(orient="records")


def md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a Markdown table, or a dash when there are no rows."""
    rows = list(rows)
    if not rows:
        return "_No rows to display._\n"
    lines = [
        "| " + " | ".join(_escape_md(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def html_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    max_rows: int = 200,
) -> str:
    """Render an HTML table with the report's styling."""
    rows = list(rows)
    if not rows:
        return '<p class="muted">No rows to display.</p>'
    truncated = len(rows) > max_rows
    shown = rows[:max_rows]
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape('' if c is None else str(c))}</td>" for c in row) + "</tr>"
        for row in shown
    )
    note = (
        f'<p class="muted">Showing the first {max_rows} of {len(rows):,} rows.</p>'
        if truncated else ""
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{note}'


def _severity_badge(severity: str) -> str:
    """Coloured pill used in the HTML report."""
    color = config.SEVERITY_COLORS.get(severity, "#8B949E")
    return (
        f'<span class="badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{html.escape(severity.upper())}</span>'
    )


# --------------------------------------------------------------------------- #
# Executive summary
# --------------------------------------------------------------------------- #
def build_executive_summary(report: InvestigationReport, limit: int = 12) -> list[str]:
    """The "major findings" bullet list, drawn from the highest-severity findings."""
    lines: list[str] = []
    for finding in report.findings:
        if finding.severity == Severity.INFO:
            continue
        if finding.category in {"leakage"} and "target_selection" in finding.title:
            continue
        lines.append(f"[{finding.severity.label}] {finding.title}")
        if len(lines) >= limit:
            break
    if not lines:
        lines.append("No quality, distribution or leakage problems exceeded the "
                     "reporting thresholds on this dataset.")
    return lines


def health_verdict(health: HealthScore) -> str:
    """One-paragraph plain-language interpretation of the health score."""
    weakest = min(health.components, key=lambda c: c.score) if health.components else None
    strongest = max(health.components, key=lambda c: c.score) if health.components else None
    text = (
        f"The dataset scores {health.score:.1f}/100 ({health.grade}). "
    )
    if strongest and weakest and strongest.key != weakest.key:
        text += (
            f"The strongest dimension is {strongest.label.lower()} ({strongest.score:.1f}/100) "
            f"and the weakest is {weakest.label.lower()} ({weakest.score:.1f}/100), where the "
            "largest deduction was applied."
        )
    return text


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def _md_overview(report: InvestigationReport) -> str:
    """Dataset overview section."""
    overview = report.overview
    types = overview.column_types
    parts = [
        "## 2. Dataset overview\n",
        md_table(
            ["Property", "Value"],
            [
                ["File name", overview.filename],
                ["Rows", f"{overview.n_rows:,}"],
                ["Columns", f"{overview.n_columns:,}"],
                ["Cells", f"{overview.total_cells:,}"],
                ["Memory usage", human_bytes(overview.memory_bytes)],
                ["Missing cells", f"{overview.missing_cells:,} ({format_percent(overview.missing_rate)})"],
                ["Duplicate rows", f"{overview.duplicate_rows:,}"],
                ["Numeric columns", f"{len(types.numeric)}: " + (", ".join(types.numeric) or "-")],
                ["Categorical columns", f"{len(types.categorical)}: " + (", ".join(types.categorical) or "-")],
                ["Datetime columns", f"{len(types.datetime)}: " + (", ".join(types.datetime) or "-")],
                ["Boolean columns", f"{len(types.boolean)}: " + (", ".join(types.boolean) or "-")],
            ],
        ),
    ]
    if overview.load_warnings:
        parts.append("**Loader notes**\n")
        parts += [f"- {warning}" for warning in overview.load_warnings]
        parts.append("")

    parts.append("### Column profile\n")
    parts.append(
        md_table(
            ["Column", "Role", "Stored type", "Non-null", "Missing %", "Unique", "Example"],
            [
                [
                    row.get("column"), row.get("role", "?"), row.get("dtype"),
                    f"{int(row.get('non_null', 0)):,}",
                    f"{float(row.get('missing_pct', 0.0)):.2f}%",
                    f"{int(row.get('unique', 0)):,}",
                    ellipsize(str(row.get("example", "")), 28),
                ]
                for row in _as_records(overview.dtype_table)
            ],
        )
    )
    return "\n".join(parts)


def _md_health(report: InvestigationReport) -> str:
    """Health score section with its audit trail."""
    health = report.health
    lines = [
        "## 3. Dataset health score\n",
        f"**{health.score:.1f}/100 — {health.grade}**\n",
        health_verdict(health) + "\n",
        md_table(
            ["Dimension", "Measured", "Sub-score", "Weight", "Contribution", "Deduction"],
            [
                [
                    c.label, c.measurement, f"{c.score:.1f}", f"{c.weight:.0%}",
                    f"{c.weighted_score:.2f}", c.penalty_reason or "-",
                ]
                for c in health.components
            ],
        ),
        "**How the score is calculated**\n",
    ]
    for note in health.notes:
        lines.append(f"- {note}")
    lines.append(
        "- Formula: `score = sum(sub_score_i * weight_i)` with linear sub-scores that "
        "reach 0 at fixed saturation thresholds (missing cells 25%, outlier cells 5%, "
        "duplicate rows 10%, columns with type problems and low-variance columns scored "
        "as proportions)."
    )
    return "\n".join(lines) + "\n"


def _md_missing(report: InvestigationReport) -> str:
    """Missing data section."""
    missing = report.missing
    lines = ["## 4. Missing data analysis\n"]
    lines.append(
        f"- Total cells: {missing.total_cells:,}\n"
        f"- Missing cells: {missing.missing_cells:,} ({format_percent(missing.missing_rate)})\n"
        f"- Columns affected: {missing.columns_with_missing:,}\n"
        f"- Rows with at least one missing value: {missing.rows_with_missing:,} "
        f"({format_percent(1 - missing.complete_row_rate)})\n"
        f"- Fully complete rows: {missing.complete_rows:,} "
        f"({format_percent(missing.complete_row_rate)})\n"
    )
    lines.append(
        md_table(
            ["Column", "Missing", "Missing %", "Severity", "Dtype"],
            [
                [
                    c.column, f"{c.missing:,}", f"{c.missing_rate * 100:.2f}%",
                    _missing_band(c.missing_rate), c.dtype,
                ]
                for c in missing.affected_columns
            ],
        )
    )
    return "\n".join(lines)


def _missing_band(rate: float) -> str:
    """Severity band label for a missing rate."""
    if rate >= config.MISSING_CRITICAL:
        return "critical"
    if rate >= config.MISSING_HIGH:
        return "high"
    if rate >= config.MISSING_MODERATE:
        return "moderate"
    if rate >= config.MISSING_LOW:
        return "low"
    return "info"


def _md_duplicates(report: InvestigationReport) -> str:
    """Duplicate analysis section."""
    duplicates = report.duplicates
    lines = [
        "## 5. Duplicate analysis\n",
        f"- Exact duplicate rows: {duplicates.duplicate_rows:,} "
        f"({format_percent(duplicates.duplicate_rate)})\n"
        f"- Distinct rows: {duplicates.unique_rows:,} of {duplicates.total_rows:,}\n"
        f"- Duplicate groups: {duplicates.duplicate_group_count:,}\n",
    ]
    examples = _as_records(duplicates.examples)
    if examples:
        columns = list(examples[0].keys())[:12]
        lines.append("**Example duplicate records**\n")
        lines.append(
            md_table(columns, [[row.get(c) for c in columns] for row in examples])
        )
    return "\n".join(lines)


def _md_variance_and_types(report: InvestigationReport) -> str:
    """Constant / low-variance and dtype-issue sections."""
    lines = ["## 6. Constant and low-variance features\n"]
    if report.variance:
        lines.append(
            md_table(
                ["Column", "Distinct values", "Dominant value", "Dominance", "Kind"],
                [
                    [
                        v.column, f"{v.n_unique:,}", ellipsize(str(v.dominant_value), 24),
                        f"{v.dominant_rate * 100:.2f}%",
                        "constant" if v.is_constant else "near-constant",
                    ]
                    for v in report.variance
                ],
            )
        )
        for item in report.variance[:5]:
            if item.note:
                lines.append(f"- **{item.column}**: {item.note}")
    else:
        lines.append("No constant or near-constant columns were detected.\n")

    lines.append("\n## 7. Data-type issues\n")
    if report.dtype_issues:
        lines.append(
            md_table(
                ["Column", "Issue", "Stored type", "Suggested type", "Affected", "Detail"],
                [
                    [
                        issue.column, issue.issue, issue.stored_dtype,
                        issue.suggested_dtype, format_percent(issue.affected_rate),
                        issue.detail,
                    ]
                    for issue in report.dtype_issues
                ],
            )
        )
    else:
        lines.append("Stored data types match the content of every column.\n")
    return "\n".join(lines)


def _md_numerical(report: InvestigationReport) -> str:
    """Descriptive statistics table."""
    stats = report.numeric_stats
    lines = ["## 8. Numerical distributions\n"]
    if not stats:
        return "\n".join(lines + ["No numerical columns were available.\n"])
    lines.append(
        md_table(
            ["Column", "N", "Missing", "Mean", "Median", "Std", "Min", "Q1", "Q3",
             "Max", "Skew", "Kurtosis", "Zeros"],
            [
                [
                    s.column, f"{s.count:,}", f"{s.missing:,}", format_number(s.mean),
                    format_number(s.median), format_number(s.std), format_number(s.minimum),
                    format_number(s.q1), format_number(s.q3), format_number(s.maximum),
                    format_number(s.skewness), format_number(s.kurtosis),
                    format_percent(s.zero_share),
                ]
                for s in stats
            ],
        )
    )
    return "\n".join(lines)


def _md_outliers(report: InvestigationReport) -> str:
    """Outlier tables per method."""
    lines = [
        "## 9. Outlier analysis\n",
        "Outliers are extreme observations, not automatically errors. The IQR rule is "
        "distribution-free; the z-score assumes an approximately normal distribution; "
        "the modified z-score (MAD) is robust to skew.\n",
    ]
    rows = []
    by_column: dict[str, dict[str, Any]] = {}
    for result in report.outliers.results:
        entry = by_column.setdefault(result.column, {"column": result.column})
        entry[result.method] = (
            f"{result.count:,} ({format_percent(result.rate)})"
            if result.available else f"n/a — {result.reason}"
        )
    rows = list(by_column.values())
    method_columns = ["iqr", "iqr_extreme", "zscore", "modified_zscore"]
    if rows:
        lines.append(
            md_table(
                ["Column"] + [m.replace("_", " ") for m in method_columns],
                [[row["column"]] + [row.get(m, "-") for m in method_columns] for row in rows],
            )
        )
    else:
        lines.append("No numeric columns were available for outlier testing.\n")
    return "\n".join(lines)


def _md_correlations(report: InvestigationReport) -> str:
    """Correlation section: strongest pairs and the causation caveat."""
    correlation = report.correlations
    lines = ["## 10. Correlation analysis\n"]
    if correlation.unavailable_reason:
        return "\n".join(lines + [correlation.unavailable_reason + "\n"])

    lines.append(
        f"Method: **{correlation.method}**; "
        f"{len(correlation.columns_analysed)} columns; "
        f"{len(correlation.pairs)} distinct pairs. Self-correlations are excluded.\n"
    )
    top = correlation.pairs[:15]
    lines.append(
        md_table(
            ["Column A", "Column B", "r", "Strength", "Direction", "Pairs (n)", "p-value"],
            [
                [
                    p.column_a, p.column_b, f"{p.r:+.4f}", p.strength, p.direction,
                    f"{p.n_observations:,}",
                    "-" if p.p_value is None else f"{p.p_value:.3g}",
                ]
                for p in top
            ],
        )
    )
    if correlation.multicollinear:
        lines.append(
            "\n**Multicollinearity (|r| >= "
            f"{config.MULTICOLLINEARITY_FLAG})**: "
            + "; ".join(
                f"{p.column_a} / {p.column_b} (r = {p.r:+.3f})"
                for p in correlation.multicollinear[:10]
            )
            + ".\n"
        )
    lines.append(
        "\nCorrelation is a measure of **association**, not causation. A high "
        "coefficient can arise from a shared driver, from selection in the data or "
        "from the way a column was computed. Inspect scatter plots before interpreting "
        "any relationship causally.\n"
    )
    return "\n".join(lines)


def _md_categorical(report: InvestigationReport) -> str:
    """Categorical distribution table."""
    stats = report.categorical_stats
    lines = ["## 11. Categorical distributions\n"]
    if not stats:
        return "\n".join(lines + ["No categorical columns were available.\n"])
    lines.append(
        md_table(
            ["Column", "N", "Unique", "Top value", "Top count", "Top %", "2nd %",
             "Rare levels", "Entropy (bits)", "Flags"],
            [
                [
                    s.column, f"{s.count:,}", f"{s.n_unique:,}",
                    ellipsize(str(s.top_value), 24), f"{s.top_count:,}",
                    f"{s.top_rate * 100:.2f}%", f"{s.second_rate * 100:.2f}%",
                    f"{s.rare_categories:,}",
                    "-" if s.entropy is None else f"{s.entropy:.2f}",
                    ", ".join(
                        filter(None, [
                            "dominant" if s.is_dominant else "",
                            "high-cardinality" if s.is_high_cardinality else "",
                        ])
                    ) or "-",
                ]
                for s in stats
            ],
        )
    )
    dominant = [s for s in stats if s.is_dominant]
    if dominant:
        worst = max(dominant, key=lambda s: s.top_rate)
        lines.append(
            f"\n**Dominant category warning:** {worst.top_rate * 100:.1f}% of "
            f"'{worst.column}' observations belong to the single category "
            f"'{ellipsize(str(worst.top_value), 40)}'.\n"
        )
    return "\n".join(lines)


def _md_anomalies(report: InvestigationReport) -> str:
    """Anomaly detection section."""
    anomalies = report.anomalies
    lines = ["## 12. Statistical anomalies (Isolation Forest)\n"]
    if not anomalies.available:
        return "\n".join(lines + [f"Not available: {anomalies.reason}\n"])

    lines.append(
        f"- Rows scored: {anomalies.n_rows_scored:,}\n"
        f"- Features used: {len(anomalies.features_used)} "
        f"({', '.join(anomalies.features_used[:12])}{'...' if len(anomalies.features_used) > 12 else ''})\n"
        f"- Contamination setting: {anomalies.contamination:.3f}\n"
        f"- Anomalies detected: {anomalies.n_anomalies:,} "
        f"({format_percent(anomalies.anomaly_rate)})\n"
    )
    if anomalies.reason:
        lines.append(f"- Note: {anomalies.reason}\n")

    if not anomalies.top_deviating_features.empty:
        lines.append("\n**Features in which anomalies deviate most (robust z)**\n")
        lines.append(
            md_table(
                ["Feature", "Mean robust |z| in anomalies", "Max robust |z|"],
                [
                    [row["feature"], f"{row['mean_robust_z']:.2f}",
                     f"{row['max_abs_robust_z']:.2f}"]
                    for row in _as_records(anomalies.top_deviating_features.head(10))
                ],
            )
        )

    records = anomalies.records
    if not records.empty and "is_anomaly" in records.columns:
        flagged = records[records["is_anomaly"]]
        if not flagged.empty:
            display_columns = [c for c in flagged.columns
                               if c not in {"is_anomaly", "anomaly_score", "anomaly_decision"}][:10]
            lines.append("\n**Most anomalous records**\n")
            lines.append(
                md_table(
                    ["row_index"] + [str(c) for c in display_columns],
                    [
                        [row.get("row_index")] + [row.get(c) for c in display_columns]
                        for row in _as_records(flagged.head(config.MAX_EXAMPLE_ROWS))
                    ],
                )
            )

    lines.append(
        "\nAnomaly detection identifies **unusual observations, not necessarily "
        "fraudulent or incorrect ones**. Isolation Forest isolates points that are easy "
        "to separate from the rest of the data; a rare but perfectly valid record will "
        "be flagged as readily as a corrupted one.\n"
    )
    return "\n".join(lines)


def _md_leakage(report: InvestigationReport) -> str:
    """Leakage signals section, always phrased as "potential"."""
    leakage = report.leakage
    lines = ["## 13. Potential data leakage\n"]
    lines.append(
        "Leakage cannot be proven from a dataset alone: it depends on how the data was "
        "collected and when each value becomes available. The items below are "
        "**heuristic signals worth investigating**, not verdicts.\n"
    )
    if leakage.target:
        lines.append(f"Target used for target-aware checks: `{leakage.target}`\n")
    else:
        lines.append("No target column was identified, so target-aware checks were skipped.\n")

    actionable = [s for s in leakage.signals if s.signal != "target_selection"]
    if not actionable:
        lines.append("\nNo leakage signals fired.\n")
        return "\n".join(lines)

    lines.append(
        "\n"
        + md_table(
            ["Column", "Signal", "Severity", "Confidence", "Reasoning", "Action"],
            [
                [s.column, s.signal, s.severity.value, s.confidence,
                 ellipsize(s.reasoning, 220), ellipsize(s.recommendation, 160)]
                for s in actionable
            ],
        )
    )
    return "\n".join(lines)


def _md_recommendations(report: InvestigationReport) -> str:
    """Final action list."""
    lines = ["## 14. Recommendations\n"]
    for index, recommendation in enumerate(report.recommendations, start=1):
        lines.append(f"{index}. {recommendation}")
    return "\n".join(lines) + "\n"


def _md_appendix(report: InvestigationReport) -> str:
    """Methodology, configuration and limitations."""
    return "\n".join([
        "## Appendix A — Methodology\n",
        "- **Missing data**: per-cell null counts on the raw frame; no imputation is "
        "applied to the dataset itself.",
        "- **Duplicates**: exact row equality across every column (pandas "
        "`duplicated`), reported as a count, a rate and example records.",
        "- **Constant / near-constant**: a column is near-constant when one value covers "
        f">= {config.NEAR_CONSTANT_DOMINANCE:.0%} of its non-missing values.",
        "- **Outliers**: IQR fences (k = "
        f"{config.IQR_MULTIPLIER}, extreme k = {config.IQR_EXTREME_MULTIPLIER}), standard "
        f"z-score (|z| > {config.ZSCORE_THRESHOLD}) and modified z-score "
        f"(|z_MAD| > {config.MODIFIED_ZSCORE_THRESHOLD}).",
        "- **Correlation**: pairwise-complete Pearson / Spearman / Kendall coefficients; "
        "p-values are computed for the strongest pairs only (budget: "
        f"{config.MAX_PAIRS_FOR_PVALUE:,} pairs).",
        "- **Categorical association**: bias-corrected Cramer's V with a chi-square test.",
        "- **Anomalies**: scikit-learn Isolation Forest on median-imputed, standardised "
        "numeric features.",
        "- **Health score**: weighted sum of five measured dimensions "
        f"({', '.join(f'{k} {v:.0%}' for k, v in config.HEALTH_WEIGHTS.items())}).",
        "- **Leakage**: heuristic rules over names, cardinality, duplicated columns, "
        "target correlation and single-feature separation.",
        "\n## Appendix B — Limitations\n",
        "1. Only delimited text files (CSV/TSV/PSV) are supported; Excel, Parquet, JSON "
        "and databases must be exported first.",
        "2. Correlation and association measures capture monotone/linear structure only; "
        "non-linear dependencies can hide from these statistics.",
        "3. Isolation Forest is scale-aware and neighbourhood-based: on very wide or "
        "heavily correlated feature sets it can over-flag correlated regions.",
        "4. Leakage screening is name- and pattern-based. It cannot see the data "
        "collection timeline or the model's deployment context.",
        "5. The health score measures hygiene, not scientific validity: a clean dataset "
        "can still be biased, unrepresentative or irrelevant to the question asked.",
        "6. Anomaly scores and outlier counts depend on the chosen contamination and "
        "threshold settings, which are shown with every result so they can be challenged.",
        "\n## Appendix C — Configuration used\n",
        md_table(
            ["Parameter", "Value"],
            [
                ["Contamination", f"{report.anomalies.contamination:.3f}"],
                ["Correlation method", report.correlations.method],
                ["Isolation Forest estimators", config.ANOMALY_DEFAULT_ESTIMATORS],
                ["Random seed", config.RANDOM_STATE],
                ["Analysis time", f"{report.analysis_seconds:.2f}s"],
                ["Generated (UTC)", report.generated_at],
            ],
        ),
    ]) + "\n"


def build_markdown_report(report: InvestigationReport) -> str:
    """Full Markdown investigation report."""
    overview = report.overview
    header = [
        "# DATA AUTOPSY REPORT\n",
        f"**Dataset:** {overview.filename}  ",
        f"**Generated:** {report.generated_at} (UTC)  ",
        f"**Rows:** {overview.n_rows:,}  ",
        f"**Columns:** {overview.n_columns:,}  ",
        f"**Memory:** {human_bytes(overview.memory_bytes)}  ",
        f"**Health:** {report.health.score:.1f}/100 ({report.health.grade})  ",
        f"**Analysis time:** {report.analysis_seconds:.2f}s\n",
        "## 1. Executive summary\n",
        health_verdict(report.health) + "\n",
        "**Major findings**\n",
    ]
    header += [f"- {line}" for line in build_executive_summary(report)]
    header.append("\n**Recommendations**\n")
    header += [f"{i}. {r}" for i, r in enumerate(report.recommendations, start=1)]

    sections = [
        _md_overview(report),
        _md_health(report),
        _md_missing(report),
        _md_duplicates(report),
        _md_variance_and_types(report),
        _md_numerical(report),
        _md_outliers(report),
        _md_correlations(report),
        _md_categorical(report),
        _md_anomalies(report),
        _md_leakage(report),
        _md_recommendations(report),
        _md_appendix(report),
    ]
    footer = (
        "\n---\n\n"
        f"Generated by DATA AUTOPSY on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        "All statistics were computed from the uploaded dataset; no synthetic or "
        "imputed values were introduced during analysis.\n"
    )
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + footer


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
_HTML_STYLE = """
:root {
  --bg: #0D1117; --panel: #161B22; --panel-2: #1C2128; --border: #30363D;
  --text: #C9D1D9; --muted: #8B949E; --accent: #58A6FF;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 20px 64px; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 14px; line-height: 1.6;
}
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: 1.5px; }
h2 { font-size: 20px; margin: 40px 0 12px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
h3 { font-size: 15px; margin: 22px 0 8px; color: var(--accent); text-transform: uppercase; letter-spacing: 1px; }
p, li { color: var(--text); }
a { color: var(--accent); }
.muted { color: var(--muted); }
.subtitle { color: var(--muted); margin-bottom: 22px; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin: 18px 0 6px; }
.card {
  flex: 1 1 150px; background: var(--panel); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px;
}
.card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
.card .note { color: var(--muted); font-size: 12px; }
.score-ring {
  display: inline-flex; align-items: center; justify-content: center;
  width: 116px; height: 116px; border-radius: 50%; font-size: 30px; font-weight: 700;
}
table { width: 100%; border-collapse: collapse; margin: 10px 0 6px; font-size: 13px; }
th, td { border: 1px solid var(--border); padding: 7px 9px; text-align: left; vertical-align: top; }
th { background: var(--panel-2); color: var(--muted); font-weight: 600; }
tr:nth-child(even) td { background: rgba(255,255,255,0.015); }
.badge {
  display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px;
  font-weight: 700; letter-spacing: .5px;
}
.finding { background: var(--panel); border: 1px solid var(--border); border-left-width: 4px;
  border-radius: 8px; padding: 12px 16px; margin: 10px 0; }
.finding h4 { margin: 0 0 6px; font-size: 14px; }
.finding p { margin: 4px 0; }
.finding .rec { color: var(--muted); font-size: 13px; }
.chart { background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px; margin: 14px 0; }
ol, ul { padding-left: 22px; }
footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; }
code { background: var(--panel-2); padding: 1px 5px; border-radius: 4px; }
"""


def _html_score_card(score: float) -> str:
    """Circular score indicator drawn with an inline SVG ring."""
    color = "#3FB950" if score >= 75 else "#D29922" if score >= 60 else "#F85149"
    circumference = 2 * 3.141592653589793 * 52
    filled = circumference * max(0.0, min(1.0, score / 100.0))
    return (
        '<svg width="132" height="132" viewBox="0 0 132 132" role="img" '
        'aria-label="Dataset health score">'
        f'<circle cx="66" cy="66" r="52" fill="none" stroke="#21262D" stroke-width="11"/>'
        f'<circle cx="66" cy="66" r="52" fill="none" stroke="{color}" stroke-width="11" '
        f'stroke-linecap="round" stroke-dasharray="{filled:.1f} {circumference - filled:.1f}" '
        'transform="rotate(-90 66 66)"/>'
        f'<text x="66" y="72" text-anchor="middle" fill="{color}" font-size="26" '
        f'font-weight="700">{score:.0f}</text>'
        f'<text x="66" y="90" text-anchor="middle" fill="#8B949E" font-size="11">/100</text>'
        '</svg>'
    )


def _html_cards(report: InvestigationReport) -> str:
    """Metric cards shown under the report header."""
    overview = report.overview
    cards = [
        ("Rows", f"{overview.n_rows:,}", f"{overview.n_columns:,} columns"),
        ("Cells", f"{overview.total_cells:,}", f"{overview.missing_cells:,} missing"),
        ("Missing", format_percent(overview.missing_rate, 2),
         f"{report.missing.columns_with_missing} column(s) affected"),
        ("Duplicates", f"{overview.duplicate_rows:,}", format_percent(report.duplicates.duplicate_rate, 2)),
        ("Numeric cols", f"{len(overview.column_types.numeric):,}",
         f"{report.outliers.total_columns_with_outliers} with outliers"),
        ("Anomalies", f"{report.anomalies.n_anomalies:,}" if report.anomalies.available else "n/a",
         format_percent(report.anomalies.anomaly_rate, 2) if report.anomalies.available else report.anomalies.reason[:40]),
        ("Leakage signals", f"{report.leakage.n_columns_flagged:,}",
         "columns to investigate" if report.leakage.n_columns_flagged else "none detected"),
        ("Memory", human_bytes(overview.memory_bytes), f"analysed in {report.analysis_seconds:.2f}s"),
    ]
    return '<div class="cards">' + "".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="note">{html.escape(str(note))}</div></div>'
        for label, value, note in cards
    ) + "</div>"


def _html_findings(findings: Sequence[Finding], limit: int = 40) -> str:
    """Findings rendered as bordered cards."""
    shown = [f for f in findings if f.severity != Severity.INFO][:limit]
    if not shown:
        return '<p class="muted">No findings above informational level.</p>'
    blocks = []
    for finding in shown:
        color = config.SEVERITY_COLORS[finding.severity.value]
        blocks.append(
            f'<div class="finding" style="border-left-color:{color}">'
            f"<h4>{_severity_badge(finding.severity.value)} {html.escape(finding.title)}</h4>"
            f"<p>{html.escape(finding.detail)}</p>"
            + (f'<p class="rec"><strong>Action:</strong> {html.escape(finding.recommendation)}</p>'
               if finding.recommendation else "")
            + "</div>"
        )
    return "".join(blocks)


def build_html_report(
    report: InvestigationReport,
    figures: dict[str, str] | None = None,
    include_plotlyjs: str | bool = "cdn",
) -> str:
    """Render the full HTML report.

    Args:
        report: The investigation to render.
        figures: Optional mapping of section title to an HTML chart fragment
            (as produced by :func:`src.charts.figure_to_html`).
        include_plotlyjs: ``"cdn"`` (small file, needs internet), ``True``/``"inline"``
            (fully offline, larger file) or ``False`` (assumes plotly.js is present).
    """
    figures = figures or {}
    chart_blocks = ""
    for title, fragment in figures.items():
        chart_blocks += (
            f'<h3>{html.escape(title)}</h3><div class="chart">{fragment}</div>'
        )

    overview = report.overview
    columns_rows = [
        [
            row.get("column"), row.get("role", ""), row.get("dtype"),
            f"{int(row.get('non_null', 0)):,}",
            f"{float(row.get('missing_pct', 0.0)):.2f}%",
            f"{int(row.get('unique', 0)):,}",
        ]
        for row in _as_records(overview.dtype_table)
    ]

    missing_rows = [
        [c.column, f"{c.missing:,}", f"{c.missing_rate * 100:.2f}%", c.dtype,
         _severity_badge(_missing_band(c.missing_rate))]
        for c in report.missing.affected_columns
    ]

    numeric_rows = [
        [
            s.column, f"{s.count:,}", format_number(s.mean), format_number(s.median),
            format_number(s.std), format_number(s.minimum), format_number(s.q1),
            format_number(s.q3), format_number(s.maximum), format_number(s.skewness),
            format_number(s.kurtosis), format_percent(s.zero_share),
        ]
        for s in report.numeric_stats
    ]

    outlier_rows = []
    grouped: dict[str, dict[str, Any]] = {}
    for result in report.outliers.results:
        entry = grouped.setdefault(result.column, {"column": result.column})
        entry[result.method] = (
            f"{result.count:,} ({format_percent(result.rate)})" if result.available
            else f'<span class="muted">n/a — {html.escape(result.reason)}</span>'
        )
    for entry in grouped.values():
        outlier_rows.append([
            entry["column"], entry.get("iqr", "-"), entry.get("iqr_extreme", "-"),
            entry.get("zscore", "-"), entry.get("modified_zscore", "-"),
        ])

    correlation_rows = [
        [p.column_a, p.column_b, f"{p.r:+.4f}", p.strength, p.direction,
         f"{p.n_observations:,}", "-" if p.p_value is None else f"{p.p_value:.3g}"]
        for p in report.correlations.pairs[:20]
    ]

    categorical_rows = [
        [s.column, f"{s.count:,}", f"{s.n_unique:,}", ellipsize(str(s.top_value), 28),
         f"{s.top_rate * 100:.2f}%", f"{s.second_rate * 100:.2f}%",
         ", ".join(filter(None, [
             "dominant" if s.is_dominant else "",
             "high-cardinality" if s.is_high_cardinality else "",
             "many rare levels" if s.rare_categories >= 5 else "",
         ])) or "-"]
        for s in report.categorical_stats
    ]

    health_rows = [
        [c.label, c.measurement, f"{c.score:.1f}", f"{c.weight:.0%}",
         f"{c.weighted_score:.2f}", c.penalty_reason or "-"]
        for c in report.health.components
    ]

    leakage = [s for s in report.leakage.signals if s.signal != "target_selection"]
    leakage_rows = [
        [s.column, s.signal, _severity_badge(s.severity.value), s.confidence,
         html.escape(s.reasoning), html.escape(s.recommendation)]
        for s in leakage
    ]

    anomaly_section = "<p>Not available: " + html.escape(report.anomalies.reason) + "</p>"
    if report.anomalies.available:
        deviation_rows = [
            [row["feature"], f"{row['mean_robust_z']:.2f}", f"{row['max_abs_robust_z']:.2f}"]
            for row in _as_records(report.anomalies.top_deviating_features.head(10))
        ]
        records = report.anomalies.records
        flagged = records[records["is_anomaly"]] if not records.empty else pd.DataFrame()
        display_columns = [c for c in flagged.columns
                           if c not in {"is_anomaly", "anomaly_score", "anomaly_decision"}][:10]
        anomaly_section = (
            "<ul>"
            f"<li>Rows scored: {report.anomalies.n_rows_scored:,}</li>"
            f"<li>Features used: {len(report.anomalies.features_used)}</li>"
            f"<li>Contamination: {report.anomalies.contamination:.3f}</li>"
            f"<li>Anomalies: {report.anomalies.n_anomalies:,} "
            f"({format_percent(report.anomalies.anomaly_rate)})</li>"
            "</ul>"
            + ("<h3>Where anomalies deviate</h3>"
               + html_table(["Feature", "Mean robust |z|", "Max robust |z|"], deviation_rows)
               if deviation_rows else "")
            + ("<h3>Most anomalous records</h3>"
               + html_table(
                   ["row_index"] + [str(c) for c in display_columns],
                   [[row.get("row_index")] + [row.get(c) for c in display_columns]
                    for row in _as_records(flagged.head(config.MAX_EXAMPLE_ROWS))],
               ) if not flagged.empty else "")
        )

    summary_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in build_executive_summary(report)
    )
    recommendation_items = "".join(
        f"<li>{html.escape(item)}</li>" for item in report.recommendations
    )
    health_notes = "".join(f"<li>{html.escape(n)}</li>" for n in report.health.notes)
    formula = (
        "score = &Sigma; (sub-score<sub>i</sub> &times; weight<sub>i</sub>), with linear "
        "sub-scores that reach 0 at fixed saturation thresholds: missing cells 25%, "
        "outlier cells 5%, duplicate rows 10%; type problems and low-variance columns are "
        "scored as proportions of all columns."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DATA AUTOPSY — {html.escape(overview.filename)}</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
<div class="wrap">
  <h1>DATA AUTOPSY REPORT</h1>
  <p class="subtitle">
    <strong>{html.escape(overview.filename)}</strong> &middot; generated
    {html.escape(report.generated_at)} (UTC) &middot; analysed in
    {report.analysis_seconds:.2f}s &middot; analysis is deterministic and rule-based
    (no AI/model scoring is used)
  </p>

  <div class="cards">
    <div class="card" style="flex:0 0 160px">{_html_score_card(report.health.score)}
      <div class="note" style="text-align:center;margin-top:6px">
        Grade: <strong>{html.escape(report.health.grade)}</strong></div>
    </div>
    <div style="flex:1 1 520px">{_html_cards(report)}</div>
  </div>

  <h2>1. Executive summary</h2>
  <p>{html.escape(health_verdict(report.health))}</p>
  <h3>Major findings</h3>
  <ul>{summary_items}</ul>
  <h3>Recommendations</h3>
  <ol>{recommendation_items}</ol>

  <h2>2. Dataset overview</h2>
  <h3>Structure</h3>
  {html_table(["Column", "Role", "Stored type", "Non-null", "Missing %", "Unique"], columns_rows)}
  <h3>Column groups</h3>
  <ul>
    <li><strong>Numerical ({len(overview.column_types.numeric)})</strong>:
        {html.escape(", ".join(overview.column_types.numeric) or "none")}</li>
    <li><strong>Categorical ({len(overview.column_types.categorical)})</strong>:
        {html.escape(", ".join(overview.column_types.categorical) or "none")}</li>
    <li><strong>Datetime ({len(overview.column_types.datetime)})</strong>:
        {html.escape(", ".join(overview.column_types.datetime) or "none")}</li>
    <li><strong>Boolean ({len(overview.column_types.boolean)})</strong>:
        {html.escape(", ".join(overview.column_types.boolean) or "none")}</li>
  </ul>
  {('<h3>Loader notes</h3><ul>' + "".join(f"<li>{html.escape(w)}</li>" for w in overview.load_warnings) + "</ul>") if overview.load_warnings else ""}
  {chart_blocks}

  <h2>3. Dataset health score</h2>
  <p><strong>{report.health.score:.1f}/100 — {html.escape(report.health.grade)}.</strong></p>
  {html_table(["Dimension", "Measured", "Sub-score", "Weight", "Contribution", "Deduction"], health_rows)}
  <h3>How the score is calculated</h3>
  <p>{formula}</p>
  <ul>{health_notes}</ul>

  <h2>4. Missing data</h2>
  <ul>
    <li>Missing cells: {report.missing.missing_cells:,} of {report.missing.total_cells:,}
        ({format_percent(report.missing.missing_rate)})</li>
    <li>Columns affected: {report.missing.columns_with_missing}</li>
    <li>Rows with at least one missing value: {report.missing.rows_with_missing:,}
        ({format_percent(1 - report.missing.complete_row_rate)})</li>
  </ul>
  {html_table(["Column", "Missing", "Missing %", "Dtype", "Severity"], missing_rows)}

  <h2>5. Duplicate analysis</h2>
  <ul>
    <li>Exact duplicate rows: {report.duplicates.duplicate_rows:,}
        ({format_percent(report.duplicates.duplicate_rate)})</li>
    <li>Duplicate groups: {report.duplicates.duplicate_group_count:,}</li>
    <li>Distinct rows: {report.duplicates.unique_rows:,} of {report.duplicates.total_rows:,}</li>
  </ul>
  {html_table(
      [str(c) for c in (list(report.duplicates.examples.columns) or ["-"])],
      [[row.get(c) for c in report.duplicates.examples.columns]
       for row in _as_records(report.duplicates.examples)],
      max_rows=10,
  ) if not report.duplicates.examples.empty else '<p class="muted">No duplicate examples.</p>'}

  <h2>6. Constant and low-variance features</h2>
  {html_table(
      ["Column", "Distinct values", "Dominant value", "Dominance", "Kind"],
      [[v.column, f"{v.n_unique:,}", ellipsize(str(v.dominant_value), 28),
        f"{v.dominant_rate * 100:.2f}%", "constant" if v.is_constant else "near-constant"]
       for v in report.variance],
  ) if report.variance else '<p class="muted">None detected.</p>'}

  <h2>7. Data-type issues</h2>
  {html_table(
      ["Column", "Issue", "Stored type", "Suggested", "Affected", "Detail"],
      [[i.column, i.issue, i.stored_dtype, i.suggested_dtype,
        format_percent(i.affected_rate), html.escape(i.detail)] for i in report.dtype_issues],
  ) if report.dtype_issues else '<p class="muted">Stored types match column content.</p>'}

  <h2>8. Numerical distributions</h2>
  {html_table(
      ["Column", "N", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max", "Skew",
       "Kurtosis", "Zeros"], numeric_rows,
  ) if numeric_rows else '<p class="muted">No numerical columns.</p>'}

  <h2>9. Outliers</h2>
  <p class="muted">An outlier is an extreme observation, not automatically an error.
  The IQR rule is distribution-free, the z-score assumes approximate normality, and the
  modified z-score (MAD) resists skew and masking.</p>
  {html_table(["Column", "IQR (1.5)", "IQR (3.0)", "Z-score (3.0)", "Modified z (3.5)"],
              outlier_rows) if outlier_rows else '<p class="muted">No numerical columns.</p>'}

  <h2>10. Correlation</h2>
  {f'<p class="muted">{html.escape(report.correlations.unavailable_reason)}</p>'
     if report.correlations.unavailable_reason else
     html_table(["Column A", "Column B", "r", "Strength", "Direction", "Pairs (n)", "p-value"],
                correlation_rows)}
  <p class="muted">Correlation measures association, not causation: a shared driver or
  the way a column was computed can produce a high coefficient. Self-correlations are
  excluded.</p>

  <h2>11. Categorical distributions</h2>
  {html_table(["Column", "N", "Unique", "Top value", "Top %", "2nd %", "Flags"],
              categorical_rows) if categorical_rows else '<p class="muted">No categorical columns.</p>'}

  <h2>12. Statistical anomalies (Isolation Forest)</h2>
  {anomaly_section}
  <p class="muted">Anomaly detection highlights unusual observations; it does not imply
  fraud, error or invalidity.</p>

  <h2>13. Potential data leakage</h2>
  <p>Leakage cannot be proven from a dataset alone. Each row below is a heuristic signal
  worth investigating, with the reasoning that produced it.</p>
  {f'<p>Target used for target-aware checks: <code>{html.escape(str(report.leakage.target))}</code></p>'
     if report.leakage.target else '<p class="muted">No target column identified.</p>'}
  {html_table(["Column", "Signal", "Severity", "Confidence", "Reasoning", "Action"],
              leakage_rows) if leakage_rows else '<p class="muted">No leakage signals fired.</p>'}

  <h2>14. All findings</h2>
  {_html_findings(report.findings, limit=60)}

  <footer>
    Generated by <strong>DATA AUTOPSY</strong> — a rule-based statistical and
    data-quality investigation tool. Every reported statistic was computed from the
    uploaded dataset; no values were imputed or invented for this report, and no AI
    model contributes to any score. Interactive charts require an internet connection
    (Plotly is loaded from a CDN); every number in this document is also present as
    plain text.
  </footer>
</div>
{'<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>' if include_plotlyjs == "cdn" else ''}
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# File naming
# --------------------------------------------------------------------------- #
def suggested_filename(report: InvestigationReport, extension: str) -> str:
    """Build a safe, descriptive download filename."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", report.overview.filename.rsplit(".", 1)[0]) or "dataset"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"data_autopsy_{stem}_{stamp}.{extension.lstrip('.')}"
