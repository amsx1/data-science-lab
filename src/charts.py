"""Plotly figure builders.

Every function returns a ready-to-render :class:`plotly.graph_objects.Figure` with
a consistent dark theme, sane axis ranges and no external assets. Degenerate
inputs (empty columns, fewer than two usable points) return an annotated empty
figure rather than raising, so a single odd column can never break the dashboard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src import config
from src.schema import AnomalyResult, Finding, HealthScore, MissingReport, OutlierResult
from src.utils import ellipsize, finite_values

#: Palette shared by the dashboard and the exported report.
PALETTE = {
    "bg": "#0E1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#C9D1D9",
    "muted": "#8B949E",
    "accent": "#58A6FF",
    "positive": "#3FB950",
    "warning": "#D29922",
    "danger": "#F85149",
    "violet": "#BC8CFF",
    "teal": "#39C5CF",
}
COLORWAY = [
    PALETTE["accent"], PALETTE["positive"], PALETTE["warning"],
    PALETTE["violet"], PALETTE["teal"], PALETTE["danger"],
]

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
)


def _apply_layout(figure: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    """Apply the shared dark theme to a figure."""
    figure.update_layout(
        title=dict(text=title, font=dict(size=14, color=PALETTE["text"])),
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"], family=FONT_FAMILY, size=12),
        colorway=COLORWAY,
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["muted"])),
        hoverlabel=dict(bgcolor=PALETTE["panel"], font=dict(color=PALETTE["text"])),
    )
    figure.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                        linecolor=PALETTE["grid"])
    figure.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
                        linecolor=PALETTE["grid"])
    return figure


def empty_figure(message: str, height: int = 240) -> go.Figure:
    """Placeholder figure that explains why nothing can be drawn."""
    figure = go.Figure()
    figure.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        font=dict(color=PALETTE["muted"], size=13),
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _apply_layout(figure, "", height=height)


# --------------------------------------------------------------------------- #
# Structure and quality
# --------------------------------------------------------------------------- #
def health_gauge(health: HealthScore) -> go.Figure:
    """Donut gauge for the overall health score."""
    score = float(health.score)
    color = PALETTE["positive"] if score >= 75 else (
        PALETTE["warning"] if score >= 60 else PALETTE["danger"]
    )
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="/100", font=dict(size=32, color=PALETTE["text"])),
            domain=dict(x=[0.02, 0.98], y=[0.18, 1.0]),   # leaves room for the grade label
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=PALETTE["muted"],
                          tickfont=dict(color=PALETTE["muted"], size=10)),
                bar=dict(color=color, thickness=0.28),
                bgcolor=PALETTE["panel"],
                borderwidth=0,
                steps=[
                    dict(range=[0, 40], color="rgba(248,81,73,0.18)"),
                    dict(range=[40, 60], color="rgba(240,136,62,0.16)"),
                    dict(range=[60, 75], color="rgba(210,153,34,0.16)"),
                    dict(range=[75, 90], color="rgba(63,185,80,0.14)"),
                    dict(range=[90, 100], color="rgba(63,185,80,0.22)"),
                ],
            ),
        )
    )
    # The grade is shown as the chart title: placing it inside the figure collides
    # with the score number on small screens.
    return _apply_layout(figure, f"Grade: {health.grade}", height=270)


def health_breakdown(health: HealthScore) -> go.Figure:
    """Horizontal bar chart of the five weighted health dimensions."""
    if not health.components:
        return empty_figure("No health components available.")
    labels = [c.label for c in health.components]
    scores = [c.score for c in health.components]
    weights = [c.weight for c in health.components]
    colors = [
        PALETTE["positive"] if s >= 75 else PALETTE["warning"] if s >= 60 else PALETTE["danger"]
        for s in scores
    ]
    figure = go.Figure(
        go.Bar(
            x=scores, y=labels, orientation="h", marker=dict(color=colors),
            text=[f"{s:.1f}  (weight {w:.0%})" for s, w in zip(scores, weights)],
            textposition="auto", hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
        )
    )
    figure.update_xaxes(range=[0, 100], title="Sub-score")
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, "Health score components", height=300)


def findings_by_severity(findings: list[Finding]) -> go.Figure:
    """Donut chart counting findings per severity level."""
    if not findings:
        return empty_figure("No findings recorded.")
    order = ["critical", "high", "moderate", "low", "info"]
    counts = {level: 0 for level in order}
    for finding in findings:
        counts[finding.severity.value] += 1
    labels = [level.capitalize() for level in order if counts[level] > 0]
    values = [counts[level] for level in order if counts[level] > 0]
    colors = [config.SEVERITY_COLORS[level] for level in order if counts[level] > 0]
    figure = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.55,
            marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=2)),
            textinfo="label+value", textfont=dict(size=11),
        )
    )
    return _apply_layout(figure, "Findings by severity", height=300)


def missing_values_bar(missing: MissingReport, limit: int = 15) -> go.Figure:
    """Horizontal bar chart of missing share per affected column."""
    if not missing.affected_columns:
        return empty_figure("No missing values detected in any column.", height=260)
    top = missing.affected_columns[:limit]
    names = [c.column for c in top][::-1]
    rates = [c.missing_rate * 100 for c in top][::-1]
    counts = [c.missing for c in top][::-1]
    colors = [
        PALETTE["danger"] if r >= 50 else PALETTE["warning"] if r >= 20 else PALETTE["accent"]
        for r in rates
    ]
    figure = go.Figure(
        go.Bar(
            x=rates, y=[ellipsize(n, 28) for n in names], orientation="h",
            marker=dict(color=colors),
            text=[f"{r:.1f}% ({c:,})" for r, c in zip(rates, counts)],
            textposition="auto",
            hovertemplate="%{y}<br>%{x:.2f}% missing<extra></extra>",
        )
    )
    figure.update_xaxes(title="Missing (%)", range=[0, 100])
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, "Missing values by column", height=max(280, 26 * len(names) + 110))


def missingness_matrix(frame: pd.DataFrame, max_rows: int = 400) -> go.Figure:
    """Binary heatmap showing *where* values are missing across the table."""
    if frame.empty:
        return empty_figure("Empty dataset.")
    sample = frame if len(frame) <= max_rows else frame.sample(max_rows, random_state=config.RANDOM_STATE)
    sample = sample.sort_index()
    matrix = sample.isna().astype(int)
    if matrix.to_numpy().sum() == 0:
        return empty_figure("No missing values to visualise.", height=260)

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy().T,
            x=[str(i) for i in matrix.index],
            y=[ellipsize(str(c), 26) for c in matrix.columns],
            colorscale=[[0, PALETTE["panel"]], [1, PALETTE["danger"]]],
            showscale=False,
            hovertemplate="row %{x}<br>%{y}: %{z}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Row index", showticklabels=False)
    figure.update_yaxes(autorange="reversed")
    height = max(260, min(700, 22 * matrix.shape[1] + 120))
    return _apply_layout(
        figure,
        f"Missingness map (capped at {max_rows} rows; red = missing)",
        height=height,
    )


# --------------------------------------------------------------------------- #
# Numerical distributions and outliers
# --------------------------------------------------------------------------- #
def numeric_histogram(
    frame: pd.DataFrame, column: str, outlier_indices: list[int] | None = None
) -> go.Figure:
    """Histogram with an optional overlay marking detected outliers."""
    if column not in frame.columns:
        return empty_figure(f"Column '{column}' is not available.")
    values = finite_values(frame[column])
    if values.size == 0:
        return empty_figure(f"Column '{column}' has no finite numeric values.")

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values, nbinsx=min(60, max(10, int(np.sqrt(values.size)) + 1)),
            marker=dict(color=PALETTE["accent"], line=dict(color=PALETTE["bg"], width=0.5)),
            name="All values", hovertemplate="Value %{x}<br>Count %{y}<extra></extra>",
        )
    )

    if outlier_indices:
        positions = [i for i in outlier_indices if 0 <= i < len(frame)]
        if positions:
            outlier_values = finite_values(frame[column].iloc[positions])
            if outlier_values.size:
                figure.add_trace(
                    go.Histogram(
                        x=outlier_values, nbinsx=min(60, max(10, int(np.sqrt(values.size)) + 1)),
                        marker=dict(color=PALETTE["danger"]), name="Outliers (IQR)",
                        hovertemplate="Outlier value %{x}<br>Count %{y}<extra></extra>",
                    )
                )
                figure.update_layout(barmode="overlay")

    figure.update_xaxes(title=ellipsize(column, 40))
    figure.update_yaxes(title="Frequency")
    return _apply_layout(figure, f"Distribution of {ellipsize(column, 40)}")


def numeric_box(frame: pd.DataFrame, column: str) -> go.Figure:
    """Box plot with the IQR fences drawn as reference lines."""
    if column not in frame.columns:
        return empty_figure(f"Column '{column}' is not available.")
    values = finite_values(frame[column])
    if values.size == 0:
        return empty_figure(f"Column '{column}' has no finite numeric values.")

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    figure = go.Figure(
        go.Box(
            x=values, name=ellipsize(column, 30), boxpoints="outliers",
            marker=dict(color=PALETTE["accent"], outliercolor=PALETTE["danger"], size=4),
            line=dict(color=PALETTE["accent"]), fillcolor="rgba(88,166,255,0.25)",
        )
    )
    if iqr > 0:
        for fence, label in (
            (q1 - config.IQR_MULTIPLIER * iqr, "lower fence"),
            (q3 + config.IQR_MULTIPLIER * iqr, "upper fence"),
        ):
            figure.add_vline(
                x=fence, line=dict(color=PALETTE["warning"], dash="dash", width=1),
                annotation_text=label, annotation_font_color=PALETTE["muted"],
            )
    figure.update_xaxes(title=ellipsize(column, 40))
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, f"Box plot of {ellipsize(column, 40)}", height=320)


def outlier_scatter(
    frame: pd.DataFrame, column_x: str, column_y: str, anomaly_indices: list[int] | None = None
) -> go.Figure:
    """Scatter plot of two numeric columns, optionally highlighting flagged rows."""
    if column_x not in frame.columns or column_y not in frame.columns:
        return empty_figure("Select two numerical columns that exist in the dataset.")

    data = frame[[column_x, column_y]].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if data.shape[0] < 2:
        return empty_figure("Not enough paired numeric values to plot.")

    highlighted = set(anomaly_indices or []) & set(data.index)
    normal = data.loc[[i for i in data.index if i not in highlighted]]
    flagged = data.loc[sorted(highlighted)] if highlighted else data.iloc[0:0]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=normal[column_x], y=normal[column_y], mode="markers", name="Typical",
            marker=dict(color=PALETTE["accent"], size=5, opacity=0.55),
            hovertemplate=f"{ellipsize(column_x, 20)}=%{{x}}<br>{ellipsize(column_y, 20)}=%{{y}}<extra></extra>",
        )
    )
    if not flagged.empty:
        figure.add_trace(
            go.Scattergl(
                x=flagged[column_x], y=flagged[column_y], mode="markers", name="Flagged",
                marker=dict(color=PALETTE["danger"], size=9, symbol="x"),
                hovertemplate=f"{ellipsize(column_x, 20)}=%{{x}}<br>{ellipsize(column_y, 20)}=%{{y}}<extra></extra>",
            )
        )
    figure.update_xaxes(title=ellipsize(column_x, 30))
    figure.update_yaxes(title=ellipsize(column_y, 30))
    return _apply_layout(figure, f"{ellipsize(column_y, 26)} vs {ellipsize(column_x, 26)}")


def outlier_bars(results: list[OutlierResult], method: str = "iqr") -> go.Figure:
    """Outlier rate per numeric column for one method, split by severity band."""
    selected = [r for r in results if r.method == method and r.available]
    selected = [r for r in selected if r.count > 0]
    if not selected:
        return empty_figure(f"No {method} outliers detected in any column.", height=260)

    selected.sort(key=lambda r: r.rate, reverse=True)
    names = [ellipsize(r.column, 26) for r in selected]
    rates = [r.rate * 100 for r in selected]
    colors = [
        PALETTE["danger"] if r >= 15 else PALETTE["warning"] if r >= 5 else PALETTE["accent"]
        for r in rates
    ]
    figure = go.Figure(
        go.Bar(
            x=rates, y=names, orientation="h", marker=dict(color=colors),
            text=[f"{r:.2f}% ({res.count:,})" for r, res in zip(rates, selected)],
            textposition="auto", hovertemplate="%{y}<br>%{x:.2f}% of values<extra></extra>",
        )
    )
    figure.update_xaxes(title="Share of values flagged (%)")
    figure.update_yaxes(autorange="reversed")
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, f"Outlier rate per column ({method.upper().replace('_', ' ')})",
                         height=max(280, 26 * len(names) + 110))


# --------------------------------------------------------------------------- #
# Correlations and categories
# --------------------------------------------------------------------------- #
def correlation_heatmap(matrix: pd.DataFrame, method: str = "pearson") -> go.Figure:
    """Symmetric correlation heatmap with the diagonal blanked out."""
    if matrix.empty:
        return empty_figure("Correlation matrix is unavailable (need >= 2 numeric columns).")

    values = matrix.to_numpy(dtype="float64").copy()
    np.fill_diagonal(values, np.nan)     # self-correlation is never informative
    labels = [ellipsize(str(c), 22) for c in matrix.columns]

    figure = go.Figure(
        go.Heatmap(
            z=values, x=labels, y=labels,
            colorscale=[
                [0.0, "#1F6FEB"], [0.35, "#102A54"], [0.5, PALETTE["panel"]],
                [0.65, "#5A2D2D"], [1.0, "#F85149"],
            ],
            zmin=-1, zmax=1, zmid=0,
            colorbar=dict(title=dict(text="r", font=dict(color=PALETTE["muted"])),
                          tickfont=dict(color=PALETTE["muted"])),
            hovertemplate="%{y} vs %{x}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    size = 0.42 * len(labels) + 3.0
    figure.update_yaxes(autorange="reversed", tickfont=dict(size=min(12, 160 / max(len(labels), 4))))
    figure.update_xaxes(tickfont=dict(size=min(12, 160 / max(len(labels), 4))))
    return _apply_layout(
        figure, f"{method.capitalize()} correlation (diagonal hidden)", height=int(max(380, 34 * size))
    )


def correlation_bar(pairs: list, limit: int = 12) -> go.Figure:
    """Ranked bar chart of the strongest correlations (positive and negative)."""
    if not pairs:
        return empty_figure("No correlations to display.", height=260)
    top = pairs[:limit][::-1]
    labels = [f"{ellipsize(p.column_a, 18)} vs {ellipsize(p.column_b, 18)}" for p in top]
    values = [p.r for p in top]
    colors = [PALETTE["positive"] if v >= 0 else PALETTE["danger"] for v in values]
    figure = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h", marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in values], textposition="auto",
            hovertemplate="%{y}<br>r = %{x:.4f}<extra></extra>",
        )
    )
    figure.update_xaxes(title="Correlation coefficient (r)", range=[-1, 1])
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, "Strongest relationships", height=max(300, 26 * len(labels) + 110))


def category_bar(table: pd.DataFrame, column: str, max_categories: int = 15) -> go.Figure:
    """Frequency bar chart for a categorical column."""
    if table.empty:
        return empty_figure(f"Column '{column}' has no non-missing values to chart.", height=260)
    working = table.head(max_categories).copy()
    working = working[~working["value"].astype(str).str.startswith("(other")]
    figure = go.Figure(
        go.Bar(
            x=[ellipsize(str(v), 24) for v in working["value"]],
            y=working["count"],
            marker=dict(color=PALETTE["teal"]),
            text=[f"{p:.1f}%" for p in working["percent"]],
            textposition="auto",
            hovertemplate="%{x}<br>%{y:,} rows<extra></extra>",
        )
    )
    figure.update_xaxes(title=ellipsize(column, 30), tickangle=-30)
    figure.update_yaxes(title="Count")
    figure.update_layout(showlegend=False)
    return _apply_layout(figure, f"Category frequency — {ellipsize(column, 30)}")


def category_pie(table: pd.DataFrame, column: str, max_categories: int = 8) -> go.Figure:
    """Donut chart showing category dominance at a glance."""
    if table.empty:
        return empty_figure(f"Column '{column}' has no non-missing values to chart.", height=300)
    working = table.head(max_categories).copy()
    figure = go.Figure(
        go.Pie(
            labels=[ellipsize(str(v), 22) for v in working["value"]],
            values=working["count"], hole=0.5,
            marker=dict(line=dict(color=PALETTE["bg"], width=2)),
            textinfo="percent", textfont=dict(size=11),
            hovertemplate="%{label}<br>%{value:,} rows (%{percent})<extra></extra>",
        )
    )
    return _apply_layout(figure, f"Share of categories — {ellipsize(column, 28)}", height=320)


# --------------------------------------------------------------------------- #
# Anomalies
# --------------------------------------------------------------------------- #
def anomaly_scatter(
    anomalies: AnomalyResult, column_x: str, column_y: str
) -> go.Figure:
    """Scatter of the two most informative features with anomalies highlighted."""
    if not anomalies.available or anomalies.records.empty:
        return empty_figure(anomalies.reason or "Anomaly detection is unavailable.")
    if column_x not in anomalies.records.columns or column_y not in anomalies.records.columns:
        return empty_figure("The selected columns were not used in the anomaly model.")

    data = anomalies.records
    normal = data[~data["is_anomaly"]]
    flagged = data[data["is_anomaly"]]

    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=normal[column_x], y=normal[column_y], mode="markers", name="Typical",
            marker=dict(color=PALETTE["accent"], size=5, opacity=0.45),
            customdata=normal[["anomaly_percentile"]].to_numpy(),
            hovertemplate=f"{ellipsize(column_x, 20)}=%{{x}}<br>{ellipsize(column_y, 20)}=%{{y}}"
                          "<br>percentile %{customdata[0]:.1f}<extra></extra>",
        )
    )
    if not flagged.empty:
        figure.add_trace(
            go.Scattergl(
                x=flagged[column_x], y=flagged[column_y], mode="markers", name="Anomaly",
                marker=dict(color=PALETTE["danger"], size=9, symbol="x"),
                customdata=flagged[["anomaly_percentile"]].to_numpy(),
                hovertemplate=f"{ellipsize(column_x, 20)}=%{{x}}<br>{ellipsize(column_y, 20)}=%{{y}}"
                              "<br>percentile %{customdata[0]:.1f}<extra></extra>",
            )
        )
    figure.update_xaxes(title=ellipsize(column_x, 30))
    figure.update_yaxes(title=ellipsize(column_y, 30))
    return _apply_layout(figure, "Anomalies in feature space (Isolation Forest)")


def anomaly_score_distribution(anomalies: AnomalyResult) -> go.Figure:
    """Histogram of Isolation Forest scores with the decision threshold marked."""
    if not anomalies.available or anomalies.records.empty:
        return empty_figure(anomalies.reason or "Anomaly detection is unavailable.")
    scores = anomalies.records["anomaly_score"].to_numpy(dtype="float64")
    flagged = anomalies.records["is_anomaly"].to_numpy(dtype=bool)

    figure = go.Figure()
    figure.add_trace(
        go.Histogram(x=scores[~flagged], nbinsx=50, name="Typical",
                     marker=dict(color=PALETTE["accent"]),
                     hovertemplate="score %{x:.4f}<br>count %{y}<extra></extra>")
    )
    figure.add_trace(
        go.Histogram(x=scores[flagged], nbinsx=50, name="Anomaly",
                     marker=dict(color=PALETTE["danger"]),
                     hovertemplate="score %{x:.4f}<br>count %{y}<extra></extra>")
    )
    if flagged.any():
        threshold = float(np.max(scores[flagged]))
        figure.add_vline(x=threshold, line=dict(color=PALETTE["warning"], dash="dash"),
                         annotation_text="decision threshold",
                         annotation_font_color=PALETTE["muted"])
    figure.update_layout(barmode="overlay")
    figure.update_xaxes(title="Anomaly score (lower = more unusual)")
    figure.update_yaxes(title="Count")
    return _apply_layout(figure, "Distribution of anomaly scores")


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def figure_to_html(figure: go.Figure, include_plotlyjs: bool | str = False) -> str:
    """Serialise a figure to an embeddable HTML fragment."""
    return figure.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displayModeBar": False, "responsive": True},
    )


def explorer_scatter(frame: pd.DataFrame, column_x: str, column_y: str, color: str | None = None) -> go.Figure:
    """Quick interactive scatter used by the data-exploration tab."""
    if column_x not in frame.columns or column_y not in frame.columns:
        return empty_figure("Choose two columns that exist in the dataset.")
    try:
        figure = px.scatter(
            frame, x=column_x, y=column_y, color=color if color in frame.columns else None,
            opacity=0.6, render_mode="webgl",
        )
    except Exception:  # noqa: BLE001 - non-numeric or unhashable colour columns
        return empty_figure("These columns cannot be plotted as a scatter (non-numeric axes).")
    figure.update_traces(marker=dict(size=6))
    figure.update_layout(showlegend=bool(color in frame.columns))
    return _apply_layout(figure, f"{ellipsize(column_y, 26)} vs {ellipsize(column_x, 26)}")
