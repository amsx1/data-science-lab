"""Dashboard styling helpers.

Kept separate from ``ui_sections`` so the CSS and the small HTML component
builders can be reviewed (and reused) independently of the page layout.
Everything is inline-styled or scoped by class, so no external asset is needed.
"""

from __future__ import annotations

import html
from typing import Iterable, Sequence

from src import config
from src.schema import Finding, HealthScore, Severity

#: Colours mirrored from src.config so charts and HTML agree.
PALETTE = {
    "bg": "#0E1117",
    "panel": "#161B22",
    "panel_alt": "#1C2128",
    "border": "#30363D",
    "text": "#C9D1D9",
    "muted": "#8B949E",
    "accent": "#58A6FF",
}

#: Dashboard CSS. Streamlit's own dark theme is extended rather than replaced.
CSS = f"""
<style>
    .stApp {{ background: {PALETTE['bg']}; }}
    section[data-testid="stSidebar"] {{
        background: #0B0F14;
        border-right: 1px solid {PALETTE['border']};
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: {PALETTE['text']}; }}

    .da-brand {{
        font-size: 20px; font-weight: 800; letter-spacing: 3px; color: {PALETTE['text']};
        margin: 0;
    }}
    .da-brand span {{ color: {PALETTE['accent']}; }}
    .da-tagline {{ color: {PALETTE['muted']}; font-size: 12px; margin: 2px 0 14px; }}

    .da-hero {{
        background: linear-gradient(135deg, #131A22 0%, #0E1117 60%);
        border: 1px solid {PALETTE['border']}; border-radius: 14px;
        padding: 22px 26px; margin-bottom: 18px;
    }}
    .da-hero h1 {{ font-size: 30px; margin: 0 0 6px; letter-spacing: 2px; }}
    .da-hero p {{ color: {PALETTE['muted']}; margin: 0; font-size: 14px; }}

    .da-card {{
        background: {PALETTE['panel']}; border: 1px solid {PALETTE['border']};
        border-radius: 10px; padding: 12px 14px; height: 100%;
    }}
    .da-card .label {{
        color: {PALETTE['muted']}; font-size: 11px; text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .da-card .value {{ font-size: 22px; font-weight: 650; color: {PALETTE['text']}; }}
    .da-card .note {{ color: {PALETTE['muted']}; font-size: 11.5px; }}
    .da-card .accent {{ font-size: 22px; font-weight: 650; }}

    .da-section {{
        font-size: 19px; font-weight: 650; margin: 26px 0 2px; color: {PALETTE['text']};
    }}
    .da-section-sub {{ color: {PALETTE['muted']}; font-size: 13px; margin-bottom: 10px; }}

    .da-finding {{
        background: {PALETTE['panel']}; border: 1px solid {PALETTE['border']};
        border-left-width: 4px; border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    }}
    .da-finding .title {{ font-weight: 650; font-size: 14px; color: {PALETTE['text']}; }}
    .da-finding .body {{ color: {PALETTE['text']}; font-size: 13px; margin-top: 4px; }}
    .da-finding .action {{ color: {PALETTE['muted']}; font-size: 12.5px; margin-top: 6px; }}

    .da-badge {{
        display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 10.5px;
        font-weight: 700; letter-spacing: .6px; margin-right: 8px; vertical-align: middle;
    }}
    .da-note {{
        background: rgba(88,166,255,0.08); border: 1px solid rgba(88,166,255,0.35);
        border-radius: 8px; padding: 10px 14px; color: {PALETTE['text']}; font-size: 13px;
    }}
    .da-reco {{
        background: {PALETTE['panel_alt']}; border: 1px solid {PALETTE['border']};
        border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 13px;
    }}
    .da-footer {{ color: {PALETTE['muted']}; font-size: 12px; margin-top: 40px;
        border-top: 1px solid {PALETTE['border']}; padding-top: 14px; }}
    code {{ color: #A5D6FF; }}
</style>
"""


def severity_color(severity: Severity | str) -> str:
    """Hex colour for a severity value or name."""
    key = severity.value if isinstance(severity, Severity) else str(severity)
    return config.SEVERITY_COLORS.get(key, PALETTE["muted"])


def badge(severity: Severity | str, text: str | None = None) -> str:
    """Small coloured pill for a severity level."""
    key = severity.value if isinstance(severity, Severity) else str(severity)
    color = severity_color(key)
    label = text or key.upper()
    return (
        f'<span class="da-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55">{html.escape(label)}</span>'
    )


def card(label: str, value: str, note: str = "", accent: str | None = None) -> str:
    """Metric card as an HTML block (used instead of ``st.metric`` for styling)."""
    value_style = f' style="color:{accent}"' if accent else ""
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    return (
        '<div class="da-card">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value"{value_style}>{html.escape(value)}</div>'
        f"{note_html}</div>"
    )


def hero(title: str, subtitle: str) -> str:
    """Page hero banner."""
    return (
        f'<div class="da-hero"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p></div>"
    )


def section(title: str, subtitle: str = "") -> str:
    """Section heading with an optional explanatory subtitle."""
    sub = f'<div class="da-section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="da-section">{html.escape(title)}</div>{sub}'


def note(text: str) -> str:
    """Highlighted explanatory note."""
    return f'<div class="da-note">{text}</div>'


def recommendation(text: str, index: int | None = None) -> str:
    """Numbered recommendation block."""
    prefix = f"<strong>{index}.</strong> " if index is not None else ""
    return f'<div class="da-reco">{prefix}{html.escape(text)}</div>'


def finding_block(finding: Finding) -> str:
    """Render one finding as a bordered card."""
    color = severity_color(finding.severity)
    action = (
        f'<div class="action"><strong>Recommended action:</strong> '
        f"{html.escape(finding.recommendation)}</div>"
        if finding.recommendation else ""
    )
    return (
        f'<div class="da-finding" style="border-left-color:{color}">'
        f'<div class="title">{badge(finding.severity)}'
        f"{html.escape(finding.title)}</div>"
        f'<div class="body">{html.escape(finding.detail)}</div>'
        f"{action}</div>"
    )


def findings_html(findings: Iterable[Finding], limit: int = 12) -> str:
    """Concatenate several findings into one HTML block."""
    blocks = [finding_block(f) for f in list(findings)[:limit]]
    return "".join(blocks)


def health_summary_html(health: HealthScore) -> str:
    """One-line health summary with the grade highlighted."""
    color = (
        PALETTE["accent"] if health.score >= 75
        else config.SEVERITY_COLORS["moderate"] if health.score >= 60
        else config.SEVERITY_COLORS["critical"]
    )
    return note(
        f"<strong>Dataset health: <span style='color:{color}'>"
        f"{health.score:.1f}/100</span> — {html.escape(health.grade)}.</strong> "
        "The score is a weighted average of five measured dimensions; every deduction "
        "is listed in the Health tab."
    )


def footer_html(version: str = "1.0.0") -> str:
    """Dashboard footer."""
    return (
        '<div class="da-footer">DATA AUTOPSY v'
        f"{html.escape(version)} — deterministic, rule-based statistical analysis. "
        "No AI model contributes to any score or finding: every number is reproducible "
        "from the uploaded file. Anomaly detection is an unsupervised "
        "nearest-neighbour-style algorithm (Isolation Forest), not an AI judgement."
        "</div>"
    )


def columns_for(items: Sequence[str], per_row: int = 4) -> list[list[str]]:
    """Split items into rows of at most ``per_row`` entries (layout helper)."""
    return [list(items[i:i + per_row]) for i in range(0, len(items), per_row)]
