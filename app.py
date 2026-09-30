"""DATA AUTOPSY — Streamlit dashboard entry point.

Run locally with::

    streamlit run app.py

The dashboard is a thin presentation layer: all analysis happens in the
Streamlit-free modules under ``src/`` (see ``src/investigation.py`` for the
pipeline), so what you see here is exactly what the test-suite verifies.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src import config, ui_sections, ui_theme
from src.data_loader import DatasetLoadError, load_dataframe
from src.investigation import InvestigationOptions, run_investigation
from src.utils import format_percent, human_bytes

SAMPLE_CSV = Path(__file__).resolve().parent / "sample_data" / "students.csv"

MAX_UPLOAD_MB = 200
#: Rows analysed for very large files; the loader reports the truncation.
MAX_ROWS = 200_000

SECTIONS = [
    "Overview",
    "Health score",
    "Missing data",
    "Duplicates",
    "Columns & types",
    "Numerical statistics",
    "Outliers",
    "Correlations",
    "Categorical analysis",
    "Anomalies",
    "Potential leakage",
    "Investigation report",
    "Clean data",
]


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
def configure_page() -> None:
    """Set page metadata and inject the dashboard stylesheet."""
    st.set_page_config(
        page_title="DATA AUTOPSY — dataset investigation",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(ui_theme.CSS, unsafe_allow_html=True)


def initialise_state() -> None:
    """Create the session-state keys the dashboard relies on."""
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("frame", None)
    st.session_state.setdefault("data_hash", None)
    st.session_state.setdefault("run_options", None)
    st.session_state.setdefault("file_name", None)
    st.session_state.setdefault("load_warnings", [])
    st.session_state.setdefault("pending_autorun", False)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _data_fingerprint(payload: bytes) -> str:
    """Short, stable hash of an uploaded payload."""
    return hashlib.sha256(payload).hexdigest()[:16]


def sidebar_upload(show_demo: bool = True) -> tuple[bytes | None, str | None]:
    """File uploader plus the demo-dataset shortcut.

    Args:
        show_demo: Render the one-click demo button. Hidden once a dataset is loaded
            so the sidebar never offers two competing entry points.
    """
    st.sidebar.markdown(
        '<p class="da-brand">DATA <span>AUTOPSY</span></p>'
        '<p class="da-tagline">Statistical &amp; data-quality investigation</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.sidebar.file_uploader(
        "Upload a dataset (CSV / TSV / TXT)",
        type=["csv", "tsv", "txt"],
        help=f"Delimited text files up to {MAX_UPLOAD_MB} MB. Excel/Parquet/JSON must be "
             "exported to CSV first.",
    )

    if uploaded is not None:
        payload = uploaded.getvalue()
        if len(payload) > MAX_UPLOAD_MB * 1024 * 1024:
            st.sidebar.error(
                f"This file is larger than {MAX_UPLOAD_MB} MB. Sample it or export a "
                "smaller extract."
            )
            return None, None
        return payload, uploaded.name

    if show_demo and ui_sections.button("Load synthetic demo dataset", key="sidebar_demo"):
        if not SAMPLE_CSV.exists():
            st.sidebar.error(
                "sample_data/students.csv was not found. Run "
                "`python sample_data/make_sample_data.py` first."
            )
            return None, None
        return SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name

    return None, None


def sidebar_options(frame: pd.DataFrame) -> InvestigationOptions:
    """Investigation settings, all of which are shown in the report appendix."""
    st.sidebar.markdown("### Investigation settings")

    contamination = st.sidebar.slider(
        "Anomaly contamination",
        min_value=config.ANOMALY_CONTAMINATION_MIN,
        max_value=config.ANOMALY_CONTAMINATION_MAX,
        value=config.ANOMALY_DEFAULT_CONTAMINATION,
        step=0.001,
        format="%.3f",
        help="Expected share of anomalies. Higher values force the model to flag more "
             "rows, so keep it realistic (0.01–0.10 for most datasets).",
    )
    correlation_method = st.sidebar.selectbox(
        "Correlation method", ["pearson", "spearman", "kendall"], index=0,
        help="Pearson for linear relationships, Spearman/Kendall for monotone ones "
             "(more robust to outliers).",
    )
    include_mad = st.sidebar.checkbox(
        "Include modified z-score (MAD) outlier rule", value=True,
        help="Robust to skew and to the masking effect that hides outliers from the "
             "plain z-score.",
    )
    enable_anomalies = st.sidebar.checkbox("Run anomaly detection", value=True)
    enable_leakage = st.sidebar.checkbox("Screen for potential data leakage", value=True)

    target_options = ["Auto-detect"] + [str(c) for c in frame.columns]
    target_choice = st.sidebar.selectbox(
        "Target column (used by leakage checks)", target_options, index=0,
        help="Auto-detection looks for conventional names (target, label, outcome, "
             "is_*, *_flag). Override it when your target is named differently.",
    )

    return InvestigationOptions(
        contamination=float(contamination),
        correlation_method=str(correlation_method),
        include_modified_zscore=bool(include_mad),
        leakage_target=None if target_choice == "Auto-detect" else str(target_choice),
        enable_anomaly_detection=bool(enable_anomalies),
        enable_leakage_screening=bool(enable_leakage),
    )


def sidebar_dataset_summary(report, options: InvestigationOptions) -> None:
    """Compact summary of the loaded dataset and the active settings."""
    overview = report.overview
    st.sidebar.markdown("### Loaded dataset")
    st.sidebar.markdown(
        f"**{overview.filename}**  \n"
        f"{overview.n_rows:,} rows · {overview.n_columns} columns  \n"
        f"{human_bytes(overview.memory_bytes)} · "
        f"{format_percent(overview.missing_rate, 1)} missing  \n"
        f"Health **{report.health.score:.1f}/100** ({report.health.grade})"
    )
    st.sidebar.caption(
        f"contamination {options.contamination:.3f} · {options.correlation_method} · "
        f"target: {options.leakage_target or 'auto'}"
    )


def sidebar_navigation() -> str:
    """Section navigation."""
    st.sidebar.markdown("### Sections")
    return st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")


# --------------------------------------------------------------------------- #
# Investigation runner
# --------------------------------------------------------------------------- #
def load_payload(payload: bytes, filename: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Load an uploaded payload, reporting failures as user-facing messages."""
    try:
        result = load_dataframe(payload, filename=filename, max_rows=MAX_ROWS)
    except DatasetLoadError as exc:
        st.error(f"Could not read this file: {exc}")
        return None, []
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to the user
        st.error(
            f"Unexpected error while reading the file ({type(exc).__name__}). "
            "Verify that it is a delimited text file with a single header row."
        )
        return None, []
    return result.frame, result.warnings


def run_and_store(frame: pd.DataFrame, filename: str, warnings: list[str],
                  options: InvestigationOptions) -> None:
    """Run the investigation with a progress indicator and store the result."""
    with st.status("Running the investigation…", expanded=True) as status:
        progress_bar = st.progress(0.0)
        label = st.empty()

        def on_progress(step: str, fraction: float) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0))
            label.caption(f"{step} — {fraction * 100:.0f}%")

        try:
            report = run_investigation(
                frame, filename=filename, options=options,
                load_warnings=warnings, progress=on_progress,
            )
        except Exception as exc:  # noqa: BLE001 - surface a friendly failure instead
            status.update(label="Investigation failed", state="error")
            st.error(
                f"The analysis could not be completed ({type(exc).__name__}: {exc}). "
                "Try disabling anomaly detection or leakage screening, or upload a "
                "smaller extract."
            )
            return

        status.update(
            label=f"Investigation complete in {report.analysis_seconds:.2f}s — "
                  f"health {report.health.score:.1f}/100",
            state="complete",
            expanded=False,
        )

    st.session_state["report"] = report
    st.session_state["frame"] = frame
    st.session_state["file_name"] = filename
    st.session_state["run_options"] = options


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Wire the dashboard together."""
    configure_page()
    initialise_state()

    payload, filename = sidebar_upload(show_demo=st.session_state.get("frame") is None)

    # A newly uploaded payload is loaded immediately so the option widgets can be
    # populated with its column names.
    frame: pd.DataFrame | None = st.session_state.get("frame")
    if payload is not None and filename is not None:
        fingerprint = _data_fingerprint(payload)
        if fingerprint != st.session_state.get("data_hash"):
            frame, warnings = load_payload(payload, filename)
            if frame is None:
                st.session_state["report"] = None
                st.session_state["frame"] = None
                return
            st.session_state["frame"] = frame
            st.session_state["data_hash"] = fingerprint
            st.session_state["load_warnings"] = warnings
            st.session_state["pending_autorun"] = True

    frame = st.session_state.get("frame")

    if frame is None:
        # Empty state: offer the synthetic demo dataset as a one-click start.
        if ui_sections.render_landing():
            if SAMPLE_CSV.exists():
                demo_frame, demo_warnings = load_payload(
                    SAMPLE_CSV.read_bytes(), SAMPLE_CSV.name
                )
                if demo_frame is not None:
                    st.session_state["frame"] = demo_frame
                    st.session_state["file_name"] = SAMPLE_CSV.name
                    st.session_state["data_hash"] = _data_fingerprint(SAMPLE_CSV.read_bytes())
                    st.session_state["load_warnings"] = demo_warnings
                    st.session_state["pending_autorun"] = True
                    st.rerun()
            else:
                st.error(
                    "sample_data/students.csv was not found. Run "
                    "`python sample_data/make_sample_data.py` to generate it."
                )
        ui_sections.render_footer()
        return

    options = sidebar_options(frame)
    run_clicked = ui_sections.button("Run investigation", key="run_button", primary=True)

    previous_options = st.session_state.get("run_options")
    settings_changed = previous_options is not None and previous_options != options

    st.sidebar.markdown("---")
    if ui_sections.button("Clear dataset", key="clear_button"):
        for key in ("report", "frame", "data_hash", "run_options", "file_name"):
            st.session_state[key] = None
        st.rerun()

    if run_clicked or st.session_state.pop("pending_autorun", False):
        run_and_store(
            frame,
            st.session_state.get("file_name") or "dataset.csv",
            list(st.session_state.get("load_warnings") or []),
            options,
        )
        st.rerun()

    report = st.session_state.get("report")

    if report is not None:
        sidebar_dataset_summary(report, st.session_state.get("run_options") or options)
    section = sidebar_navigation()

    if report is None:
        st.markdown(
            ui_theme.hero(
                "Dataset loaded — investigation not started",
                f"{st.session_state.get('file_name', 'dataset')} · "
                f"{len(frame):,} rows × {frame.shape[1]} columns",
            ),
            unsafe_allow_html=True,
        )
        st.info("Press **Run investigation** in the sidebar to analyse this dataset.")
        for warning in st.session_state.get("load_warnings") or []:
            st.warning(warning)
        ui_sections.render_footer()
        return

    if settings_changed:
        st.warning(
            "Settings changed since this investigation ran. Press **Run investigation** "
            "to re-analyse with the new settings — the current view still reflects the "
            "previous run."
        )

    render_section(report, frame, section)
    ui_sections.render_footer()


def render_section(report, frame: pd.DataFrame, section: str) -> None:
    """Dispatch to the renderer for the selected section."""
    overview = report.overview

    if section == "Overview":
        st.markdown(
            ui_theme.hero(
                f"{overview.filename}",
                f"{overview.n_rows:,} rows · {overview.n_columns} columns · "
                f"{human_bytes(overview.memory_bytes)} · health "
                f"{report.health.score:.1f}/100 ({report.health.grade}) · analysed in "
                f"{report.analysis_seconds:.2f}s",
            ),
            unsafe_allow_html=True,
        )
        ui_sections.render_overview(report, frame)

    elif section == "Health score":
        ui_sections.render_health(report)
        st.markdown(ui_theme.section("Highest-severity findings"),
                    unsafe_allow_html=True)
        ui_sections.show_findings(report.critical_findings(), limit=8,
                                  empty_message="No high-severity findings.")

    elif section == "Missing data":
        ui_sections.render_missing(report, frame)

    elif section == "Duplicates":
        ui_sections.render_duplicates(report)

    elif section == "Columns & types":
        ui_sections.render_variance_and_types(report)

    elif section == "Numerical statistics":
        ui_sections.render_numerical(report, frame)

    elif section == "Outliers":
        ui_sections.render_outliers(report, frame)

    elif section == "Correlations":
        ui_sections.render_correlations(report, frame)

    elif section == "Categorical analysis":
        ui_sections.render_categorical(report, frame)

    elif section == "Anomalies":
        ui_sections.render_anomalies(report, frame)

    elif section == "Potential leakage":
        ui_sections.render_leakage(report)

    elif section == "Investigation report":
        ui_sections.render_report(report, frame)

    elif section == "Clean data":
        ui_sections.render_clean_data(
            frame,
            st.session_state.get("file_name") or "dataset.csv",
        )

    else:  # pragma: no cover - defensive
        st.info(f"Unknown section: {section}")


if __name__ == "__main__":
    main()

