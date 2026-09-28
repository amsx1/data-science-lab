"""Application tests: the Streamlit app must render every section without errors.

Streamlit's own ``AppTest`` harness runs the real script, executes the real
callbacks and reports exceptions, so this file verifies the dashboard end to end
(loading the demo dataset, running the investigation, visiting all sections and
producing the downloadable reports) without needing a browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"

#: Every navigation entry the sidebar exposes.
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
]


def _load_demo(app: AppTest) -> AppTest:
    """Click the landing-page button that loads the synthetic demo dataset."""
    app.button(key="landing_demo").click()
    app.run(timeout=180)
    return app


def test_app_starts_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=180).run()

    assert not app.exception
    assert app.title or app.markdown                     # something rendered
    assert any("DATA" in str(block.value) for block in app.markdown)


def test_demo_dataset_loads_and_investigation_runs() -> None:
    app = _load_demo(AppTest.from_file(str(APP_PATH), default_timeout=180).run())

    assert not app.exception
    # The investigation auto-runs after loading, so findings and metrics are present.
    assert app.session_state["report"] is not None
    assert app.session_state["frame"] is not None
    assert app.session_state["report"].health.score >= 0


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_renders(section: str) -> None:
    app = _load_demo(AppTest.from_file(str(APP_PATH), default_timeout=180).run())
    app.radio[0].set_value(section).run(timeout=180)

    assert not app.exception, f"section '{section}' raised: {app.exception}"


def test_disabling_anomaly_detection_changes_the_result() -> None:
    """The sidebar options must actually reach the analysis engine."""
    app = _load_demo(AppTest.from_file(str(APP_PATH), default_timeout=180).run())
    assert app.session_state["report"].anomalies.available is True

    # Checkbox order: modified z-score, anomaly detection, leakage screening.
    app.checkbox[1].set_value(False).run(timeout=180)
    app.button(key="run_button").click()
    app.run(timeout=180)

    assert not app.exception
    assert app.session_state["report"].anomalies.available is False


def test_download_buttons_are_offered() -> None:
    app = _load_demo(AppTest.from_file(str(APP_PATH), default_timeout=180).run())
    app.radio[0].set_value("Investigation report").run(timeout=180)

    assert not app.exception
    download_labels = [element.label for element in app.get("download_button")]
    assert any("Markdown" in label for label in download_labels)
    assert any("HTML" in label for label in download_labels)


def test_clear_dataset_returns_to_the_landing_page() -> None:
    app = _load_demo(AppTest.from_file(str(APP_PATH), default_timeout=180).run())
    app.button(key="clear_button").click()
    app.run(timeout=180)

    assert not app.exception
    assert app.session_state["report"] is None
    assert app.session_state["frame"] is None
    assert app.button(key="landing_demo")                  # landing page is back
