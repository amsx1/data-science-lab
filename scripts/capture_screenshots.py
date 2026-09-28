"""Capture screenshots of the running dashboard with a headless browser.

Usage (the dashboard must already be running on the given URL)::

    streamlit run app.py --server.port 8501 &
    python scripts/capture_screenshots.py --url http://localhost:8501 --out screenshots

Requires ``pip install playwright`` and ``python -m playwright install chromium``.
The script loads the synthetic demo dataset through the real UI, so the images
show genuine application output rather than a mock-up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIEWPORT = {"width": 1600, "height": 1000}

#: Sections captured, in navigation order (name, file name).
SECTIONS: list[tuple[str, str]] = [
    ("Overview", "dashboard_overview.png"),
    ("Health score", "dashboard_health.png"),
    ("Missing data", "dashboard_missing.png"),
    ("Outliers", "dashboard_outliers.png"),
    ("Correlations", "dashboard_correlations.png"),
    ("Anomalies", "dashboard_anomalies.png"),
    ("Potential leakage", "dashboard_leakage.png"),
    ("Investigation report", "dashboard_report.png"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Screenshot the DATA AUTOPSY dashboard.")
    parser.add_argument("--url", default="http://localhost:8501", help="Dashboard URL.")
    parser.add_argument("--out", default="screenshots", help="Output directory.")
    parser.add_argument("--wait", type=int, default=90, help="Timeout in seconds.")
    return parser.parse_args()


def main() -> int:
    """Drive the dashboard and save one PNG per section."""
    args = parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed: pip install playwright && "
              "python -m playwright install chromium", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout_ms = args.wait * 1000

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        page.goto(args.url, wait_until="domcontentloaded", timeout=timeout_ms)
        page.wait_for_selector("text=DATA AUTOPSY", timeout=timeout_ms)
        page.wait_for_timeout(3000)                      # let Streamlit finish painting
        page.screenshot(path=str(out_dir / "dashboard_landing.png"))
        print("captured landing page")

        page.get_by_role("button", name="Load the synthetic demo dataset").click()
        page.wait_for_selector("text=Column profile", timeout=timeout_ms)
        page.wait_for_timeout(2500)

        sidebar = "section[data-testid='stSidebar']"
        for section, filename in SECTIONS:
            page.click(f"{sidebar} >> text=\"{section}\"")
            page.wait_for_timeout(3500)                 # rendering + chart layout
            page.screenshot(path=str(out_dir / filename))
            print(f"captured {section} -> {filename}")
            body = page.inner_text("body")
            if "Traceback" in body or "StreamlitAPIException" in body:
                print(f"WARNING: an error is visible in section '{section}'", file=sys.stderr)

        browser.close()

    print(f"Screenshots written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
