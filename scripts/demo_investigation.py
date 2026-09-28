"""Command-line runner: investigate a CSV without starting the dashboard.

Useful for CI, for scripted batch checks and for verifying the analysis engine
end-to-end:

    python scripts/demo_investigation.py sample_data/students.csv
    python scripts/demo_investigation.py my_data.csv --out reports/ --contamination 0.03

Exit codes: ``0`` success, ``1`` the dataset could not be loaded or analysed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the file directly from a clone (python scripts/demo_investigation.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import reporting  # noqa: E402
from src.data_loader import DatasetLoadError, load_dataframe  # noqa: E402
from src.investigation import InvestigationOptions, run_investigation  # noqa: E402
from src.utils import human_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a DATA AUTOPSY investigation and write Markdown/HTML reports."
    )
    parser.add_argument("csv_path", help="Path to the CSV file to investigate.")
    parser.add_argument("--out", default=None, help="Directory for the generated reports.")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Isolation Forest contamination (default: 0.05).")
    parser.add_argument("--correlation", default="pearson",
                        choices=["pearson", "spearman", "kendall"],
                        help="Correlation method (default: pearson).")
    parser.add_argument("--target", default=None,
                        help="Column to treat as the target for leakage checks.")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary block.")
    return parser.parse_args()


def main() -> int:
    """Run the investigation and print a short console summary."""
    args = parse_args()

    try:
        load_result = load_dataframe(args.csv_path)
    except DatasetLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    options = InvestigationOptions(
        contamination=args.contamination,
        correlation_method=args.correlation,
        leakage_target=args.target,
    )

    if not args.quiet:
        print(f"Loaded {load_result.filename}: "
              f"{len(load_result.frame):,} rows x {load_result.frame.shape[1]} columns "
              f"({human_bytes(load_result.frame.memory_usage(deep=True).sum())}, "
              f"encoding {load_result.encoding}, delimiter '{load_result.delimiter}')")
        for warning in load_result.warnings:
            print(f"  loader note: {warning}")

    def progress(label: str, fraction: float) -> None:
        if not args.quiet:
            print(f"  [{fraction * 100:5.1f}%] {label}")

    report = run_investigation(
        load_result.frame,
        filename=load_result.filename,
        options=options,
        load_warnings=load_result.warnings,
        progress=progress,
    )

    print("\n" + "=" * 72)
    print("DATA AUTOPSY REPORT")
    print("=" * 72)
    print(f"Dataset : {report.overview.filename}")
    print(f"Shape   : {report.overview.n_rows:,} rows x {report.overview.n_columns} columns")
    print(f"Health  : {report.health.score:.1f}/100 ({report.health.grade})")
    print("\nMajor findings")
    for line in reporting.build_executive_summary(report):
        print(f"  - {line}")
    print("\nRecommendations")
    for index, recommendation in enumerate(report.recommendations, start=1):
        print(f"  {index}. {recommendation}")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = out_dir / reporting.suggested_filename(report, "md")
        html_path = out_dir / reporting.suggested_filename(report, "html")
        markdown_path.write_text(reporting.build_markdown_report(report), encoding="utf-8")
        html_path.write_text(reporting.build_html_report(report), encoding="utf-8")
        print(f"\nReports written to:\n  {markdown_path}\n  {html_path}")

    print(f"\nCompleted in {report.analysis_seconds:.2f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
