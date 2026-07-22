"""
Turns a saved load_test.json into a markdown report and an HTML dashboard.
Useful for regenerating a report from a load test you already ran (the
`--report` flag on load_test.py does this automatically for new runs).

Usage:
    python scripts/generate_load_test_report.py --input results/load_test.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.monitoring.load_test_report import generate_html_dashboard, generate_markdown_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/load_test.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    summary = json.loads(Path(args.input).read_text())

    md = generate_markdown_report(summary)
    md_path = Path(args.output_dir) / "load_test_report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)

    html_path = generate_html_dashboard(summary, output_path=str(Path(args.output_dir) / "load_test_dashboard.html"))

    print(f"Wrote {md_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
