"""Publish a badge and a small report from coverage.py's measured totals."""

import html
import json
import os
from pathlib import Path
import sys


def main(source, destination):
    report = json.loads(Path(source).read_text(encoding="utf-8"))
    totals = report["totals"]
    percent = float(totals["percent_covered"])
    color = "4c1" if percent >= 90 else "97ca00" if percent >= 75 else "dfb317"
    label = f"{percent:.1f}%"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="130" height="20" '
        'role="img" aria-label="Coverage: ' + label + '">'
        "<title>Coverage: " + label + "</title>"
        '<rect width="75" height="20" fill="#555"/>'
        '<rect x="75" width="55" height="20" fill="#' + color + '"/>'
        '<g fill="#fff" text-anchor="middle" font-family="Verdana,sans-serif" '
        'font-size="11"><text x="37.5" y="14">coverage</text>'
        '<text x="102.5" y="14">' + html.escape(label) + "</text></g></svg>\n"
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "coverage.svg").write_text(svg, encoding="utf-8")
    summary = {
        "commit": os.environ.get("GITHUB_SHA"),
        "measurement": "statement and branch coverage, entire package",
        "totals": totals,
    }
    (destination / "coverage-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    message = (
        f"## Coverage: {label}\n\n"
        "Measured over the entire package, including unexecuted modules. "
        "The percentage combines statements and branches.\n\n"
        f"- Covered statements: {totals['covered_lines']}/{totals['num_statements']}\n"
        f"- Covered branches: {totals['covered_branches']}/{totals['num_branches']}\n"
        f"- Source commit: {os.environ.get('GITHUB_SHA', 'local')}\n"
    )
    (destination / "README.md").write_text(message, encoding="utf-8")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(message)
    print(message)


if __name__ == "__main__":
    main(*sys.argv[1:])
