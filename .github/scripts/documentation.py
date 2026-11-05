"""Build the real Sphinx site and validate citation metadata."""

import json
import re
from urllib.parse import unquote, urlsplit
from pathlib import Path
import subprocess
import sys
from pybtex.database import parse_file
import yaml


def main():
    root = Path(__file__).resolve().parents[2]
    for link in re.findall(
        r"\]\(([^)]+)\)", (root / "README.md").read_text(encoding="utf-8")
    ):
        url = urlsplit(link)
        if url.scheme or not url.path:
            continue
        target = (root / unquote(url.path)).resolve()
        if not target.is_relative_to(root) or not target.exists():
            raise ValueError("Broken README target: " + link)
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    assert (
        citation["cff-version"] == "1.2.0" and citation["title"] and citation["authors"]
    )
    entries = parse_file(str(root / "CITATION.bib")).entries
    assert entries and all(entry.fields.get("title") for entry in entries.values())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-W",
            "--keep-going",
            str(root / "docs"),
            str(root / "site"),
        ],
        check=True,
    )
    report = {
        "source_pages": len(list((root / "docs").rglob("*.rst")))
        + len(list((root / "docs").rglob("*.md"))),
        "citation_entries": len(entries),
        "html_build": "site/index.html",
        "builder": "Sphinx / sphinx_rtd_theme",
        "warnings_as_errors": True,
    }
    (root / "documentation-results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
