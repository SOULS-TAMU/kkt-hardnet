"""Validate local documentation links/citations and build an offline HTML artifact."""

import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit
from markdown_it import MarkdownIt
from pybtex.database import parse_file
import yaml


def main():
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        p
        for p in root.rglob("*.md")
        if ".git" not in p.parts
        and not any(part.startswith(".") for part in p.relative_to(root).parts[:-1])
        and "site" not in p.parts
        and "fixtures" not in p.parts
    )
    files.append(root / ".github/validation/README.md")
    parser = MarkdownIt("commonmark").enable("table")
    failures = []
    output = root / "site"
    output.mkdir(exist_ok=True)
    checked = 0
    for file in files:
        source = file.read_text(encoding="utf-8")
        tokens = parser.parse(source)
        links = re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)', source)
        for token in tokens:
            for child in token.children or []:
                if child.type in {"link_open", "image"}:
                    links.append(child.attrGet("href") or child.attrGet("src"))
        for link in links:
            url = urlsplit(link)
            if url.scheme or link.startswith("//"):
                continue
            target = (file.parent / unquote(url.path)).resolve() if url.path else file
            checked += 1
            if not target.is_relative_to(root) or not target.exists():
                failures.append(
                    f"{file.relative_to(root)}: missing local target {link}"
                )
            elif url.fragment and target.suffix == ".md":
                headings = re.findall(
                    r"^#{1,6}\s+(.+?)\s*#*$", target.read_text(encoding="utf-8"), re.M
                )
                anchors = {
                    re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-")
                    for h in headings
                }
                if unquote(url.fragment) not in anchors:
                    failures.append(f"{file.relative_to(root)}: missing heading {link}")
        page = output / file.relative_to(root).with_suffix(".html")
        page.parent.mkdir(parents=True, exist_ok=True)
        rendered = parser.render(source)
        rendered = re.sub(r'(href=["\'][^"\']+)\.md(?=[#"\'])', r"\1.html", rendered)
        page.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
            + html.escape(file.stem)
            + "</title><style>body{max-width:960px;margin:3rem auto;padding:0 1rem;font:17px/1.6 system-ui}pre{overflow:auto;background:#f5f5f5;padding:1rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5rem}img{max-width:100%}</style><body>"
            + rendered
            + "</body></html>",
            encoding="utf-8",
        )
    for path in root.iterdir():
        if path.name == "assets":
            shutil.copytree(path, output / "assets", dirs_exist_ok=True)
        elif (
            path.is_file()
            and path.suffix != ".md"
            and path.name
            in {
                "LICENSE",
                "NOTICE",
                "AUTHORS",
                "CITATION.bib",
                "CITATION.cff",
                "pyproject.toml",
            }
        ):
            shutil.copy2(path, output / path.name)
    citation = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
    assert (
        citation["cff-version"] == "1.2.0" and citation["title"] and citation["authors"]
    )
    entries = parse_file(str(root / "CITATION.bib")).entries
    assert entries and all(entry.fields.get("title") for entry in entries.values())
    assert "[project]" in (root / "pyproject.toml").read_text()
    if failures:
        raise SystemExit("\n".join(failures))
    report = {
        "markdown_files": len(files),
        "local_links": checked,
        "citation_entries": len(entries),
        "html_build": "site/README.html",
    }
    (root / "documentation-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
