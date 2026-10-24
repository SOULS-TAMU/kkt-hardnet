"""Track cognitive complexity, type errors, duplication and dependency boundaries."""

import argparse
import hashlib
import ast
from collections import Counter
import json
import re
import subprocess
import sys
from pathlib import Path
import complexipy


def static_cycles(root):
    files = {
        ".".join(p.relative_to(root / "src").with_suffix("").parts).removesuffix(
            ".__init__"
        ): p
        for p in (root / "src").rglob("*.py")
    }
    graph = {name: set() for name in files}
    for name, path in files.items():
        parent = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = (
                    parent.split(".")[: len(parent.split(".")) - node.level + 1]
                    if node.level
                    else []
                )
                owner = ".".join(prefix + ([node.module] if node.module else []))
                targets = [owner] + [owner + "." + a.name for a in node.names]
            graph[name].update(t for t in targets if t in files and t != name)

    # Reachability identifies strongly connected components, including lazy/type-only edges.
    def reachable(start):
        found, pending = set(), [start]
        while pending:
            item = pending.pop()
            if item not in found:
                found.add(item)
                pending.extend(graph[item] - found)
        return found

    reach = {name: reachable(name) for name in graph}
    components = {
        tuple(sorted(other for other in reach[name] if name in reach[other]))
        for name in graph
    }
    return sorted(
        " -> ".join(component) for component in components if len(component) > 1
    )


def measure(root, package, duplicates):
    complexities = {}
    forbidden = {"kkthn", "daehn", "nlpoptnet"} - {package}
    violations = []
    for path in sorted((root / "src").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        for function in complexipy.code_complexity(source).functions:
            key = relative + ":" + function.name
            complexities[key] = max(complexities.get(key, 0), function.complexity)
        for node in ast.walk(ast.parse(source)):
            imports = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in imports:
                if name.split(".")[0] in forbidden:
                    violations.append(f"{relative}:{node.lineno}: {name}")
    # Fixed target/dependency policy makes the baseline independent of optional stubs.
    command = [
        sys.executable,
        "-m",
        "mypy",
        "src/" + package,
        "--follow-imports=skip",
        "--ignore-missing-imports",
        "--python-version=3.12",
        "--platform=linux",
        "--show-error-codes",
        "--no-error-summary",
        "--no-pretty",
        "--no-incremental",
    ]
    run = subprocess.run(command, cwd=root, text=True, capture_output=True)
    if run.returncode not in (0, 1):
        raise RuntimeError(run.stdout + run.stderr)
    errors = Counter()
    for line in run.stdout.splitlines():
        match = re.match(r"(.+?):\d+(?::\d+)?: error: (.+)", line)
        if match:
            errors[match[1].replace("\\", "/") + ": " + match[2]] += 1
    if run.returncode == 1 and not errors:
        raise RuntimeError(
            "mypy failed without parseable diagnostics: " + run.stdout + run.stderr
        )
    report = json.loads(duplicates.read_text(encoding="utf-8"))
    total = report["statistics"]["total"]
    return {
        "complexity": complexities,
        "type_errors": dict(errors),
        "duplicated_lines": total["duplicatedLines"],
        "duplication_percent": float(total["percentage"]),
        "boundary_violations": violations,
        "static_cycles": static_cycles(root),
        "duplicate_fingerprints": dict(
            Counter(
                hashlib.sha256(
                    (
                        "|".join(
                            sorted(
                                [
                                    clone["firstFile"]["name"].replace("\\", "/"),
                                    clone["secondFile"]["name"].replace("\\", "/"),
                                ]
                            )
                        )
                        + "|"
                        + " ".join(clone["fragment"].split())
                    ).encode()
                ).hexdigest()
                for clone in report["duplicates"]
            )
        ),
    }


def regressions(current, baseline):
    failures = list(current["boundary_violations"])
    for name, score in current["complexity"].items():
        allowed = max(15, baseline["complexity"].get(name, 15))
        if score > allowed:
            failures.append(f"Cognitive complexity {name}: {score} > {allowed}")
    for error, count in current["type_errors"].items():
        if count > baseline["type_errors"].get(error, 0):
            failures.append(f"New type errors: {error} ({count})")
    if current["duplicated_lines"] > baseline["duplicated_lines"]:
        failures.append("Duplicated line count increased.")
    for cycle in set(current.get("static_cycles", [])) - set(
        baseline.get("static_cycles", [])
    ):
        failures.append("New static import cycle: " + cycle)
    for fingerprint, count in current.get("duplicate_fingerprints", {}).items():
        if count > baseline.get("duplicate_fingerprints", {}).get(fingerprint, 0):
            failures.append("New duplicated block: " + fingerprint)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duplication_report", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    package = "kkthn"
    current = measure(root, package, args.duplication_report)
    output = root / "quality-results.json"
    output.write_text(
        json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path = root / ".github/validation/quality.json"
    if args.write_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        failures = regressions(current, json.loads(path.read_text()))
        if failures:
            raise SystemExit("\n".join(failures))
    print(
        json.dumps(
            {
                "functions": len(current["complexity"]),
                "maximum_cognitive_complexity": max(
                    current["complexity"].values(), default=0
                ),
                "type_error_count": sum(current["type_errors"].values()),
                "duplicated_lines": current["duplicated_lines"],
                "duplication_percent": current["duplication_percent"],
                "boundary_violations": current["boundary_violations"],
                "static_import_cycles": len(current.get("static_cycles", [])),
            },
            indent=2,
        )
    )
    # Prove the gate rejects regressions without modifying package sources.
    changed = {**current, "duplicated_lines": current["duplicated_lines"] + 1}
    assert regressions(changed, current)


if __name__ == "__main__":
    main()
