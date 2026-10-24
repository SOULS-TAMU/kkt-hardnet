"""Freeze public Python signatures; additions are allowed, removals/changes need review."""

import argparse
import ast
import json
from pathlib import Path


def surface(root):
    entries = {}
    for path in sorted((root / "src").rglob("*.py")):
        module = ".".join(path.relative_to(root / "src").with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def collect(body, prefix=""):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("_") and node.name not in {
                        "__init__",
                        "__call__",
                    }:
                        continue
                    entries[f"{module}:{prefix}{node.name}"] = (
                        ast.unparse(node.args)
                        + " | decorators="
                        + ",".join(ast.unparse(d) for d in node.decorator_list)
                    )
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    collect(node.body, prefix + node.name + ".")
                    for field in node.body:
                        if isinstance(field, ast.AnnAssign) and isinstance(
                            field.target, ast.Name
                        ):
                            entries[
                                f"{module}:{prefix}{node.name}.{field.target.id}"
                            ] = ast.unparse(field.annotation) + (
                                "=" + ast.unparse(field.value)
                                if field.value is not None
                                else ""
                            )
                elif isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if "__all__" in names and isinstance(
                        node.value, (ast.List, ast.Tuple)
                    ):
                        for item in node.value.elts:
                            if isinstance(item, ast.Constant) and isinstance(
                                item.value, str
                            ):
                                entries[f"{module}:export:{item.value}"] = "declared"
                    if "_EXPORTS" in names and isinstance(node.value, ast.Dict):
                        for key, value in zip(node.value.keys, node.value.values):
                            if isinstance(key, ast.Constant):
                                entries[f"{module}:export:{key.value}"] = ast.unparse(
                                    value
                                )

        collect(tree.body)
    return entries


def compare(current, baseline):
    return {
        name: {"expected": value, "actual": current.get(name)}
        for name, value in baseline.items()
        if current.get(name) != value
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = root / ".github/validation/api.json"
    current = surface(root)
    if args.write_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        changes = compare(current, json.loads(path.read_text()))
        if changes:
            raise SystemExit(
                "Public API changed; review compatibility before updating the baseline:\n"
                + json.dumps(changes, indent=2)
            )
        assert compare({"f": "x"}, {"f": "x"}) == {}
        assert compare({}, {"f": "x"}) and compare({"f": "y"}, {"f": "x"})
        print(f"Public API compatible: {len(current)} declarations checked.")


if __name__ == "__main__":
    main()
