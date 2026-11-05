"""Run installed distributions outside the checkout, including numerical and export checks."""

import argparse
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PACKAGE = "kkthn"
DISTRIBUTION = "kkt-hardnet"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--distribution", default=DISTRIBUTION)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    module = importlib.import_module(PACKAGE)
    installed = Path(module.__file__).resolve()
    if root in installed.parents:
        raise RuntimeError(
            "Smoke test imported the checkout instead of an installed wheel"
        )
    assert version(args.distribution) == args.version
    assert module.__version__ == args.version
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
    )
    environment.pop("PYTHONPATH", None)
    scripts = sorted((root / "docs/examples").glob("*.py"))
    assert scripts, "No runnable release examples found"
    with tempfile.TemporaryDirectory(prefix=PACKAGE + "-smoke-") as directory:
        work = Path(directory)
        for script in scripts:
            target = work / script.name
            shutil.copyfile(script, target)
            subprocess.run(
                [sys.executable, "-I", str(target)],
                cwd=work,
                env=environment,
                check=True,
                timeout=300,
            )
    report = {
        "distribution": args.distribution,
        "version": version(args.distribution),
        "installed_module": str(installed),
        "examples": [path.name for path in scripts],
    }
    Path("package-smoke-results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
