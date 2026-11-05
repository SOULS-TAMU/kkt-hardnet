"""Run checked documentation examples in isolated working directories."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    scripts = sorted((ROOT / "docs/examples").glob("*.py"))
    if not scripts:
        raise RuntimeError("No checked documentation examples")
    environment = dict(
        os.environ,
        JAX_ENABLE_X64="true",
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    for script in scripts:
        with tempfile.TemporaryDirectory(prefix="hardnet-doc-example-") as temporary:
            subprocess.run(
                [sys.executable, str(script)],
                cwd=temporary,
                env=environment,
                check=True,
                timeout=240,
            )
    print(json.dumps({"examples_passed": len(scripts)}, indent=2))


if __name__ == "__main__":
    main()
