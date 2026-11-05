"""Build release artifacts in a temporary source copy without changing production metadata."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    output = (root / "dist").resolve()
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="testpypi-source-") as directory:
        source = Path(directory) / "source"
        shutil.copytree(
            root,
            source,
            ignore=shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                "*.egg-info",
                "build",
                "dist",
                "site",
                ".pytest_cache",
                ".ruff_cache",
                ".mypy_cache",
                "validation-base",
                "quality-duplication",
                "coverage-site",
            ),
        )
        config = source / "pyproject.toml"
        text = config.read_text(encoding="utf-8")
        # The canonical kkt-hardnet name already belongs to an older TestPyPI project.
        # This test-only alias changes distribution metadata, not imports or source APIs.
        if 'name = "kkt-hardnet"' in text:
            text = text.replace('name = "kkt-hardnet"', 'name = "kkthn"', 1)
            config.write_text(text, encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "build", "--outdir", str(output), str(source)],
            check=True,
        )


if __name__ == "__main__":
    main()
