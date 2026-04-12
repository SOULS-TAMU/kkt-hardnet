from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _command_succeeds(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return False
    return result.returncode == 0


def _cuda_available() -> bool:
    forced = os.environ.get("KKTHN_REQUIREMENTS", "").strip().lower()
    if forced in {"cpu", "gpu"}:
        return forced == "gpu"
    if forced:
        raise RuntimeError("KKTHN_REQUIREMENTS must be either 'cpu' or 'gpu' when set.")

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
        return False
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return True
    if _has_command("nvidia-smi") and _command_succeeds(["nvidia-smi"]):
        return True
    if _has_command("nvcc") and _command_succeeds(["nvcc", "--version"]):
        return True
    return False


def _requirements_file() -> Path:
    filename = "requirements.gpu.txt" if _cuda_available() else "requirements.txt"
    package_root = Path(__file__).resolve().parent
    local_file = package_root / filename
    if local_file.exists():
        return local_file
    repo_file = package_root.parent / filename
    if repo_file.exists():
        return repo_file
    raise FileNotFoundError(f"Could not find {filename} for kkthn installation.")


def _read_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


selected_file = _requirements_file()
print(f"kkthn install: using dependencies from {selected_file.name}")

setup(install_requires=_read_requirements(selected_file))
