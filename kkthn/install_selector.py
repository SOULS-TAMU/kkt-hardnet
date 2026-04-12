from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _has_command(name: str) -> bool:
    return shutil.which(name) is not None


def _command_succeeds(command: list[str]) -> bool:
    try:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return False
    return result.returncode == 0


def is_native_windows() -> bool:
    return platform.system().lower() == "windows"


def dependency_mode() -> tuple[str, str]:
    forced = os.environ.get("KKTHN_REQUIREMENTS", "").strip().lower()
    if forced in {"cpu", "gpu"}:
        if forced == "gpu" and is_native_windows():
            return "gpu", "forced by KKTHN_REQUIREMENTS=gpu; native Windows JAX CUDA wheels may be unavailable"
        return forced, f"forced by KKTHN_REQUIREMENTS={forced}"
    if forced:
        raise RuntimeError("KKTHN_REQUIREMENTS must be either 'cpu' or 'gpu' when set.")

    if is_native_windows():
        return "cpu", "native Windows detected; JAX CUDA plugin wheels are not available for this platform"
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
        return "cpu", "CUDA_VISIBLE_DEVICES=-1"
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return "gpu", "CUDA_HOME/CUDA_PATH detected"
    if _has_command("nvidia-smi") and _command_succeeds(["nvidia-smi"]):
        return "gpu", "nvidia-smi detected"
    if _has_command("nvcc") and _command_succeeds(["nvcc", "--version"]):
        return "gpu", "nvcc detected"
    return "cpu", "CUDA was not detected"


def requirements_file(package_root: Path) -> tuple[Path, str, str]:
    mode, reason = dependency_mode()
    filename = "requirements.gpu.txt" if mode == "gpu" else "requirements.txt"
    local_file = package_root / filename
    if local_file.exists():
        return local_file, mode, reason
    repo_file = package_root.parent / filename
    if repo_file.exists():
        return repo_file, mode, reason
    raise FileNotFoundError(f"Could not find {filename} for kkthn installation.")


def read_requirements(path: Path) -> list[str]:
    requirements = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            continue
        requirements.append(line)
    return requirements


def selection_message(package_root: Path) -> str:
    selected_file, selected_mode, selected_reason = requirements_file(package_root)
    return f"kkthn install: using {selected_mode.upper()} dependencies from {selected_file.name} ({selected_reason})."
