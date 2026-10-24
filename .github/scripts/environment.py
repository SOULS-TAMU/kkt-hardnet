"""Record only public runtime configuration for cross-platform CI artifacts."""

from importlib import metadata
import json
from pathlib import Path
import platform
import sys

import jax


def main():
    versions = {}
    for name in (
        "kkt-hardnet",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "ohagi",
        "purin",
        "flax",
        "optax",
        "orbax-checkpoint",
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "python": sys.version,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "versions": versions,
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": bool(jax.config.jax_enable_x64),
    }
    Path("ci-environment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
