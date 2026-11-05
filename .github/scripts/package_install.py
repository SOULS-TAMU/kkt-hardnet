"""Install exact stack wheels from PyPI or TestPyPI and verify dependencies."""

import argparse
from email.parser import BytesParser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

TARGET = "kkt-hardnet"
DEPENDENCIES = {"purin": "1.0.0", "ohagi": "1.0.0"}


def wheel_identity(path):
    with zipfile.ZipFile(path) as archive:
        metadata = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        fields = BytesParser().parsebytes(archive.read(metadata))
    return fields["Name"], fields["Version"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--local-dir", type=Path, help="Use matching local wheels before downloading"
    )
    parser.add_argument(
        "--testpypi", action="store_true", help="Use the isolated testing index"
    )
    args = parser.parse_args()
    target = "kkthn" if args.testpypi and TARGET == "kkt-hardnet" else TARGET
    index = (
        "https://test.pypi.org/simple/" if args.testpypi else "https://pypi.org/simple/"
    )
    requirements = dict(DEPENDENCIES, **{target: args.version})
    with tempfile.TemporaryDirectory(prefix="stack-wheels-") as directory:
        downloads = Path(directory)
        wheels = []
        origins = {}
        for name, version in requirements.items():
            matches = (
                []
                if args.local_dir is None
                else [
                    path
                    for path in args.local_dir.glob("*.whl")
                    if wheel_identity(path) == (name, version)
                ]
            )
            if len(matches) > 1:
                raise ValueError("Ambiguous local wheel: " + name)
            if matches:
                wheel = matches[0].resolve()
                origins[name] = "local build"
            else:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "download",
                        "--no-deps",
                        "--only-binary=:all:",
                        "--index-url",
                        index,
                        "--dest",
                        str(downloads),
                        name + "==" + version,
                    ],
                    check=True,
                )
                wheel = next(
                    path
                    for path in downloads.glob("*.whl")
                    if wheel_identity(path) == (name, version)
                )
                origins[name] = "TestPyPI" if args.testpypi else "PyPI"
            wheels.append(str(wheel))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://pypi.org/simple/",
                "--only-binary=:all:",
                "--report",
                "package-install-results.json",
                *wheels,
            ],
            check=True,
        )
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(
        json.dumps({"versions": requirements, "stack_wheel_origins": origins}, indent=2)
    )


if __name__ == "__main__":
    main()
