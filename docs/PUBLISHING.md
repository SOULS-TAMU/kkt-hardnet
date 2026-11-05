# Production PyPI publishing

Distribution: **`kkt-hardnet`**. Import: **`kkthn`**. Version: **2.0.0**.

## Trusted Publisher

Configure production PyPI to trust `SOULS-TAMU/kkt-hardnet`, workflow `release.yml`,
environment `pypi`. The GitHub environment permits only `main` deployments.
Use the existing project's Publishing settings for `kkt-hardnet`; use pending
publishers for the first PURIN, OHAGI and DAEHN releases. No long-lived API token
is stored in these repositories.

## Consistency gates

The shared release check requires agreement between `pyproject.toml`, literal
runtime `__version__` values, CFF/BibTeX citations and the versioned changelog.
Release dates must agree before publication; an intended tag must be `v<version>`.
The built wheel and sdist must agree on name, version, Apache-2.0 license and
Python requirement, include LICENSE/NOTICE, and contain no direct dependency URLs.
Publishing an existing package-index version is rejected before the upload job.
Negative tests prove that version/tag mismatches and unreleased entries fail.

GitHub release tags identify the original PyPI publishing commit; repository-only
changes after publication belong under `Unreleased`. Existing release tags and
PyPI files are never moved or overwritten.

## Release sequence

1. Publish PURIN, then OHAGI, then KKTHN and DAEHN.
2. Dispatch **PyPI release** manually on `main`.
3. The workflow builds a source distribution and a wheel from it, validates
   metadata, and checks the installed candidate with actual numerical examples.
4. Python 3.15 candidate checks must pass on Linux, Windows and macOS before upload.
5. After upload, clean installations from **production PyPI only** are checked on
   Python 3.10–3.15 across Linux, Windows and macOS, plus Linux ARM64 on Python 3.13.
   Every installation runs `pip check` and the real projection/training/export
   examples outside the source checkout.
6. Review all jobs and their installation reports before announcing the release.

Build artifacts are not overwritten. Re-run failed verification jobs rather than
re-uploading an already published filename. A corrected package needs a new version.

## User installation

```shell
python -m pip install kkt-hardnet
```

OHAGI and PURIN are resolved automatically for frontend installs. Production
packages use their canonical names; the TestPyPI-only `kkthn` alias is not used.
To verify a release from a repository checkout in a clean environment:

```shell
python .github/scripts/package_install.py --version 2.0.0
python .github/scripts/package_smoke.py --version 2.0.0
```

## Optional Orbax on Windows

Recent Orbax wheels contain deeply nested compatibility fixtures. If pip reports
Windows path-length errors, an extended-length interpreter path can install the
optional dependency without changing system settings:

```powershell
$envPython = (Resolve-Path .venv\Scripts\python.exe).Path
& ("\\?\" + $envPython) -m pip install orbax-checkpoint
```

The base packages do not require Flax or Orbax. Numerical checkpoint and resume
checks cover both the functional JAX and optional Flax backbones.

[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)

---

# TestPyPI publishing

The production distribution is `kkt-hardnet`; the Python import is `kkthn`.
The fresh TestPyPI project is **`kkthn`**, because TestPyPI already contains an
older `kkt-hardnet` project. The test build changes the distribution name only
in a temporary source copy; the repository metadata and eventual PyPI package
name remain `kkt-hardnet`.
The prepared version is **2.0.0**. This stage does not publish to PyPI.

## Account setup

Configure a GitHub Trusted Publisher on **TestPyPI**, using:

| Field | Value |
| --- | --- |
| Project | `kkthn` |
| Owner | `SOULS-TAMU` |
| Repository | `kkt-hardnet` |
| Workflow filename | `testpypi.yml` |
| Environment | `testpypi` |

For a new project, use a pending publisher in the account's publishing settings.
For an existing project, add the publisher in that project's publishing settings.
This authorizes only this repository/workflow/environment combination to upload
this TestPyPI project with short-lived OIDC credentials. No API token is stored.

## Publish and verify

1. Publish in dependency order: PURIN, OHAGI, then KKTHN and DAEHN.
2. Run **TestPyPI release** manually from the repository's Actions tab on `main`.
3. The workflow builds a source distribution and a wheel from that source
   distribution, validates the description and metadata, then installs the
   candidate and runs numerical/training/export examples before uploading.
4. After upload, clean Linux, Windows and macOS jobs install the exact stack
   wheels from TestPyPI. Linux additionally checks Python 3.10 and 3.14.
   Runtime dependencies come from PyPI, and `pip check` must pass.
5. Inspect the install reports and ensure every job passes before a PyPI release.

To verify the uploaded version locally in a fresh virtual environment:

```shell
python .github/scripts/package_install.py --testpypi --version 2.0.0
python .github/scripts/package_smoke.py --distribution kkthn --version 2.0.0
```

The installer downloads the stack packages from TestPyPI with `--no-deps`, then
installs those wheel files with third-party dependencies from PyPI. It does not
mix two package indexes in one resolver operation. The smoke tests import the
installed distributions rather than the checkout, run from a temporary folder,
and check projections and applicable training/model export/load round trips.

TestPyPI does not allow overwriting a previously uploaded filename. If a release
needs correction, choose a new version and update its dependency test versions
consistently before rerunning. A failed post-upload check is not a successful release.

[Trusted Publishing documentation](https://docs.pypi.org/trusted-publishers/using-a-publisher/)
· [TestPyPI guide](https://packaging.python.org/en/latest/guides/using-testpypi/)
