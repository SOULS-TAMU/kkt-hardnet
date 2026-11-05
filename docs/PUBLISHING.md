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
python .github/scripts/testpypi_install.py --version 2.0.0
python .github/scripts/package_smoke.py --version 2.0.0
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
