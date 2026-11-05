# Build these docs

Use Python 3.12 for the documentation toolchain, separately from the
package's runtime support. In an isolated environment, from the repository root:

```shell
python -m pip install -r docs/requirements.txt
python -m pip install .
python -m sphinx -b html -W --keep-going docs docs/_build/html
python .github/scripts/documentation_examples.py
```

Open `docs/_build/html/index.html`. Generated HTML, doctrees and example outputs
are ignored by Git. The example runner uses temporary working directories and
does not overwrite trained benchmark artifacts.

The theme and navigation follow the previous KKT-HardNet Read the Docs site.
API pages import the real package; core imports are not mocked to hide missing
dependencies. The documentation dependency baseline is distinct from the wider
runtime compatibility matrix.

## Read the Docs

The repository-root `.readthedocs.yaml` installs the package at `.` for the current
`src/` layout, builds `docs/conf.py` and fails on Sphinx warnings. Import the
repository into the same Read the Docs account/organization as KKT-HardNet,
enable GitHub integration, and build `latest` from `main`. Activate release
versions when their matching source/configuration is ready; an archive branch
does not itself establish a published package release.

Read the Docs can provide pull-request previews when enabled for the project.
GitHub validation also uploads a complete HTML preview artifact. See
[documentation versioning](https://docs.readthedocs.com/platform/stable/versions.html).

## Hosted documentation

The current guide is published at [kkthn.readthedocs.io](https://kkthn.readthedocs.io/en/latest/).
The Read the Docs project builds the repository default branch, `main`; a
GitHub push webhook requests subsequent builds. GitHub Actions separately
checks the strict HTML build and executes the downloadable examples.
