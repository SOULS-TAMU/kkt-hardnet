# Installation

Use Python 3.10 or newer and an isolated environment. The distribution name is
`kkt-hardnet`; the Python import is `kkthn`.

```shell
python -m venv .venv
# PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install kkt-hardnet
python -c "import kkthn; print(kkthn.__version__)"
```

PyPI resolves the shared-stack dependencies automatically. For development from a
local checkout, install the sibling source packages in dependency order using
`python -m pip install -e <folder>`: PURIN, OHAGI, then the frontend.
Start with CPU execution and choose supported JAX/CUDA dependencies if needed.

## Optional integrations

From a local stack checkout, install `python -m pip install "./ohagi[flax,orbax]"`
for optional Flax backbones and Orbax training checkpoints. OHAGI also offers
`optax`, `plotting` and `export` extras. Functional JAX backbones remain the default.
Deploying a compatible frozen JAX graph does not require Flax or Orbax.

The production release installation checks cover Python 3.10–3.15 on Linux,
Windows and macOS, plus Linux ARM64. See [publishing and verification](PUBLISHING.md).
