# Installation

## Development version

```shell
python -m pip install "purin @ git+https://github.com/SOULS-TAMU/Purin.git@9ab708b470c4d9359e57f7bd214b0b5515aa4115"
python -m pip install "ohagi @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
python -m pip install "git+https://github.com/SOULS-TAMU/kkt-hardnet.git@development"
```

For an editable checkout, first install the shared dependencies above, then run
`python -m pip install -e ".[dev]"` from this repository's root.
Published releases may differ from this development branch.

## Optional dependencies

```shell
python -m pip install "ohagi[flax,orbax] @ git+https://github.com/SOULS-TAMU/Ohagi.git@e28f687e840d6840b259e687323029c3cc49bd59"
```

Functional JAX backbones are the default. Flax supplies an optional backbone
adapter; Orbax stores training state for resumption. Neither is needed to load
an exported deployment bundle.

## Platforms

Python **3.10–3.14** is tested on **Linux, Windows and macOS**, with additional
Linux ARM64 and pinned JAX baseline checks. Package installation and optional
Flax/Orbax integrations are tested across all three OSes. These are CPU checks;
accelerator support follows the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Build

```shell
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

Packaging uses `src/` and `pyproject.toml`.

[Back to README](../README.md)
