# Install

## Editable Install

From the repository root:

```bash
python -m pip install -e kkthn
python -c "import kkthn; print(kkthn.__version__)"
```

For CUDA 12:

```bash
python -m pip install -e "kkthn[cuda12]"
python -c "import kkthn; print(kkthn.__version__)"
```

## PyPI Install

After publishing:

```bash
pip install kkt-hardnet
```

For CUDA 12:

```bash
pip install "kkt-hardnet[cuda12]"
```

## Python 3.13

The package now uses conditional dependency pins:

- Python `< 3.13`: the original stack
- Python `>= 3.13`: a compatible stack with `jax==0.4.38`, `jaxlib==0.4.38`, `numpy==2.2.6`, and `scipy==1.16.3`

## CPU / GPU Dependency Selection

Dependency selection is explicit:

- `pip install kkt-hardnet` installs the default CPU dependency set
- `pip install "kkt-hardnet[cuda12]"` installs the CUDA 12 extra chosen by the user
- no device detection or `KKTHN_REQUIREMENTS` environment variable is used during installation
