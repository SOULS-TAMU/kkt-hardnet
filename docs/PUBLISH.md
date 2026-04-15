# Publish To PyPI

## 1. Update Metadata

Edit:

- `kkthn/pyproject.toml`
- `kkthn/src/kkthn/__init__.py`
- `docs/VERSION.md`

Keep the version aligned across those files.

## 2. Clean Old Artifacts

```bash
rm -rf kkthn/build dist kkthn/src/*.egg-info *.egg-info
```

## 3. Build

From the repository root:

```bash
python -m pip install --upgrade build twine
python -m build kkthn
```

That creates the source distribution and wheel under `dist/`.

## 4. Check The Distribution

```bash
python -m twine check kkthn/dist/*
```

## 5. Upload To TestPyPI

```bash
python -m twine upload --repository testpypi kkthn/dist/*
```

Install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple kkt-hardnet
```

Install the CUDA 12 extra from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple "kkt-hardnet[cuda12]"
```

## 6. Upload To PyPI

```bash
python -m twine upload dist/*
```

## 7. Verify

In a fresh environment:

```bash
pip install kkt-hardnet
python -c "from kkthn import KKTHardNet; print(KKTHardNet)"
```

For CUDA 12 verification:

```bash
pip install "kkt-hardnet[cuda12]"
python -c "from kkthn import KKTHardNet; print(KKTHardNet)"
```

## Notes

- The project name on PyPI is `kkt-hardnet`
- The import path stays `kkthn`
- CPU is the default install path
- CUDA support is exposed through the `cuda12` optional dependency extra
