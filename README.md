# KKTHardNet

KKT-HardNet is a JAX pipeline for data-based regression with a hard KKT projection layer.

The runner is self-contained: QP, QCQP, NLP, and nonconvex problem factories,
`jaxmodel`, and `solgen` are bundled inside this codespace. The `general` case
loads `case/general/model_definition.py` and generates synthetic labels with
SciPy SLSQP. For builder-defined algebraic constraints, edit `build_problem()`
in `run_general.py`. See `docs/PROBLEM.md` for both styles.

Typical smoke runs:

```powershell
python main.py --type qp --action run --p 2 --n 4 --me 1 --mi 1 --samples 8 --epochs 1 --batch_size 2
python main.py --type general --action run --samples 8 --epochs 1 --batch_size 2
python run_general.py --mode forward --samples 8 --epochs 1 --batch_size 2
```

Set `noise_scale` in `case/*/data.json`, in `run_general.py`'s `DATA`
dictionary, or with `--noise_scale` to add `noise_scale * N(0, 1)` Gaussian
noise to generated labels.

`main.py` is for forward-mode cases. For builder-defined inverse problems, edit
`build_problem()`, `DATA["inv_param"]`, `DATA["inv_param_label"]`, and
optionally `DATA["inv_param_init"]` in `run_general.py`, then run:

```powershell
python run_general.py --mode inverse
```

The package module is `kkthn`.

## Editable Package Install

The installable package lives in `kkthn` and uses a `src` layout.

```powershell
pip install -e kkthn
```

`pip install -e kkthn` automatically chooses package requirements:

- CUDA detected: uses `kkthn/requirements.gpu.txt`
- CUDA not detected: uses `kkthn/requirements.txt`
- Native Windows: uses CPU dependencies by default because JAX CUDA plugin wheels are not available there.

To force one:

```powershell
$env:KKTHN_REQUIREMENTS = "gpu"  # or "cpu"
pip install -e kkthn
```

To print the selected dependency mode before installing:

```powershell
python kkthn\install_info.py
pip install -e kkthn
```

After that, package imports work from any directory in the active environment:

```python
from kkthn import KKTTrainConfig, train_kkt_hardnet
```
