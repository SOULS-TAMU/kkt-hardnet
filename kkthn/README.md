# kkthn

`kkthn` is the installable Python package for KKTHardNet. It provides:

- MLP backbone utilities.
- A JAX KKT hard projection layer using smoothed Fischer-Burmeister complementarity.
- Training utilities for data-based surrogate and inverse modeling.
- Problem helpers that reuse the parent NLPOptNet factories when this package is used from the monorepo.

## Install

From `D:\Projects\NLPOptNet\KKTHardNet`:

```powershell
pip install -e kkthn
```

Or from inside this package folder:

```powershell
pip install -e .
```

By default, editable installation checks for CUDA using `CUDA_HOME`, `CUDA_PATH`, `nvidia-smi`, or `nvcc`.

- CUDA detected: dependencies are read from `requirements.gpu.txt`.
- CUDA not detected: dependencies are read from `requirements.txt`.
- Native Windows: CPU dependencies are used by default because JAX CUDA plugin wheels are not available there.

You can force either path:

```powershell
$env:KKTHN_REQUIREMENTS = "gpu"
pip install -e kkthn
```

```powershell
$env:KKTHN_REQUIREMENTS = "cpu"
pip install -e kkthn
```

To print the selected dependency mode before installing:

```powershell
python install_info.py
pip install -e .
```

## Import

```python
from kkthn import KKTTrainConfig, train_kkt_hardnet
```

QP/QCQP/NLP/nonconvex problem generation depends on the parent NLPOptNet repository modules (`scripts`, `jaxmodel`, and `solgen`). When running from this monorepo, `kkthn` discovers those paths automatically.
