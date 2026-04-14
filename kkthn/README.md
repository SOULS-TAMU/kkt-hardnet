# kkthn

`kkthn` is the installable Python package for KKTHardNet. It provides:

- MLP backbone utilities.
- A JAX KKT hard projection layer using smoothed Fischer-Burmeister complementarity.
- Training utilities for data-based surrogate and inverse modeling.
- Self-contained `jaxmodel`, `solgen`, and problem factory utilities for QP,
  QCQP, NLP, nonconvex, and general examples.

## Install

From `D:\Projects\KKTHardNet`:

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
from kkthn.builder import ProblemBuilder
```

`ProblemBuilder` exposes the shared runner:

```python
ProblemBuilder.run(args, root=ROOT)                      # standard case folders
builder.run(args, root=ROOT, data=DATA, train=TRAIN)     # builder-defined case
```
