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

## Import

```python
from kkthn import KKTTrainConfig, train_kkt_hardnet
```

QP/QCQP/NLP/nonconvex problem generation depends on the parent NLPOptNet repository modules (`scripts`, `jaxmodel`, and `solgen`). When running from this monorepo, `kkthn` discovers those paths automatically.
