# KKTHardNet

KKT-HardNet is a JAX pipeline for data-based regression with a hard KKT projection layer.

The runner reuses the existing NLPOptNet factories for QP, QCQP, NLP, and nonconvex label generation. The `general` case loads `case/general/model_definition.py` and generates synthetic labels with SciPy SLSQP. For string-defined constraints, edit the `PROBLEM` dictionary in `run_general.py`.

Typical smoke runs:

```powershell
python main.py --type qp --action run --p 2 --n 4 --me 1 --mi 1 --samples 8 --epochs 1 --batch_size 2
python main.py --type general --action run --samples 8 --epochs 1 --batch_size 2
python run_general.py --samples 8 --epochs 1 --batch_size 2
```

The package module is `kkthn`.

## Editable Package Install

The installable package lives in `kkthn`, using the same `src`-layout style as `nlpopt`.

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
