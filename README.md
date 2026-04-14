# KKTHardNet

KKTHardNet is a JAX-based KKT-HardNet pipeline for learning decision variables
from problem parameters while enforcing equality, inequality, and bound
constraints through a hard projection layer.

The repository includes the runnable project, the editable `kkthn` package,
problem factories for QP/QCQP/NLP/nonconvex examples, and a symbolic builder for
general problems.

## Installation

### Windows PowerShell

```powershell
git clone <repo-url>
cd KKTHardNet

py -3.9 -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python kkthn\install_info.py

# Optional: force CPU or GPU dependency selection.
# $env:KKTHN_REQUIREMENTS = "cpu"
# $env:KKTHN_REQUIREMENTS = "gpu"

python -m pip install -e kkthn
python -c "import kkthn; print(kkthn.__version__)"
```

If `py -3.9` is unavailable, use:

```powershell
python -m venv env
```

Native Windows installs CPU dependencies by default. Use Linux or WSL for JAX
GPU wheels.

### Mac/Linux

```sh
git clone <repo-url>
cd KKTHardNet

python3 -m venv env
source env/bin/activate

python -m pip install --upgrade pip
python kkthn/install_info.py

# Optional: force CPU or GPU dependency selection.
# export KKTHN_REQUIREMENTS=cpu
# export KKTHN_REQUIREMENTS=gpu

python -m pip install -e kkthn
python -c "import kkthn; print(kkthn.__version__)"
```

On systems where `python` already points to Python 3, `python -m venv env` is
also fine.

## Building A General Problem

Edit `build_problem()` in [run_general.py](run_general.py). The symbolic builder
lets you name parameters, variables, optional inverse parameters, constraints,
bounds, and, when synthetic data is needed, an objective.

```python
from kkthn.builder import ProblemBuilder


def build_problem() -> ProblemBuilder:
    builder = ProblemBuilder(y_bound=4.0)

    x = builder.add_parameter(["x1", "x2"])
    theta = builder.add_inverse_parameter(DATA["inv_param"])
    y = builder.add_variable(["y1", "y2", "y3"])

    builder.objective = 0.5 * (y.y1**2 + y.y2**2 + y.y3**2)
    builder.constraints.add(
        theta.a0 * y.y1 + y.y2 - x.x1 == 0,
        y.y2 - theta.a1 * y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
    )
    builder.bounds.set(lower=-4.0, upper=4.0)
    return builder
```

For synthetic general data, KKTHardNet samples parameters from `DATA["x_L"]` and
`DATA["x_U"]`, solves the builder problem with SciPy SLSQP, writes
`parameters.csv` and `variables.csv`, and then trains normally.

`run_general.py` delegates preprocessing and training through the builder
instance:

```python
builder = build_problem()
builder.run(args, root=ROOT, data=DATA, train=TRAIN)
```

Run a small forward example:

```sh
python run_general.py --mode forward --samples 8 --epochs 1 --batch_size 2
```

Run inverse-parameter training:

```sh
python run_general.py --mode inverse --samples 8 --epochs 1 --batch_size 2
```

### General Problem From An Existing CSV

When the labels already exist, the builder can use a CSV dataset instead of
solving an optimization problem. In that case, define the symbols and
constraints, point the builder at the dataset, and omit the objective.

```python
from kkthn.builder import ProblemBuilder


def build_problem() -> ProblemBuilder:
    builder = ProblemBuilder(y_bound=4.0)

    x = builder.add_parameter(["x1", "x2"])
    y = builder.add_variable(["y1", "y2", "y3"])

    builder.constraints.add(
        y.y1 + y.y2 - x.x1 == 0,
        y.y2 - y.y3 - x.x2 == 0,
        y.y1**2 + y.y3**2 <= 2.0,
    )
    builder.bounds.set(lower=-4.0, upper=4.0)
    builder.set_dataset(
        "case/general/my_dataset.csv",
        parameter_columns=["x1", "x2"],
        variable_columns=["y1", "y2", "y3"],
    )
    return builder
```

The preprocessing step extracts the selected columns into the run folder as
`parameters.csv` and `variables.csv`; the training pipeline then uses those two
arrays exactly as usual.

## Running Main With Configurations

`main.py` runs the standard case folders:

- `case/qp`
- `case/qcqp`
- `case/nlp`
- `case/nonconvx`
- `case/general`

Internally, `main.py` uses the same runner entry point exposed by the builder
class:

```python
ProblemBuilder.run(args, root=ROOT)
```

Each case folder contains:

- `data.json`: problem dimensions, sampling settings, solver settings, and
  noise settings.
- `config.json`: training settings such as epochs, batch size, learning rate,
  hidden size, and train/validation split.
- `proj.json`: projection-layer settings.

Basic QP smoke run:

```sh
python main.py --type qp --action run --p 2 --n 4 --me 1 --mi 1 --samples 8 --epochs 1 --batch_size 2
```

Generate labels only:

```sh
python main.py --type qp --action data --samples 8
```

Run the block-structured general case:

```sh
python main.py --type general --action run --samples 8 --epochs 1 --batch_size 2
```

Useful overrides:

```sh
python main.py --type qp --samples 200 --epochs 100 --batch_size 40 --learning_rate 0.001
python main.py --type qcqp --p 5 --n 10 --me 5 --mi 5 --noise_scale 0.05
python main.py --type nlp --solver SCS --train_frac 0.8 --hidden_size 64 --hidden_layers 2
```

By default, run folders are written under `case/<type>/runs`. Use `--output_dir`
to choose another root:

```sh
python main.py --type qp --output_dir runs/qp_test --samples 8 --epochs 1
```

Each training run saves:

- `config.json`: one consolidated file containing the input `data`, `config`,
  and `proj` dictionaries used for the run.
- `history.csv`: per-epoch training and validation losses, constraint metrics,
  batch counts, epoch times, and per-batch times.
- `summary.json`: dimensions, final metrics, metadata, timing profile, artifact
  names, and model-weight manifest.
- `model_weights.npz`: learned MLP weights and biases, plus inverse parameters
  when inverse mode is used.
- `predictions.csv`: one row per sample with parameter values, true variables,
  raw network predictions, projected predictions, and train/validation split.

Builder-defined general runs also save:

- `problem.json`: symbolic problem metadata.
- `parameters.csv`: the parameter matrix used for training.
- `variables.csv`: the variable labels used for training.

Inverse builder runs additionally save `inverse_comparison.json`.

## More Documentation

See the `docs` folder for focused references:

- [docs/INSTALL.md](docs/INSTALL.md): detailed install notes.
- [docs/PROBLEM.md](docs/PROBLEM.md): general-problem builder and block-style
  problem definitions.
- [docs/VERSION.md](docs/VERSION.md): version history and project notes.

⚠️ Please cite our work if you use this code in your research.
Citation formats are provided below.

arXiv Preprint: https://arxiv.org/pdf/2507.08124
Journal: https://doi.org/10.1016/j.compchemeng.2025.109418

```bibtex
@article{iftakher2025physics,
  title={Physics-informed neural networks with hard nonlinear equality and inequality constraints},
  author={Iftakher, Ashfaq and Golder, Rahul and Roy, Bimol Nath and Hasan, MM Faruque},
  journal={Computers \& Chemical Engineering},
  pages={109418},
  year={2025},
  publisher={Elsevier}
}
