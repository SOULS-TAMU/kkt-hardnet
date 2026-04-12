# KKTHardNet Installation

These steps assume you cloned KKTHardNet from GitHub and want to install the
local `kkthn` package in editable mode.

Replace `<repo-url>` with the GitHub URL for this repository.

## Requirements

- Python 3.9 or newer.
- Git.
- For CPU usage: no CUDA setup is required.
- For GPU usage: use Linux or WSL with a working NVIDIA CUDA setup. Native
  Windows installs CPU dependencies by default because JAX CUDA plugin wheels
  are not available for native Windows.

## Windows PowerShell

```powershell
git clone <repo-url>
cd KKTHardNet

py -3.9 -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python kkthn\install_info.py
python -m pip install -e kkthn

python -c "import kkthn; print(kkthn.__version__)"
python main.py --type qp --action data --p 2 --n 4 --me 1 --mi 1 --samples 2
```

If `py -3.9` is not available, use:

```powershell
python -m venv env
```

If PowerShell blocks activation, run this in the same PowerShell session and
activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## macOS Or Linux

```sh
git clone <repo-url>
cd KKTHardNet

python3 -m venv env
. env/bin/activate

python -m pip install --upgrade pip
python kkthn/install_info.py
python -m pip install -e kkthn

python -c "import kkthn; print(kkthn.__version__)"
python main.py --type qp --action data --p 2 --n 4 --me 1 --mi 1 --samples 2
```

On some Linux systems, use `python` instead of `python3` if that is your Python
3 command.

## CPU And GPU Install Selection

During `python -m pip install -e kkthn`, the installer chooses one dependency
file:

- CPU: `kkthn/requirements.txt`
- GPU: `kkthn/requirements.gpu.txt`

Automatic behavior:

- Native Windows: CPU dependencies.
- macOS: CPU dependencies in normal setups.
- Linux or WSL with detected CUDA: GPU dependencies.
- Linux or WSL without detected CUDA: CPU dependencies.

The selected mode is printed by:

```sh
python kkthn/install_info.py
```

Force CPU:

```sh
export KKTHN_REQUIREMENTS=cpu
python -m pip install -e kkthn
```

Windows PowerShell:

```powershell
$env:KKTHN_REQUIREMENTS = "cpu"
python -m pip install -e kkthn
```

Force GPU on Linux or WSL:

```sh
nvidia-smi
export KKTHN_REQUIREMENTS=gpu
python -m pip install -e kkthn
```

Do not force GPU on native Windows unless you are intentionally testing a custom
JAX setup.

## Run Examples

Generate labels only:

```sh
python main.py --type qp --action data --p 2 --n 4 --me 1 --mi 1 --samples 2
```

Generate noisy labels:

```sh
python main.py --type qp --action data --p 2 --n 4 --me 1 --mi 1 --samples 2 --noise_scale 0.1
```

Run a small training job:

```sh
python main.py --type qp --action run --p 2 --n 4 --me 1 --mi 1 --samples 8 --epochs 1 --batch_size 2
```

Run the builder-defined general example:

```sh
python run_general.py --mode forward --samples 8 --epochs 1 --batch_size 2
```

For a builder-defined inverse problem, edit `build_problem()`,
`DATA["inv_param"]`, `DATA["inv_param_label"]`, and optionally
`DATA["inv_param_init"]` in `run_general.py`, then run:

```sh
python run_general.py --mode inverse
```

## Verify Package Location

```sh
python -c "import kkthn, pathlib; print(kkthn.__version__); print(pathlib.Path(kkthn.__file__).resolve())"
```

The printed path should point inside:

```text
KKTHardNet/kkthn/src/kkthn
```
