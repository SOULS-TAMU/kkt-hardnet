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
# Clone the repository
git clone <repo-url>

# Set the Repository
cd KKTHardNet

# Create the virtual environment or use one if you have available.

py -3.9 -m venv env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate the environment
.\env\Scripts\Activate.ps1

# Upgrade pip version
python -m pip install --upgrade pip

# Check your hardware configuration (cpu/gpu)
python kkthn\install_info.py

# $env:KKTHN_REQUIREMENTS = "gpu" (If GPU is available)

# Install the KKT-HardNet Package
python -m pip install -e kkthn

# Test if installation is working
python -c "import kkthn; from kkthn.builder import ProblemBuilder; print(kkthn.__version__)"

# Smoke run
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
# Clone the repository
git clone <repo-url>

# Set working directory
cd KKTHardNet

# Create virtual environment or use one if available
python3 -m venv env

# Activate the virtual environment
source env/bin/activate

# Install or upgrade pip
python -m pip install --upgrade pip

# Check your hardware (cpu/gpu)
python kkthn/install_info.py

# export KKTHN_REQUIREMENTS=gpu (If you have gpu support)

# Install the KKT-HardNet package
python -m pip install -e kkthn

# Test for installed package
python -c "import kkthn; from kkthn.builder import ProblemBuilder; print(kkthn.__version__)"

# Smoke Run
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

## Verify Package Location

```sh
python -c "import kkthn, pathlib; from kkthn.builder import ProblemBuilder; print(kkthn.__version__); print(pathlib.Path(kkthn.__file__).resolve())"
```

The printed path should point inside:

```text
KKTHardNet/kkthn/src/kkthn
```
