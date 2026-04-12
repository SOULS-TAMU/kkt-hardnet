# KKTHardNet Installation Guide

This guide installs `kkthn` as an editable Python package from the local
KKTHardNet checkout.

## 1. Choose The Install Location

Use the project root. In WSL this is usually:

```sh
cd /mnt/d/Projects/KKTHardNet
```

In Git Bash or MSYS2 this is usually:

```sh
cd /d/Projects/KKTHardNet
```

On native Windows PowerShell:

```powershell
cd D:\Projects\KKTHardNet
```

## 2. Create A Virtual Environment

Recommended environment location:

WSL:

```sh
python -m venv /mnt/d/Projects/virtual_envs/env
```

Git Bash or MSYS2:

```sh
python -m venv /d/Projects/virtual_envs/env
```

On native Windows PowerShell:

```powershell
py -3.9 -m venv D:\Projects\virtual_envs\env
```

If `py -3.9` is not available, use the full Python path, for example:

```powershell
C:\Users\DELL\AppData\Local\Programs\Python\Python39\python.exe -m venv D:\Projects\virtual_envs\env
```

## 3. Activate The Virtual Environment

WSL:

```sh
source /mnt/d/Projects/virtual_envs/env/bin/activate
```

Git Bash or MSYS2:

```sh
source /d/Projects/virtual_envs/env/bin/activate
```

Native Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Projects\virtual_envs\env\Scripts\Activate.ps1
```

Native Windows Command Prompt:

```bat
D:\Projects\virtual_envs\env\Scripts\activate.bat
```

## 4. Upgrade Pip

```sh
python -m pip install --upgrade pip
```

## 5. Install `kkthn`

From the KKTHardNet project root:

```sh
python kkthn/install_info.py
python -m pip install -e kkthn
```

The first command prints which dependency set will be used. The editable install
then installs `kkthn` from:

```text
KKTHardNet/kkthn
```

## CPU And GPU Dependency Selection

`kkthn` selects dependencies at install time:

- CPU dependencies: `kkthn/requirements.txt`
- GPU dependencies: `kkthn/requirements.gpu.txt`

Automatic behavior:

- Native Windows installs CPU dependencies by default.
- Linux or WSL installs GPU dependencies when CUDA is detected through
  `CUDA_HOME`, `CUDA_PATH`, `nvidia-smi`, or `nvcc`.
- Linux or WSL installs CPU dependencies when CUDA is not detected.
- Setting `CUDA_VISIBLE_DEVICES=-1` forces CPU mode for automatic detection.

Native Windows note: GPU mode is not selected automatically because JAX CUDA
plugin wheels are not available for native Windows. For CUDA acceleration, use
Linux or WSL with a working NVIDIA CUDA setup.

## Force CPU Install

Linux, WSL, Git Bash, or MSYS2:

```sh
export KKTHN_REQUIREMENTS=cpu
python kkthn/install_info.py
python -m pip install -e kkthn
```

Native Windows PowerShell:

```powershell
$env:KKTHN_REQUIREMENTS = "cpu"
python kkthn\install_info.py
python -m pip install -e kkthn
```

## Force GPU Install

Use this on Linux or WSL after confirming CUDA is available.

```sh
nvidia-smi
export KKTHN_REQUIREMENTS=gpu
python kkthn/install_info.py
python -m pip install -e kkthn
```

Forcing GPU mode on native Windows may fail because the required JAX CUDA plugin
wheels are not published for that platform.

## Complete WSL Install Script

Save this as `install_kkthn.sh` or run the commands directly:

```sh
#!/usr/bin/env sh
set -eu

cd /mnt/d/Projects/KKTHardNet

python -m venv /mnt/d/Projects/virtual_envs/env
. /mnt/d/Projects/virtual_envs/env/bin/activate

python -m pip install --upgrade pip
python kkthn/install_info.py
python -m pip install -e kkthn

python -c "import kkthn; print(kkthn.__version__)"
```

Force CPU in the same script:

```sh
export KKTHN_REQUIREMENTS=cpu
```

Force GPU in the same script:

```sh
export KKTHN_REQUIREMENTS=gpu
```

Place the export line before `python kkthn/install_info.py`.

## Complete Git Bash Or MSYS2 Install Script

```sh
#!/usr/bin/env sh
set -eu

cd /d/Projects/KKTHardNet

python -m venv /d/Projects/virtual_envs/env
. /d/Projects/virtual_envs/env/bin/activate

python -m pip install --upgrade pip
python kkthn/install_info.py
python -m pip install -e kkthn

python -c "import kkthn; print(kkthn.__version__)"
```

## Complete Native Windows PowerShell Install

```powershell
cd D:\Projects\KKTHardNet

py -3.9 -m venv D:\Projects\virtual_envs\env

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Projects\virtual_envs\env\Scripts\Activate.ps1

python -m pip install --upgrade pip
python kkthn\install_info.py
python -m pip install -e kkthn

python -c "import kkthn; print(kkthn.__version__)"
```

## Verify The Install

```sh
python -c "import kkthn, pathlib; print(kkthn.__version__); print(pathlib.Path(kkthn.__file__).resolve())"
```

Expected result:

- The version prints, for example `0.1.4`.
- The package path points inside `D:\Projects\KKTHardNet\kkthn\src\kkthn`
  or the matching path in your Linux/WSL checkout.

## Run A Small Example

```sh
python main.py --type qp --action data --p 2 --n 4 --me 1 --mi 1 --samples 2
```

For the full training pipeline:

```sh
python main.py --type qp --action run
```

## Troubleshooting

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
D:\Projects\virtual_envs\env\Scripts\Activate.ps1
```

If `pip install -e kkthn` still points to an older KKTHardNet checkout, reinstall
from the intended project root:

```sh
cd /mnt/d/Projects/KKTHardNet
python -m pip uninstall -y kkthn
python kkthn/install_info.py
python -m pip install -e kkthn
```

If GPU installation fails on native Windows, use the CPU install there or install
from WSL/Linux with CUDA available.
