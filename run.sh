Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
D:\Projects\virtual_envs\env\Scripts\Activate.ps1
python kkthn\install_info.py
pip install -e kkthn
python -c "import kkthn; print(kkthn.__version__)"
