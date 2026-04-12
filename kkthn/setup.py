from __future__ import annotations

import sys
import warnings
from pathlib import Path

from setuptools import setup

package_root = Path(__file__).resolve().parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from install_selector import read_requirements, requirements_file, selection_message


selected_file, _selected_mode, _selected_reason = requirements_file(package_root)
message = selection_message(package_root)
print(message, flush=True)
sys.stderr.write(message + "\n")
warnings.warn(message, RuntimeWarning)

setup(install_requires=read_requirements(selected_file))
