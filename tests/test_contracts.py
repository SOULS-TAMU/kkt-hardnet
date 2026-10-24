"""Installed exports and frontend/shared ownership contracts."""

import importlib
from importlib.metadata import version
from pathlib import Path
import json
import pytest


def test_installed_distribution_and_exports():
    frontend = importlib.import_module("kkthn")
    assert frontend.__version__ == version("kkt-hardnet")
    for name in frontend.__all__:
        assert hasattr(frontend, name), name


def test_frozen_api_surface():
    from api_contract import surface, compare

    root = Path(__file__).resolve().parents[1]
    assert not compare(
        surface(root), json.loads((root / ".github/validation/api.json").read_text())
    )


def test_unknown_model_rejected(tmp_path):
    from kkthn import KKTHardNet as Facade

    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"format": "legacy"}))
    with pytest.raises((ValueError, KeyError)):
        Facade().load(path)
