import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5 import QtWidgets

from scan_plan.cli import _resolve_optics
from scan_plan.gui import CylinderApp
from scan_plan.io import load_config, load_presets
from scan_plan.solver import calculate_contrast_limits

EXAMPLE_CONFIG = os.path.join(os.path.dirname(__file__), "..", "scan_plan_config.example.json")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def empty_app(qapp, tmp_path, monkeypatch):
    """Main window with no volume and no cylinders."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    cfg_path = tmp_path / "scan_plan_config.json"
    shutil.copy(EXAMPLE_CONFIG, cfg_path)
    cfg = load_config(str(cfg_path))
    _resolve_optics(cfg, load_presets())
    w = CylinderApp(cfg, None, calculate_contrast_limits(None), config_path=str(cfg_path))
    yield w
    w.close()


def test_register_opens_without_cylinders(empty_app):
    # Checking existing scan positions starts from an empty plan.
    assert len(empty_app.get_all_active_points()) == 0
    assert empty_app.btn_register.isEnabled()


def test_exports_need_a_cylinder(empty_app):
    assert not empty_app.btn_export_nml.isEnabled()
    assert not empty_app.btn_print_console.isEnabled()
