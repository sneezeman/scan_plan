import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
from PyQt5 import QtWidgets
from scipy.spatial.transform import Rotation as R

from scan_plan.gui import RegistrationDialog


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _MainWindow(QtWidgets.QWidget):
    """The part of CylinderApp that RegistrationDialog uses, without the 3D view."""

    def __init__(self):
        super().__init__()
        self.cfg = {"prescan_pixel_size_xy": 180, "optics": {}}
        self.manual_points = []
        self.active_manual_mask = []
        self.chk_show_manual = QtWidgets.QCheckBox()
        self.spin_res = QtWidgets.QDoubleSpinBox()
        self.spin_res.setRange(0.01, 100000.0)
        self.spin_res.setValue(50.0)
        self.current_scan_res = 50.0
        self.scene_updates = 0

    def update_resolution(self):
        self.current_scan_res = float(self.spin_res.value())

    def get_all_active_points(self):
        return np.empty((0, 3))

    def refresh_cyl_list(self):
        pass

    def update_3d_scene(self):
        self.scene_updates += 1

    def _persist_session_pref(self, key, value):
        pass


def _registered_dialog(main):
    """Dialog with a computed registration (35° z-rotation, shift, 180→100 nm)."""
    dlg = RegistrationDialog(main)
    dlg.in_px.setValue(100.0)
    dlg.in_final_px.setValue(50.0)
    prescan = np.random.default_rng(3).uniform(200, 1800, size=(6, 3))
    refscan = R.from_euler("xyz", [0, 0, 35], degrees=True).apply(prescan * 1.8) + [300, -150, 80]
    dlg.table_pre.setRowCount(0)
    dlg.table_ref.setRowCount(0)
    for p, r in zip(prescan, refscan):
        dlg.add_row()
        row = dlg.table_pre.rowCount() - 1
        for c in range(3):
            dlg.table_pre.setItem(row, c, QtWidgets.QTableWidgetItem(str(p[c])))
            dlg.table_ref.setItem(row, c, QtWidgets.QTableWidgetItem(str(r[c])))
    dlg.calculate_registration()
    assert dlg.vreg_svd is not None, dlg.lbl_result.text()
    return dlg


def test_saved_machine_coords_land_on_planned_points(qapp, tmp_path):
    main = _MainWindow()
    main.chk_show_manual.setChecked(False)
    dlg = _registered_dialog(main)

    # Write motor positions exactly as SAVE MACHINE COORDINATES does.
    planned = np.array([[400.0, 900.0, 300.0], [1500.0, 200.0, 1100.0]])
    su, sv, sz = dlg.vreg_svd.refscan_to_motors(dlg.vreg_svd.transformToRefscan(planned), 50.0)
    path = tmp_path / "tiles_motor_coords.txt"
    pd.DataFrame(np.array([su, sv, sz]).T, columns=["#su", "sv", "sz"]).to_csv(
        path, sep=" ", index=False, float_format="%.04f"
    )

    added = dlg.show_machine_coords(str(path))

    # 4-decimal mm rounding moves a point by at most ~0.3 prescan pixels.
    assert len(added) == 2
    np.testing.assert_allclose(np.array(main.manual_points), planned, atol=0.5)
    assert main.active_manual_mask == [True, True]
    assert main.chk_show_manual.isChecked()
    assert main.scene_updates >= 1


PLANNED = np.array([[400.0, 900.0, 300.0], [1500.0, 200.0, 1100.0], [900.0, 1200.0, 700.0]])


def _write_beamline_file(path, vreg, pixel_sizes):
    """Write rows as '#su sv sz pixel_size tomo_n', motors computed at each row's pixel size."""
    lines = ["#su sv sz pixel_size tomo_n"]
    for p, px in zip(PLANNED, pixel_sizes):
        su, sv, sz = vreg.refscan_to_motors(vreg.transformToRefscan(p[None, :]), px)
        lines.append(f"{su[0]:.4f} {sv[0]:.4f} {sz[0]:.4f} {px:g} 3000")
    path.write_text("\n".join(lines) + "\n")


def test_pixel_size_column_drives_conversion_and_resolution(qapp, tmp_path):
    main = _MainWindow()
    dlg = _registered_dialog(main)  # Final Pixel Size and main window at 50 nm
    path = tmp_path / "scans.txt"
    _write_beamline_file(path, dlg.vreg_svd, [100.0, 100.0, 100.0])

    added = dlg.show_machine_coords(str(path))

    np.testing.assert_allclose(np.array(main.manual_points), PLANNED, atol=0.5)
    np.testing.assert_array_equal(added, [100.0, 100.0, 100.0])
    assert main.current_scan_res == 100.0
    assert dlg.in_final_px.value() == 100.0


def test_mixed_pixel_sizes_convert_per_row_and_keep_resolution(qapp, tmp_path):
    main = _MainWindow()
    dlg = _registered_dialog(main)
    path = tmp_path / "scans.txt"
    _write_beamline_file(path, dlg.vreg_svd, [100.0, 50.0, 30.0])

    dlg.show_machine_coords(str(path))

    np.testing.assert_allclose(np.array(main.manual_points), PLANNED, atol=0.5)
    assert main.current_scan_res == 50.0
