import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5 import QtWidgets
from scipy.spatial.transform import Rotation as R

from scan_plan.gui import RegistrationDialog, error_level
from tests.test_gui_machine_coords import _MainWindow, qapp  # noqa: F401

OK, WARN, BAD = "#cdeccd", "#fab219", "#d03b3b"


class _MainWithCylinders(_MainWindow):
    def __init__(self, cylinders, z_step=180, motor_limits=None):
        super().__init__()
        self._cyl = np.asarray(cylinders, dtype=float)
        self.cfg["prescan_z_step"] = z_step
        if motor_limits:
            self.cfg["motor_limits"] = motor_limits

    def get_all_active_points(self):
        return self._cyl


def _dialog(main, z_step=180, nudge=None):
    """Registration on 6 exact pairs (180 nm prescan, 100 nm refscan, 35° about z).

    `nudge` = (row, prescan offset in pixels) mis-picks one prescan point.
    """
    dlg = RegistrationDialog(main)
    dlg.in_px.setValue(100.0)
    dlg.in_final_px.setValue(50.0)
    prescan = np.random.default_rng(3).uniform(200, 1800, size=(6, 3))
    refscan = R.from_euler("z", 35, degrees=True).apply(prescan * [1.8, 1.8, z_step / 100]) + [300, -150, 80]
    if nudge is not None:
        prescan = prescan.copy()
        prescan[nudge[0]] += nudge[1]
    dlg.table_pre.setRowCount(0)
    dlg.table_ref.setRowCount(0)
    for p, r in zip(prescan, refscan):
        dlg.add_row()
        row = dlg.table_pre.rowCount() - 1
        for c in range(3):
            dlg.table_pre.setItem(row, c, QtWidgets.QTableWidgetItem(repr(float(p[c]))))
            dlg.table_ref.setItem(row, c, QtWidgets.QTableWidgetItem(repr(float(r[c]))))
    dlg.calculate_registration()
    assert dlg.vreg_svd is not None, dlg.lbl_result.text()
    return dlg


def _bg(item):
    return item.background().color().name()


def _matrix_colours(dlg):
    n = dlg.table_matrix.rowCount()
    return {(i, j): _bg(dlg.table_matrix.item(i, j)) for i in range(n) for j in range(n) if i != j}


def _error_items(dlg):
    return [dlg.table_results.item(i, 3) for i in range(dlg.table_results.rowCount())]


@pytest.mark.parametrize("err_vx, level", [(0.0, "ok"), (1.0, "ok"), (1.01, "warn"), (2.0, "warn"), (2.01, "bad")])
def test_error_levels(err_vx, level):
    assert error_level(err_vx) == level


def test_consistent_pairs_show_no_warnings(qapp):
    dlg = _dialog(_MainWithCylinders([[500, 500, 500]]))
    assert set(_matrix_colours(dlg).values()) == {OK}
    assert {_bg(it) for it in _error_items(dlg)} == {OK}
    assert "agree" in dlg.lbl_matrix_hint.text()


def test_anisotropic_prescan_is_consistent(qapp):
    dlg = _dialog(_MainWithCylinders([[500, 500, 500]], z_step=360), z_step=360)
    assert set(_matrix_colours(dlg).values()) == {OK}
    assert {_bg(it) for it in _error_items(dlg)} == {OK}


def test_mis_picked_point_is_named(qapp):
    # 10 prescan pixels off in x: 10 vx at 180 nm.
    dlg = _dialog(_MainWithCylinders([[500, 500, 500]]), nudge=(2, [10, 0, 0]))
    # A distance only sees the nudge along the pair, so count red cells per row.
    colours = _matrix_colours(dlg)
    red_per_row = [sum(colours[(i, j)] == BAD for j in range(6) if j != i) for i in range(6)]
    assert int(np.argmax(red_per_row)) == 2 and red_per_row[2] >= 3
    assert "Point 2 disagrees" in dlg.lbl_matrix_hint.text()
    errors = [float(it.text().split("·")[1].split("vx")[0]) for it in _error_items(dlg)]
    assert int(np.argmax(errors)) == 2
    assert _bg(_error_items(dlg)[2]) == BAD
    assert "!!" in _error_items(dlg)[2].text()


def test_error_cell_gives_um_and_vx(qapp):
    dlg = _dialog(_MainWithCylinders([[500, 500, 500]]))
    text = _error_items(dlg)[0].text()
    assert "µm" in text and "vx" in text
    assert "180" in dlg.lbl_err_legend.text()          # 1 vx = 180 nm, the prescan voxel


def test_cylinder_motors_have_their_own_table(qapp):
    cyl = [[400, 900, 300], [1500, 200, 1100]]
    main = _MainWithCylinders(cyl, motor_limits={"su": [-100, 100], "sv": [-100, 100], "sz": [-100, 100]})
    dlg = _dialog(main)
    assert dlg.table_results.columnCount() == 4
    assert dlg.table_motors.rowCount() == 2
    su, sv, sz = dlg.vreg_svd.refscan_to_motors(dlg.vreg_svd.transformToRefscan(np.array(cyl, float)), 50.0)
    assert float(dlg.table_motors.item(1, 3).text()) == pytest.approx(sz[1], abs=1e-4)


def test_out_of_range_motor_cell_is_red(qapp):
    main = _MainWithCylinders([[400, 900, 300]], motor_limits={"su": [-100, 100], "sv": [-100, 100], "sz": [50, 60]})
    dlg = _dialog(main)
    assert _bg(dlg.table_motors.item(0, 3)) == BAD
    assert dlg.table_motors.item(0, 1).background().style() == 0          # in range: no fill
