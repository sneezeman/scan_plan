"""
GUI classes: CylinderApp (main window) and RegistrationDialog.
"""

import os
import numpy as np
import pyvista as pv
import pandas as pd
from scipy.spatial.distance import cdist

from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor

from scan_plan.volume_registration import VolumeRegistration
from scan_plan.nml_exporter import generate_nml
from scan_plan.io import parse_nml
from scan_plan.solver import solve_global_union


class RegistrationDialog(QtWidgets.QDialog):
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.vreg_svd = None
        self.vreg_opt = None
        self.res_svd = None
        self.res_opt = None

        self.setWindowTitle("Coordinates Registration")
        self.resize(1100, 850)

        self.tabs = QtWidgets.QTabWidget()
        self.tab_input = QtWidgets.QWidget()
        self.tab_result = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_input, "Input Data")
        self.tabs.addTab(self.tab_result, "Optimization Results")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.tabs)

        self._setup_input_tab()
        self._setup_result_tab()
        for _ in range(4):
            self.add_row()

    def _setup_input_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_input)

        gb_conf = QtWidgets.QGroupBox("Machine Reference (Refscan 0)")
        fl_conf = QtWidgets.QFormLayout()
        self.in_su = QtWidgets.QLineEdit("-0.5595")
        self.in_sv = QtWidgets.QLineEdit("0.06712")
        self.in_sz = QtWidgets.QLineEdit("1.455")
        self.in_px = QtWidgets.QLineEdit("180")
        self.in_final_px = QtWidgets.QLineEdit("100")
        fl_conf.addRow("su (mm):", self.in_su)
        fl_conf.addRow("sv (mm):", self.in_sv)
        fl_conf.addRow("sz (mm):", self.in_sz)
        fl_conf.addRow("Refscan Pixel Size (nm):", self.in_px)
        fl_conf.addRow("Final Pixel Size (nm):", self.in_final_px)
        gb_conf.setLayout(fl_conf)
        layout.addWidget(gb_conf)

        gb_pts = QtWidgets.QGroupBox("Matching Points")
        l_pts_main = QtWidgets.QVBoxLayout()

        h_mode = QtWidgets.QHBoxLayout()
        h_mode.addWidget(QtWidgets.QLabel("Right-table mode:"))
        self.combo_match_mode = QtWidgets.QComboBox()
        self.combo_match_mode.addItems(["Refscan Pixels", "Motor Coordinates (su/sv/sz)"])
        self.combo_match_mode.currentIndexChanged.connect(self._on_match_mode_changed)
        h_mode.addWidget(self.combo_match_mode)
        l_pts_main.addLayout(h_mode)

        h_tables = QtWidgets.QHBoxLayout()

        v_pre = QtWidgets.QVBoxLayout()
        lbl_pre = QtWidgets.QLabel("<b>PRESCAN</b> — pre-beamtime overview volume (pixels)")
        lbl_pre.setStyleSheet("background-color: #cce5ff; padding: 4px; border: 2px solid #004085; border-radius: 3px; color: #004085;")
        v_pre.addWidget(lbl_pre)
        self.table_pre = QtWidgets.QTableWidget(0, 3)
        self.table_pre.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.table_pre.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_pre.setAlternatingRowColors(True)
        self.table_pre.setStyleSheet("QTableWidget { border: 2px solid #004085; }")
        v_pre.addWidget(self.table_pre)
        h_tables.addLayout(v_pre)

        v_ref = QtWidgets.QVBoxLayout()
        self.lbl_ref_table = QtWidgets.QLabel("<b>REFSCAN</b> — ID16A reference tomogram (pixels)")
        self.lbl_ref_table.setStyleSheet("background-color: #d4edda; padding: 4px; border: 2px solid #155724; border-radius: 3px; color: #155724;")
        v_ref.addWidget(self.lbl_ref_table)
        self.table_ref = QtWidgets.QTableWidget(0, 3)
        self.table_ref.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.table_ref.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_ref.setAlternatingRowColors(True)
        self.table_ref.setStyleSheet("QTableWidget { border: 2px solid #155724; }")
        v_ref.addWidget(self.table_ref)
        h_tables.addLayout(v_ref)
        l_pts_main.addLayout(h_tables)

        h_btn = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add Row")
        btn_add.clicked.connect(self.add_row)
        btn_del = QtWidgets.QPushButton("Remove Row")
        btn_del.clicked.connect(self.remove_row)
        btn_paste = QtWidgets.QPushButton("Paste Clipboard")
        btn_paste.clicked.connect(self.paste_from_clipboard)
        btn_paste.setStyleSheet("background-color: #f0ad4e; color: white;")
        btn_load = QtWidgets.QPushButton("Load File")
        btn_load.clicked.connect(self.load_match_points_from_file)
        btn_load.setStyleSheet("background-color: #5cb85c; color: white;")
        h_btn.addWidget(btn_add); h_btn.addWidget(btn_del); h_btn.addWidget(btn_paste); h_btn.addWidget(btn_load)
        l_pts_main.addLayout(h_btn)

        self.chk_z_only = QtWidgets.QCheckBox("Restrict Rotation to Z-Axis only")
        self.chk_z_only.setChecked(True)
        l_pts_main.addWidget(self.chk_z_only)
        gb_pts.setLayout(l_pts_main)
        layout.addWidget(gb_pts)

        gb_mat = QtWidgets.QGroupBox("Consistency Matrix")
        l_mat = QtWidgets.QVBoxLayout()
        self.table_matrix = QtWidgets.QTableWidget()
        self.table_matrix.setAlternatingRowColors(False)
        self.table_matrix.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_matrix.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        l_mat.addWidget(self.table_matrix)
        l_mat.addWidget(QtWidgets.QLabel("Upper Right: Prescan Dist | Lower Left: Refscan Dist | Heatmap: Green(0) -> Red(Max)"))
        gb_mat.setLayout(l_mat)
        layout.addWidget(gb_mat)

        btn_calc = QtWidgets.QPushButton("Calculate & Verify Models")
        btn_calc.setStyleSheet("background-color: #5bc0de; color: white; font-weight: bold; padding: 8px;")
        btn_calc.clicked.connect(self.calculate_registration)
        layout.addWidget(btn_calc)

        self.lbl_result = QtWidgets.QLabel("Status: Waiting for input...")
        self.lbl_result.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(self.lbl_result)

    def _setup_result_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_result)

        gb_model = QtWidgets.QGroupBox("Active Transformation Model")
        l_mod = QtWidgets.QHBoxLayout()
        self.combo_result_select = QtWidgets.QComboBox()
        self.combo_result_select.currentIndexChanged.connect(self.update_results_ui)
        l_mod.addWidget(QtWidgets.QLabel("Select Model to Apply:"))
        l_mod.addWidget(self.combo_result_select)
        gb_model.setLayout(l_mod)
        layout.addWidget(gb_model)

        gb_res = QtWidgets.QGroupBox("Solution Parameters")
        form = QtWidgets.QFormLayout()
        self.lbl_rot = QtWidgets.QLabel("[N/A]")
        self.lbl_cost = QtWidgets.QLabel("[N/A]")
        self.lbl_msg = QtWidgets.QLabel("[N/A]")
        form.addRow("Rotation Angles (Yaw, Pitch, Roll):", self.lbl_rot)
        form.addRow("Final Cost (Residual):", self.lbl_cost)
        form.addRow("Optimizer Status:", self.lbl_msg)
        gb_res.setLayout(form)
        layout.addWidget(gb_res)

        gb_det = QtWidgets.QGroupBox("Per-Point Analysis")
        l_det = QtWidgets.QVBoxLayout()
        self.table_results = QtWidgets.QTableWidget(0, 4)
        self.table_results.setHorizontalHeaderLabels(["ID", "Refscan (Pixels)", "Transformed", "Error (µm)"])
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        l_det.addWidget(self.table_results)
        gb_det.setLayout(l_det)
        layout.addWidget(gb_det)

        layout.addWidget(QtWidgets.QLabel("Raw Output:"))
        self.txt_raw = QtWidgets.QTextEdit()
        self.txt_raw.setReadOnly(True)
        layout.addWidget(self.txt_raw)

        btn_save = QtWidgets.QPushButton("SAVE MACHINE COORDINATES (.txt)")
        btn_save.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self.save_machine_file)
        layout.addWidget(btn_save)

    def add_row(self):
        r = self.table_pre.rowCount()
        self.table_pre.insertRow(r)
        self.table_ref.insertRow(r)
        self.table_pre.setVerticalHeaderItem(r, QtWidgets.QTableWidgetItem(str(r)))
        self.table_ref.setVerticalHeaderItem(r, QtWidgets.QTableWidgetItem(str(r)))
        for i in range(3):
            self.table_pre.setItem(r, i, QtWidgets.QTableWidgetItem("0"))
            self.table_ref.setItem(r, i, QtWidgets.QTableWidgetItem("0"))

    def remove_row(self):
        r = self.table_pre.currentRow()
        if r == -1: r = self.table_ref.currentRow()
        if r >= 0:
            self.table_pre.removeRow(r)
            self.table_ref.removeRow(r)
            for i in range(self.table_pre.rowCount()):
                self.table_pre.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i)))
                self.table_ref.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem(str(i)))

    def paste_from_clipboard(self):
        clipboard = QtWidgets.QApplication.clipboard()
        text = clipboard.text()
        if not text: return
        lines = text.strip().split('\n')
        self.table_pre.setRowCount(0)
        self.table_ref.setRowCount(0)
        for line in lines:
            parts = line.replace('\t', ' ').replace(',', ' ').split()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 6:
                self.add_row()
                cur = self.table_pre.rowCount() - 1
                for c in range(3):
                    self.table_pre.setItem(cur, c, QtWidgets.QTableWidgetItem(parts[c]))
                    self.table_ref.setItem(cur, c, QtWidgets.QTableWidgetItem(parts[c+3]))
        self.lbl_result.setText(f"Status: Pasted {len(lines)} rows.")

    def load_match_points_from_file(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Match Points", "", "Text Files (*.txt)", options=options)
        if not fileName:
            return
        try:
            with open(fileName, "r") as f:
                lines = f.readlines()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not read file: {e}")
            return

        self.table_pre.setRowCount(0)
        self.table_ref.setRowCount(0)
        count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace('\t', ' ').replace(',', ' ').split()
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 6:
                self.add_row()
                cur = self.table_pre.rowCount() - 1
                for c in range(3):
                    self.table_pre.setItem(cur, c, QtWidgets.QTableWidgetItem(parts[c]))
                    self.table_ref.setItem(cur, c, QtWidgets.QTableWidgetItem(parts[c+3]))
                count += 1
        self.lbl_result.setText(f"Status: Loaded {count} rows from file.")

    def _on_match_mode_changed(self, index):
        if index == 1:  # Motor Coordinates
            self.lbl_ref_table.setText("<b>MOTOR COORDS</b> — ID16A stage positions (su/sv/sz in mm)")
            self.lbl_ref_table.setStyleSheet("background-color: #fff3cd; padding: 4px; border: 2px solid #856404; border-radius: 3px; color: #856404;")
            self.table_ref.setHorizontalHeaderLabels(["su (mm)", "sv (mm)", "sz (mm)"])
            self.table_ref.setStyleSheet("QTableWidget { border: 2px solid #856404; }")
        else:  # Refscan Pixels
            self.lbl_ref_table.setText("<b>REFSCAN</b> — ID16A reference tomogram (pixels)")
            self.lbl_ref_table.setStyleSheet("background-color: #d4edda; padding: 4px; border: 2px solid #155724; border-radius: 3px; color: #155724;")
            self.table_ref.setHorizontalHeaderLabels(["X", "Y", "Z"])
            self.table_ref.setStyleSheet("QTableWidget { border: 2px solid #155724; }")

    def get_points(self):
        prescan, refscan = [], []
        for r in range(self.table_pre.rowCount()):
            try:
                px = float(self.table_pre.item(r, 0).text())
                py = float(self.table_pre.item(r, 1).text())
                pz = float(self.table_pre.item(r, 2).text())
                rx = float(self.table_ref.item(r, 0).text())
                ry = float(self.table_ref.item(r, 1).text())
                rz = float(self.table_ref.item(r, 2).text())
                prescan.append((px, py, pz))
                refscan.append((rx, ry, rz))
            except (ValueError, AttributeError):
                continue
        return prescan, refscan

    def update_matrix(self, prescan_pts, refscan_pts, vreg):
        N = len(prescan_pts)
        self.table_matrix.setRowCount(N)
        self.table_matrix.setColumnCount(N)
        headers = [str(i) for i in range(N)]
        self.table_matrix.setHorizontalHeaderLabels(headers)
        self.table_matrix.setVerticalHeaderLabels(headers)
        pre_scaled = vreg._scale_prescan(np.array(prescan_pts))
        ref_np = np.array(refscan_pts)
        d_pre_mat = cdist(pre_scaled, pre_scaled)
        d_ref_mat = cdist(ref_np, ref_np)
        diffs = np.abs(d_pre_mat - d_ref_mat)
        max_diff = np.max(diffs)
        if max_diff == 0: max_diff = 1.0

        for i in range(N):
            for j in range(N):
                if i == j:
                    item = QtWidgets.QTableWidgetItem("-")
                    item.setBackground(QtGui.QColor(230, 230, 230))
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                else:
                    val = d_pre_mat[i,j] if j > i else d_ref_mat[i,j]
                    diff = diffs[i,j]
                    item = QtWidgets.QTableWidgetItem(f"{val:.1f}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    ratio = diff / max_diff
                    if ratio < 0.5:
                        local_r = ratio * 2
                        color = QtGui.QColor(int(local_r*255), 255, 100)
                    else:
                        local_r = (ratio - 0.5) * 2
                        color = QtGui.QColor(255, int((1-local_r)*255), 100)
                    item.setBackground(color)
                    item.setToolTip(f"Diff: {diff:.1f}")
                self.table_matrix.setItem(i, j, item)

    def calculate_registration(self):
        # Validate numeric fields before doing work
        for name, field in [("su", self.in_su), ("sv", self.in_sv), ("sz", self.in_sz), ("Pixel Size", self.in_px)]:
            try:
                float(field.text())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Input Error", f"'{name}' must be a valid number.")
                return
        try:
            pre_px = self.main_app.cfg["prescan_pixel_size_xy"]
            su = float(self.in_su.text())
            sv = float(self.in_sv.text())
            sz = float(self.in_sz.text())
            self.ref_px = float(self.in_px.text())

            pre_pts, ref_pts_raw = self.get_points()
            if len(pre_pts) < 3:
                self.lbl_result.setText("Status: Need at least 3 matching points.")
                return

            optics = self.main_app.cfg.get('optics', {})

            # If motor coordinate mode, convert su/sv/sz → refscan pixels
            if self.combo_match_mode.currentIndex() == 1:
                try:
                    final_px = float(self.in_final_px.text())
                except ValueError:
                    final_px = 100.0
                tmp_vreg = VolumeRegistration(pre_px, optics=optics)
                tmp_vreg.addReferenceVolume(su, sv, sz, self.ref_px)
                motor_arr = np.array(ref_pts_raw)
                refscan_arr = tmp_vreg.motors_to_refscan(
                    motor_arr[:, 0], motor_arr[:, 1], motor_arr[:, 2],
                    final_px
                )
                self.ref_pts = [tuple(row) for row in refscan_arr]
            else:
                self.ref_pts = ref_pts_raw

            self.vreg_svd = VolumeRegistration(pre_px, optics=optics)
            self.vreg_opt = VolumeRegistration(pre_px, optics=optics)
            self.vreg_svd.addReferenceVolume(su, sv, sz, self.ref_px)
            self.vreg_opt.addReferenceVolume(su, sv, sz, self.ref_px)

            for p, r in zip(pre_pts, self.ref_pts):
                self.vreg_svd.addMatchPoint(p, r, 0)
                self.vreg_opt.addMatchPoint(p, r, 0)

            self.update_matrix(pre_pts, self.ref_pts, self.vreg_svd)
            z_only = self.chk_z_only.isChecked()

            self.res_svd = self.vreg_svd.fitTransformationMatrix(rot_z_only=z_only, method='svd')
            self.res_opt = self.vreg_opt.fitTransformationMatrix(rot_z_only=z_only, method='optimizer')

            n_pts = len(pre_pts)
            err_svd = np.mean(self.res_svd.distances * (self.ref_px / 1000.0))
            err_opt = np.mean(self.res_opt.distances * (self.ref_px / 1000.0))
            max_svd = np.max(self.res_svd.distances * (self.ref_px / 1000.0))
            max_opt = np.max(self.res_opt.distances * (self.ref_px / 1000.0))

            self.combo_result_select.blockSignals(True)
            self.combo_result_select.clear()
            self.combo_result_select.addItem(f"SVD (Kabsch) - Avg: {err_svd:.2f} \u00b5m, Max: {max_svd:.2f} \u00b5m ({n_pts} pts)")
            self.combo_result_select.addItem(f"Optimizer - Avg: {err_opt:.2f} \u00b5m, Max: {max_opt:.2f} \u00b5m ({n_pts} pts)")
            self.combo_result_select.blockSignals(False)

            best_idx = 0 if err_svd <= err_opt else 1
            self.combo_result_select.setCurrentIndex(best_idx)

            self.lbl_result.setText("Status: Fit Complete. Check Results Tab.")
            self.update_results_ui()
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.lbl_result.setText(f"Status: Error - {str(e)}")

    def update_results_ui(self):
        idx = self.combo_result_select.currentIndex()
        if idx == -1: return

        active_res = self.res_svd if idx == 0 else self.res_opt

        err_um = active_res.distances * (self.ref_px / 1000.0)
        n_pts = len(self.ref_pts)
        angles = active_res.rotation_angles
        self.lbl_rot.setText(f"Yaw: {angles[0]:.3f}\u00b0,  Pitch: {angles[1]:.3f}\u00b0,  Roll: {angles[2]:.3f}\u00b0")
        mean_cost = active_res.solution.fun / n_pts if n_pts > 0 else active_res.solution.fun
        self.lbl_cost.setText(f"{mean_cost:.4f}  (mean per point, {n_pts} points)")
        self.lbl_msg.setText(active_res.solution.message)

        # Human-readable summary
        lines = []
        lines.append(f"Method: {active_res.solution.message}")
        lines.append(f"Points: {n_pts}")
        lines.append(f"Total residual: {active_res.solution.fun:.4f} px")
        lines.append(f"Mean residual:  {mean_cost:.4f} px")
        lines.append(f"Mean error:     {np.mean(err_um):.2f} \u00b5m")
        lines.append(f"Max error:      {np.max(err_um):.2f} \u00b5m")
        lines.append(f"Min error:      {np.min(err_um):.2f} \u00b5m")
        lines.append("")
        lines.append(f"Rotation: Yaw={angles[0]:.4f}\u00b0, Pitch={angles[1]:.4f}\u00b0, Roll={angles[2]:.4f}\u00b0")
        if hasattr(active_res.solution, 'nit'):
            lines.append(f"Iterations: {active_res.solution.nit}")
        if hasattr(active_res.solution, 'nfev'):
            lines.append(f"Function evaluations: {active_res.solution.nfev}")
        lines.append("")
        lines.append("Per-point errors (\u00b5m):")
        for i, e in enumerate(err_um):
            lines.append(f"  Point {i}: {e:.2f} \u00b5m")
        self.txt_raw.setText("\n".join(lines))

        self.table_results.setRowCount(len(self.ref_pts))
        for i in range(len(self.ref_pts)):
            self.table_results.setItem(i, 0, QtWidgets.QTableWidgetItem(str(i)))
            self.table_results.setItem(i, 1, QtWidgets.QTableWidgetItem(str(self.ref_pts[i])))
            trans = active_res.transformed_coords[i]
            self.table_results.setItem(i, 2, QtWidgets.QTableWidgetItem(f"[{trans[0]:.2f}, {trans[1]:.2f}, {trans[2]:.2f}]"))

            e_um = err_um[i]
            item_err = QtWidgets.QTableWidgetItem(f"{e_um:.2f} \u00b5m")
            if e_um < 5.0: item_err.setBackground(QtGui.QColor(150, 255, 150))
            elif e_um < 15.0: item_err.setBackground(QtGui.QColor(255, 255, 150))
            else: item_err.setBackground(QtGui.QColor(255, 150, 150))
            self.table_results.setItem(i, 3, item_err)

    def save_machine_file(self):
        idx = self.combo_result_select.currentIndex()
        if idx == -1 or not self.vreg_svd:
            QtWidgets.QMessageBox.warning(self, "Error", "Please calculate registration first.")
            return

        active_vreg = self.vreg_svd if idx == 0 else self.vreg_opt

        pts = self.main_app.get_all_active_points()
        if self.main_app.chk_flip_x.isChecked(): pts[:, 0] = self.main_app.max_dims[0] - pts[:, 0]
        if self.main_app.chk_flip_y.isChecked(): pts[:, 1] = self.main_app.max_dims[1] - pts[:, 1]
        if self.main_app.chk_flip_z.isChecked(): pts[:, 2] = self.main_app.max_dims[2] - pts[:, 2]

        if len(pts) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No active cylinders.")
            return

        try:
            XYZcoords_refscan = active_vreg.transformToRefscan(pts)
            try: final_px = float(self.in_final_px.text())
            except ValueError: final_px = 100.0

            su, sv, sz = active_vreg.refscan_to_motors(XYZcoords_refscan, final_px)
            df = pd.DataFrame(np.array([su, sv, sz]).T, columns=["#su", "sv", "sz"])

            options = QtWidgets.QFileDialog.Options()
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Machine Coordinates", "tiles_motor_coords.txt", "Text Files (*.txt)", options=options)

            if fileName:
                df.to_csv(fileName, sep=" ", index=False, float_format="%.04f")
                fiji_filename = os.path.splitext(fileName)[0] + "_fiji.txt"
                fiji_df = pd.DataFrame(pts, columns=["x", "y", "z"])
                fiji_df["D_std"] = int(self.main_app.dims_std[0])
                fiji_df["H_std"] = int(self.main_app.dims_std[1])
                fiji_df.to_csv(fiji_filename, sep=" ", index=False, float_format="%.04f")

                nml_filename = os.path.splitext(fileName)[0] + "_webknossos.nml"
                D_std, H_std = self.main_app.dims_std
                generate_nml(nml_filename, pts, D_std, H_std, color_hex="#00FFFF")

                mp_filename = os.path.splitext(fileName)[0] + "_match_pairs.txt"
                pre_pts, ref_pts = self.get_points()
                with open(mp_filename, "w") as f:
                    f.write("# Prescan_X Prescan_Y Prescan_Z Refscan_X Refscan_Y Refscan_Z\n")
                    for p, r in zip(pre_pts, ref_pts):
                        f.write(f"{p[0]} {p[1]} {p[2]} {r[0]} {r[1]} {r[2]}\n")

                QtWidgets.QMessageBox.information(self, "Success", f"Saved 4 files:\n{fileName}\n{fiji_filename}\n{nml_filename}\n{mp_filename}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {e}")


class CylinderApp(QtWidgets.QMainWindow):
    def __init__(self, config, vol_grid, vol_data, clim):
        super().__init__()
        self.cfg = config
        self.rois = [r.copy() for r in config.get('rois', [])]
        self.vol_grid = vol_grid
        self.vol_data_orig = vol_data
        self.clim = clim

        self.z_ratio = config['prescan_z_step'] / config['prescan_pixel_size_xy']
        self.max_dims = [0,0,0]
        if vol_grid is not None:
            self.max_dims = [
                vol_grid.dimensions[0] * config['binning'],
                vol_grid.dimensions[1] * config['binning'],
                vol_grid.dimensions[2] * config['binning']
            ]

        self.all_points = np.empty((0,3))
        self.manual_points = []

        self.dims_std = (10,10)
        self.dims_exp = (10,10)
        self.active_mask = []
        self.active_manual_mask = []
        self.total_roi_shift = [0, 0, 0]
        self.current_scan_res = config['scan_pixel_size']

        self.roi_actors = []
        self.actor_std = None
        self.actor_exp = None
        self.actor_man = None
        self.vol_actor = None
        self.actor_labels = None

        self.setWindowTitle("Scan Plan Planner")
        self.showMaximized()
        self._setup_ui()

        if self.vol_grid is None:
            QtWidgets.QMessageBox.warning(self, "Volume Load Warning",
                "No valid volume was loaded.\n\nCheck your JSON config or run with --debug to investigate.\n\n"
                "You can still define grids and manual points in empty space.")
        else:
            self.update_volume_render_mode()
            self.plotter.reset_camera()

        self.recalculate_points()
        self.update_opacity()

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(420)
        panel = QtWidgets.QWidget()
        scroll.setWidget(panel)
        layout = QtWidgets.QVBoxLayout(panel)

        layout.addWidget(self._create_appearance_group())
        layout.addWidget(self._create_roi_group())
        layout.addWidget(self._create_config_group())
        layout.addWidget(self._make_collapsible("Shift Bounding Boxes", self._create_roi_shift_content()))
        layout.addWidget(self._create_auto_grid_group())
        layout.addWidget(self._make_collapsible("Manual Cylinders", self._create_manual_content()))
        layout.addWidget(self._create_orient_group())
        layout.addWidget(self._create_actions_group())

        layout.addStretch()
        main_layout.addWidget(scroll)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self.plotter.show_grid()
        main_layout.addWidget(self.plotter)

    def _make_collapsible(self, title, content_widget, collapsed=True):
        """Return a widget with a togglable header and collapsible content."""
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        toggle_btn = QtWidgets.QToolButton()
        toggle_btn.setStyleSheet("QToolButton { font-weight: bold; border: none; }")
        toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        toggle_btn.setText(title)
        toggle_btn.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        toggle_btn.setCheckable(True)
        toggle_btn.setChecked(not collapsed)

        content_widget.setVisible(not collapsed)

        def on_toggle(checked):
            toggle_btn.setArrowType(QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
            content_widget.setVisible(checked)

        toggle_btn.toggled.connect(on_toggle)

        vbox.addWidget(toggle_btn)
        vbox.addWidget(content_widget)
        return container

    def _create_config_group(self):
        grp = QtWidgets.QGroupBox("Grid Settings")
        lo = QtWidgets.QVBoxLayout()
        lo.setSpacing(5)

        h = QtWidgets.QHBoxLayout()
        self.txt_res = QtWidgets.QLineEdit(str(self.current_scan_res))
        lbl_res = QtWidgets.QLabel("Scan Px (nm):")
        lbl_res.setStyleSheet("font-weight: bold;")
        h.addWidget(lbl_res)
        h.addWidget(self.txt_res)
        btn = QtWidgets.QPushButton("Update")
        btn.clicked.connect(self.update_resolution)
        h.addWidget(btn)
        lo.addLayout(h)

        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Strict", "Center", "Coverage"])
        self.combo_mode.setCurrentIndex(1)
        self.combo_mode.currentTextChanged.connect(self.recalculate_points)
        lo.addWidget(QtWidgets.QLabel("Fill Mode:"))
        lo.addWidget(self.combo_mode)

        self.chk_4th = QtWidgets.QCheckBox("Show 4th Distance")
        self.chk_4th.toggled.connect(self.update_visibility)
        lo.addWidget(self.chk_4th)

        grp.setLayout(lo)
        return grp

    def _create_appearance_group(self):
        grp = QtWidgets.QGroupBox("Appearance")
        lo = QtWidgets.QVBoxLayout()
        lo.setSpacing(2)

        h_rend = QtWidgets.QHBoxLayout()
        h_rend.addWidget(QtWidgets.QLabel("Volume Blending:"))
        self.combo_render = QtWidgets.QComboBox()
        self.combo_render.addItems(["Composite", "MIP (Maximum)", "MinIP (Minimum)", "Average"])
        self.combo_render.currentTextChanged.connect(self.update_volume_render_mode)
        h_rend.addWidget(self.combo_render)
        lo.addLayout(h_rend)

        lo.addWidget(QtWidgets.QLabel("Cyl Opacity"))
        self.slider_cyl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cyl.setRange(0, 100)
        self.slider_cyl.setValue(60)
        self.slider_cyl.valueChanged.connect(self.update_opacity)
        lo.addWidget(self.slider_cyl)

        h_vol = QtWidgets.QHBoxLayout()
        h_vol.addWidget(QtWidgets.QLabel("Vol Opacity"))
        self.slider_vol = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_vol.setRange(0, 100)
        self.slider_vol.setValue(10)
        self.slider_vol.valueChanged.connect(self.update_opacity)
        h_vol.addWidget(self.slider_vol)
        h_vol.addWidget(QtWidgets.QLabel("Max:"))
        self.txt_vol_max = QtWidgets.QLineEdit("1.0")
        self.txt_vol_max.setFixedWidth(40)
        self.txt_vol_max.editingFinished.connect(self.update_opacity)
        h_vol.addWidget(self.txt_vol_max)
        lo.addLayout(h_vol)

        grp.setLayout(lo)
        return grp

    def _create_orient_group(self):
        grp = QtWidgets.QGroupBox("Output Flip")
        lo = QtWidgets.QHBoxLayout()
        self.chk_flip_x = QtWidgets.QCheckBox("X")
        self.chk_flip_y = QtWidgets.QCheckBox("Y")
        self.chk_flip_z = QtWidgets.QCheckBox("Z")
        lo.addWidget(self.chk_flip_x)
        lo.addWidget(self.chk_flip_y)
        lo.addWidget(self.chk_flip_z)
        grp.setLayout(lo)
        return grp

    def _create_roi_shift_content(self):
        widget = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(widget)
        lo.setContentsMargins(10, 5, 10, 5)
        lo.setSpacing(2)
        h = QtWidgets.QHBoxLayout()
        self.txt_step = QtWidgets.QLineEdit("10")
        self.txt_step.setFixedWidth(50)
        h.addWidget(QtWidgets.QLabel("Step:"))
        h.addWidget(self.txt_step)
        h.addStretch()
        self.lbl_offset = QtWidgets.QLabel("[0,0,0]")
        h.addWidget(self.lbl_offset)
        lo.addLayout(h)

        for i, ax in enumerate(['X','Y','Z']):
            hx = QtWidgets.QHBoxLayout()
            bm = QtWidgets.QPushButton(f"- {ax}")
            bm.clicked.connect(lambda _, a=i: self.nudge_rois(a, -1))
            bp = QtWidgets.QPushButton(f"+ {ax}")
            bp.clicked.connect(lambda _, a=i: self.nudge_rois(a, 1))
            hx.addWidget(bm)
            hx.addWidget(bp)
            lo.addLayout(hx)

        br = QtWidgets.QPushButton("Reset Shift")
        br.clicked.connect(self.reset_rois)
        lo.addWidget(br)
        return widget

    def _create_roi_group(self):
        grp = QtWidgets.QGroupBox("Bounding Boxes")
        lo = QtWidgets.QVBoxLayout()
        lo.setSpacing(2)
        bnml = QtWidgets.QPushButton("Load NML")
        bnml.clicked.connect(self.load_nml_dialog)
        lo.addWidget(bnml)

        self.roi_list_widget = QtWidgets.QListWidget()
        self.roi_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.refresh_roi_list()
        lo.addWidget(self.roi_list_widget)

        h = QtWidgets.QHBoxLayout()
        self.txt_roi = QtWidgets.QLineEdit()
        self.txt_roi.setPlaceholderText("x,y,z,w,h,d")
        badd = QtWidgets.QPushButton("Add")
        badd.clicked.connect(self.add_roi_from_text)
        h.addWidget(self.txt_roi)
        h.addWidget(badd)
        lo.addLayout(h)

        bdel = QtWidgets.QPushButton("Delete Selected")
        bdel.clicked.connect(self.delete_selected_rois)
        lo.addWidget(bdel)
        grp.setLayout(lo)
        return grp

    def _create_manual_content(self):
        widget = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(widget)
        lo.setContentsMargins(10, 5, 10, 5)
        lo.setSpacing(2)
        self.txt_man_input = QtWidgets.QTextEdit()
        self.txt_man_input.setPlaceholderText("Paste: X, Y, Z (one per line)")
        self.txt_man_input.setFixedHeight(60)
        lo.addWidget(self.txt_man_input)

        h = QtWidgets.QHBoxLayout()
        b_add_man = QtWidgets.QPushButton("Add Bulk")
        b_add_man.clicked.connect(self.add_manual_points)
        b_clr_man = QtWidgets.QPushButton("Clear Input")
        b_clr_man.clicked.connect(lambda: self.txt_man_input.clear())
        h.addWidget(b_add_man)
        h.addWidget(b_clr_man)
        lo.addLayout(h)

        self.chk_show_manual = QtWidgets.QCheckBox("Show Manual Cylinders")
        self.chk_show_manual.setChecked(True)
        self.chk_show_manual.toggled.connect(self.update_3d_scene)
        lo.addWidget(self.chk_show_manual)

        self.man_list_widget = QtWidgets.QListWidget()
        self.man_list_widget.itemChanged.connect(self.on_man_item_changed)
        lo.addWidget(self.man_list_widget)

        b_del_man = QtWidgets.QPushButton("Delete Selected")
        b_del_man.clicked.connect(self.delete_manual_points)
        lo.addWidget(b_del_man)
        return widget

    def _create_auto_grid_group(self):
        grp = QtWidgets.QGroupBox("Generated Cylinders")
        lo = QtWidgets.QVBoxLayout()
        self.cyl_list_widget = QtWidgets.QListWidget()
        self.cyl_list_widget.itemChanged.connect(self.on_cyl_item_changed)
        lo.addWidget(self.cyl_list_widget)

        h = QtWidgets.QHBoxLayout()
        ba = QtWidgets.QPushButton("All")
        ba.clicked.connect(lambda: self.set_all_cyls(True))
        bn = QtWidgets.QPushButton("None")
        bn.clicked.connect(lambda: self.set_all_cyls(False))
        h.addWidget(ba)
        h.addWidget(bn)
        lo.addLayout(h)
        grp.setLayout(lo)
        return grp

    def _create_actions_group(self):
        grp = QtWidgets.QGroupBox("Export & Registration")
        lo = QtWidgets.QVBoxLayout()

        be_nml = QtWidgets.QPushButton("EXPORT NML (TILES)")
        be_nml.clicked.connect(self.export_nml_tiles)
        be_nml.setStyleSheet("background-color: #f0ad4e; color: white;")
        lo.addWidget(be_nml)

        be = QtWidgets.QPushButton("PRINT CONSOLE")
        be.clicked.connect(self.export_coordinates)
        be.setStyleSheet("background-color: blue; color: white;")
        lo.addWidget(be)

        breg = QtWidgets.QPushButton("REGISTER COORDINATES")
        breg.clicked.connect(self.open_registration_dialog)
        breg.setStyleSheet("background-color: purple; color: white; font-weight: bold; padding: 10px;")
        lo.addWidget(breg)

        grp.setLayout(lo)
        return grp

    def update_volume_render_mode(self):
        if self.vol_grid is None: return
        if self.vol_actor is not None:
            self.plotter.remove_actor(self.vol_actor)

        mode_str = self.combo_render.currentText()
        blend_mode = "composite"
        if "MIP" in mode_str: blend_mode = "maximum"
        elif "MinIP" in mode_str: blend_mode = "minimum"
        elif "Average" in mode_str: blend_mode = "average"

        self.vol_actor = self.plotter.add_volume(self.vol_grid, cmap="gray", clim=self.clim, opacity="linear", blending=blend_mode)
        self.update_opacity()

    def export_nml_tiles(self):
        pts = self.get_all_active_points()
        if len(pts) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No active cylinders to export.")
            return

        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NML Tiles", "tiles.nml", "NML Files (*.nml)", options=options)
        if fileName:
            base, ext = os.path.splitext(fileName)
            D_std, H_std = self.dims_std
            generate_nml(base + "_std" + ext, pts, D_std, H_std, color_hex="#00FFFF")
            D_exp, H_exp = self.dims_exp
            generate_nml(base + "_exp" + ext, pts, D_exp, H_exp, color_hex="#FF00FF")
            QtWidgets.QMessageBox.information(self, "Success", f"Saved NML bounding boxes.")

    def add_manual_points(self):
        text = self.txt_man_input.toPlainText()
        for line in text.strip().split('\n'):
            parts = line.replace(',', ' ').split()
            if len(parts) >= 3:
                try:
                    self.manual_points.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
                    self.active_manual_mask.append(True)
                except ValueError: pass
        self.refresh_manual_list()
        self.update_3d_scene()
        self.txt_man_input.clear()

    def delete_manual_points(self):
        indices = sorted([item.row() for item in self.man_list_widget.selectedIndexes()], reverse=True)
        for i in indices:
            del self.manual_points[i]
            del self.active_manual_mask[i]
        self.refresh_manual_list()
        self.update_3d_scene()

    def refresh_manual_list(self):
        self.man_list_widget.blockSignals(True)
        self.man_list_widget.clear()
        for i, p in enumerate(self.manual_points):
            item = QtWidgets.QListWidgetItem(f"Man #{i}: {p.astype(int)}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if self.active_manual_mask[i] else QtCore.Qt.Unchecked)
            self.man_list_widget.addItem(item)
        self.man_list_widget.blockSignals(False)

    def on_man_item_changed(self, item):
        self.active_manual_mask[self.man_list_widget.row(item)] = (item.checkState() == QtCore.Qt.Checked)
        self.update_3d_scene()

    def get_all_active_points(self):
        final_points = []
        idx_auto = np.where(self.active_mask)[0]
        if len(idx_auto) > 0:
            final_points.append(self.all_points[idx_auto])
        if self.chk_show_manual.isChecked() and len(self.manual_points) > 0:
            man_active = [p for p, a in zip(self.manual_points, self.active_manual_mask) if a]
            if len(man_active) > 0:
                final_points.append(np.array(man_active))

        if len(final_points) == 0: return np.empty((0,3))
        return np.vstack(final_points)

    def open_registration_dialog(self):
        self.reg_dialog = RegistrationDialog(self)
        self.reg_dialog.in_final_px.setText(str(self.current_scan_res))
        self.reg_dialog.show()

    def load_nml_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open NML", "", "NML (*.nml)", options=options)
        if fileName:
            new = parse_nml(fileName)
            if new:
                self.rois = new
                self.refresh_roi_list()
                self.recalculate_points()

    def update_resolution(self):
        try:
            self.current_scan_res = float(self.txt_res.text())
            self.recalculate_points()
        except ValueError: pass

    def nudge_rois(self, ax, d):
        try: s = int(self.txt_step.text())
        except ValueError: s = 10
        delta = s * d

        for r in self.rois:
            if ax == 0: r['x'] += delta
            elif ax == 1: r['y'] += delta
            elif ax == 2: r['z'] += delta

        self.total_roi_shift[ax] += delta
        self.lbl_offset.setText(str(self.total_roi_shift))
        self.refresh_roi_list()
        self.recalculate_points()

    def reset_rois(self):
        for r in self.rois:
            r['x'] -= self.total_roi_shift[0]
            r['y'] -= self.total_roi_shift[1]
            r['z'] -= self.total_roi_shift[2]

        self.total_roi_shift = [0,0,0]
        self.lbl_offset.setText(str(self.total_roi_shift))
        self.refresh_roi_list()
        self.recalculate_points()

    def refresh_roi_list(self):
        self.roi_list_widget.clear()
        for i, r in enumerate(self.rois):
            self.roi_list_widget.addItem(f"Box {i}: {list(r.values())}")

    def add_roi_from_text(self):
        try:
            p = [int(x) for x in self.txt_roi.text().split(',')]
            if len(p) == 6:
                self.rois.append({'x':p[0],'y':p[1],'z':p[2],'w':p[3],'h':p[4],'d':p[5]})
                self.refresh_roi_list()
                self.recalculate_points()
            else:
                self.statusBar().showMessage("ROI format: x,y,z,w,h,d (6 integers)", 5000)
        except (ValueError, AttributeError):
            self.statusBar().showMessage("Bad ROI format. Expected: x,y,z,w,h,d", 5000)

    def delete_selected_rois(self):
        for i in sorted([item.row() for item in self.roi_list_widget.selectedIndexes()], reverse=True):
            del self.rois[i]
        self.refresh_roi_list()
        self.recalculate_points()

    def recalculate_points(self):
        mode = self.combo_mode.currentText().lower()
        self.all_points, self.dims_std, self.dims_exp = solve_global_union(self.rois, self.current_scan_res, self.cfg, mode)
        self.active_mask = np.ones(len(self.all_points), dtype=bool)
        self.refresh_cyl_list()
        self.update_3d_scene()

    def update_cyl_list_labels(self):
        self.cyl_list_widget.blockSignals(True)
        current_seq = 0
        for i in range(self.cyl_list_widget.count()):
            it = self.cyl_list_widget.item(i)
            p = self.all_points[i].astype(int)
            if it.checkState() == QtCore.Qt.Checked:
                it.setText(f"Seq #{current_seq} (Orig {i}): {p}")
                current_seq += 1
            else:
                it.setText(f"--- (Orig {i}): {p}")
        self.cyl_list_widget.blockSignals(False)

    def refresh_cyl_list(self):
        self.cyl_list_widget.blockSignals(True)
        self.cyl_list_widget.clear()
        for i, p in enumerate(self.all_points):
            item = QtWidgets.QListWidgetItem(f"Seq #{i} (Orig {i}): {p.astype(int)}")
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked)
            self.cyl_list_widget.addItem(item)
        self.cyl_list_widget.blockSignals(False)

    def on_cyl_item_changed(self, item):
        self.active_mask[self.cyl_list_widget.row(item)] = (item.checkState() == QtCore.Qt.Checked)
        self.update_cyl_list_labels()
        self.update_3d_scene()

    def set_all_cyls(self, s):
        self.cyl_list_widget.blockSignals(True)
        st = QtCore.Qt.Checked if s else QtCore.Qt.Unchecked
        for i in range(self.cyl_list_widget.count()):
            self.cyl_list_widget.item(i).setCheckState(st)
            self.active_mask[i] = s
        self.cyl_list_widget.blockSignals(False)
        self.update_cyl_list_labels()
        self.update_3d_scene()

    def update_3d_scene(self):
        if self.actor_std: self.plotter.remove_actor(self.actor_std)
        if self.actor_exp: self.plotter.remove_actor(self.actor_exp)
        if self.actor_man: self.plotter.remove_actor(self.actor_man)
        if self.actor_labels: self.plotter.remove_actor(self.actor_labels)

        for act in self.roi_actors:
            self.plotter.remove_actor(act)
        self.roi_actors.clear()

        for r in self.rois:
            o = np.array([r['x'], r['y'], r['z']*self.z_ratio])
            va=np.array([r['w'], 0, 0])
            vb=np.array([0, r['h'], 0])
            vc=np.array([0, 0, r['d']*self.z_ratio])
            cube = pv.Cube(bounds=(o[0], o[0]+va[0], o[1], o[1]+vb[1], o[2], o[2]+vc[2]))
            act = self.plotter.add_mesh(cube, style='wireframe', color='cyan', line_width=2)
            self.roi_actors.append(act)

        # Collect all label points and sequential IDs
        label_points = []
        label_ids = []
        seq = 0

        idx_auto = np.where(self.active_mask)[0]
        if len(idx_auto) > 0:
            vp = self.all_points[idx_auto].copy()
            vp[:,2] *= self.z_ratio
            d_std, h_std = self.dims_std; d_exp, h_exp = self.dims_exp
            vis_H_std = h_std * self.z_ratio; vis_H_exp = h_exp * self.z_ratio

            c1 = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=d_std/2, height=vis_H_std)
            self.actor_std = self.plotter.add_mesh(pv.PolyData(vp).glyph(geom=c1, scale=False), color='cyan', opacity=self.slider_cyl.value()/100)

            c2 = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=d_exp/2, height=vis_H_exp)
            self.actor_exp = self.plotter.add_mesh(pv.PolyData(vp).glyph(geom=c2, scale=False), color='magenta', opacity=self.slider_cyl.value()/100)

            for pt in vp:
                label_points.append(pt)
                label_ids.append(str(seq))
                seq += 1

        if self.chk_show_manual.isChecked() and len(self.manual_points) > 0:
            man_active = [p for p, a in zip(self.manual_points, self.active_manual_mask) if a]
            if len(man_active) > 0:
                mp = np.array(man_active).copy()
                mp[:, 2] *= self.z_ratio
                vis_H_std = self.dims_std[1] * self.z_ratio

                c_man = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=self.dims_std[0]/2, height=vis_H_std)
                self.actor_man = self.plotter.add_mesh(pv.PolyData(mp).glyph(geom=c_man, scale=False), color='yellow', opacity=self.slider_cyl.value()/100)

                for pt in mp:
                    label_points.append(pt)
                    label_ids.append(f"M{seq}")
                    seq += 1

        if label_points:
            label_poly = pv.PolyData(np.array(label_points))
            label_poly["labels"] = label_ids
            self.actor_labels = self.plotter.add_point_labels(
                label_poly, "labels", font_size=12, text_color="white",
                shadow=True, point_size=1, render_points_as_spheres=False,
                always_visible=True, shape_opacity=0.4
            )

        self.update_visibility()
        self.update_opacity()

    def update_visibility(self):
        show = self.chk_4th.isChecked()
        if self.actor_std: self.actor_std.SetVisibility(not show)
        if self.actor_exp: self.actor_exp.SetVisibility(show)

    def update_opacity(self):
        cyl_op = self.slider_cyl.value() / 100.0
        if self.actor_std: self.actor_std.GetProperty().SetOpacity(cyl_op)
        if self.actor_exp: self.actor_exp.GetProperty().SetOpacity(cyl_op)
        if self.actor_man: self.actor_man.GetProperty().SetOpacity(cyl_op)

        if self.vol_actor:
            try: vol_max = float(self.txt_vol_max.text())
            except ValueError: vol_max = 1.0

            vol_op = (self.slider_vol.value() / 100.0) * vol_max
            otf = self.vol_actor.GetProperty().GetScalarOpacity()
            otf.RemoveAllPoints()
            otf.AddPoint(self.clim[0], 0.0)
            otf.AddPoint(self.clim[1], vol_op)

        self.plotter.render()

    def export_coordinates(self):
        pts = self.get_all_active_points()
        if len(pts) == 0:
            print("\n=== SCAN EXPORT ===\nNo active cylinders.\n")
            return

        if self.chk_flip_x.isChecked(): pts[:,0] = self.max_dims[0]-pts[:,0]
        if self.chk_flip_y.isChecked(): pts[:,1] = self.max_dims[1]-pts[:,1]
        if self.chk_flip_z.isChecked(): pts[:,2] = self.max_dims[2]-pts[:,2]

        print("\n=== SCAN EXPORT ===")
        print(f"Total Active (Auto + Manual): {len(pts)}")
        print("ID, X, Y, Z")
        for i,p in enumerate(pts):
            print(f"{i},{int(p[0])},{int(p[1])},{int(p[2])}")
