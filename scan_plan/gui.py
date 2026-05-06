"""
GUI classes: CylinderApp (main window) and RegistrationDialog.
"""

import os
import re
import numpy as np
import pyvista as pv
import pandas as pd
from scipy.spatial.distance import cdist

from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor

from scan_plan.volume_registration import VolumeRegistration
from scan_plan.nml_exporter import generate_nml
from scan_plan.io import parse_nml, detect_tiff_dims, update_user_config_keys
from scan_plan.solver import solve_bbox_grids, solve_line_coverage


class ConfigDialog(QtWidgets.QDialog):
    """Startup wizard for editing the scan_plan JSON config without opening the file."""

    EDITABLE_KEYS = (
        "volume_path", "binning",
        "raw_dims", "raw_dtype", "raw_header_bytes",
        "prescan_pixel_size_xy", "prescan_z_step", "scan_pixel_size",
    )

    def __init__(self, config, config_path, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.config_path = config_path
        self.accepted_proceed = False

        self.setWindowTitle("Scan Plan — Configuration")
        self.resize(640, 0)

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            f"Edit your session settings below. Changes are saved to:\n  {config_path}"
        )
        intro.setStyleSheet("color: #555;")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form_box = QtWidgets.QGroupBox("Session Settings")
        form = QtWidgets.QFormLayout()

        # Volume file picker
        h_vol = QtWidgets.QHBoxLayout()
        self.txt_volume = QtWidgets.QLineEdit(str(config.get("volume_path", "")))
        # Whenever the path field is committed (Enter / focus-out), try to
        # auto-fill raw_dims from the filename pattern.
        self.txt_volume.editingFinished.connect(
            lambda: self._try_prefill_dims_from_filename(self.txt_volume.text())
        )
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_volume)
        h_vol.addWidget(self.txt_volume)
        h_vol.addWidget(btn_browse)
        form.addRow("Volume file:", h_vol)

        self.spin_binning = QtWidgets.QSpinBox()
        self.spin_binning.setRange(1, 64)
        self.spin_binning.setValue(int(config.get("binning", 1)))
        form.addRow("Binning:", self.spin_binning)

        dims = config.get("raw_dims", [2048, 2048, 2048])
        h_dims = QtWidgets.QHBoxLayout()
        self.spin_dim_x = QtWidgets.QSpinBox(); self.spin_dim_x.setRange(1, 100000); self.spin_dim_x.setValue(int(dims[0]))
        self.spin_dim_y = QtWidgets.QSpinBox(); self.spin_dim_y.setRange(1, 100000); self.spin_dim_y.setValue(int(dims[1]))
        self.spin_dim_z = QtWidgets.QSpinBox(); self.spin_dim_z.setRange(1, 100000); self.spin_dim_z.setValue(int(dims[2]))
        for w in (self.spin_dim_x, self.spin_dim_y, self.spin_dim_z):
            w.setSuffix(" px")
        for lbl, w in (("X", self.spin_dim_x), ("Y", self.spin_dim_y), ("Z", self.spin_dim_z)):
            h_dims.addWidget(QtWidgets.QLabel(lbl))
            h_dims.addWidget(w)
        btn_detect = QtWidgets.QPushButton("Detect TIFF")
        btn_detect.clicked.connect(self._detect_dims)
        h_dims.addWidget(btn_detect)
        form.addRow("Raw dimensions:", h_dims)

        self.combo_dtype = QtWidgets.QComboBox()
        self.combo_dtype.setEditable(True)
        for dt in ("float32", "float64", "uint8", "uint16", "uint32", "int16", "int32"):
            self.combo_dtype.addItem(dt)
        self.combo_dtype.setCurrentText(str(config.get("raw_dtype", "float32")))
        form.addRow("Raw dtype:", self.combo_dtype)

        self.spin_header = QtWidgets.QSpinBox()
        self.spin_header.setRange(0, 1_000_000_000)
        self.spin_header.setValue(int(config.get("raw_header_bytes", 0)))
        self.spin_header.setSuffix(" B")
        form.addRow("Raw header bytes:", self.spin_header)

        self.spin_prescan_xy = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_xy.setRange(0.01, 100000.0)
        self.spin_prescan_xy.setDecimals(2)
        self.spin_prescan_xy.setValue(float(config.get("prescan_pixel_size_xy", 150)))
        self.spin_prescan_xy.setSuffix(" nm")
        form.addRow("Prescan pixel size XY:", self.spin_prescan_xy)

        self.spin_prescan_z = QtWidgets.QDoubleSpinBox()
        self.spin_prescan_z.setRange(0.01, 100000.0)
        self.spin_prescan_z.setDecimals(2)
        self.spin_prescan_z.setValue(float(config.get("prescan_z_step", 150)))
        self.spin_prescan_z.setSuffix(" nm")
        form.addRow("Prescan Z step:", self.spin_prescan_z)

        self.spin_scan_px = QtWidgets.QDoubleSpinBox()
        self.spin_scan_px.setRange(0.01, 100000.0)
        self.spin_scan_px.setDecimals(2)
        self.spin_scan_px.setValue(float(config.get("scan_pixel_size", 20)))
        self.spin_scan_px.setSuffix(" nm")
        form.addRow("Scan pixel size:", self.spin_scan_px)

        form_box.setLayout(form)
        layout.addWidget(form_box)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Save && Continue")
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText("Cancel")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse_volume(self):
        start_dir = ""
        cur = self.txt_volume.text().strip()
        if cur:
            cand_dir = os.path.dirname(cur)
            if os.path.isdir(cand_dir):
                start_dir = cand_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Volume",
            start_dir,
            "Volumes (*.tif *.tiff *.raw *.vol);;All files (*)"
        )
        if path:
            self.txt_volume.setText(path)
            self._detect_dims(path)
            self._try_prefill_dims_from_filename(path)

    # Match `_<int>x<int>x<int>` immediately before the file extension.
    # Accepts either lowercase `x` or uppercase `X` as the separator.
    _DIMS_PATTERN = re.compile(r"_(\d+)[xX](\d+)[xX](\d+)$")

    def _try_prefill_dims_from_filename(self, path):
        """If the basename ends in `_XxYxZ.<ext>` (e.g. sample_512x512x512.raw),
        prefill the raw_dims spinboxes with those integers.

        TIFF auto-detection (via _detect_dims) is authoritative when it
        succeeds, so for `.tif` / `.tiff` we skip this — the header values
        already populated the fields. For headerless `.raw` / `.vol` (and
        anything else), the filename pattern is the only signal we have.
        """
        if not path:
            return
        base = os.path.basename(str(path))
        stem, ext = os.path.splitext(base)
        if ext.lower() in (".tif", ".tiff"):
            return
        m = self._DIMS_PATTERN.search(stem)
        if not m:
            return
        try:
            x, y, z = int(m.group(1)), int(m.group(2)), int(m.group(3))
        except ValueError:
            return
        self.spin_dim_x.setValue(x)
        self.spin_dim_y.setValue(y)
        self.spin_dim_z.setValue(z)

    def _detect_dims(self, path=None):
        if not isinstance(path, str) or not path:
            path = self.txt_volume.text().strip()
        if not path or not os.path.exists(path):
            return
        det = detect_tiff_dims(path)
        if det is None:
            return
        new_dims, new_dtype = det
        self.spin_dim_x.setValue(int(new_dims[0]))
        self.spin_dim_y.setValue(int(new_dims[1]))
        self.spin_dim_z.setValue(int(new_dims[2]))
        self.combo_dtype.setCurrentText(str(new_dtype))

    def _on_accept(self):
        # Spinboxes guarantee numeric values — no try/except needed.
        updates = {
            "volume_path": self.txt_volume.text().strip(),
            "binning": self.spin_binning.value(),
            "raw_dims": [
                self.spin_dim_x.value(),
                self.spin_dim_y.value(),
                self.spin_dim_z.value(),
            ],
            "raw_dtype": self.combo_dtype.currentText().strip(),
            "raw_header_bytes": self.spin_header.value(),
            "prescan_pixel_size_xy": self.spin_prescan_xy.value(),
            "prescan_z_step": self.spin_prescan_z.value(),
            "scan_pixel_size": self.spin_scan_px.value(),
        }
        self.config.update(updates)
        self.accepted_proceed = True
        self.accept()

    def get_updates(self):
        return {k: self.config[k] for k in self.EDITABLE_KEYS if k in self.config}


class RegistrationDialog(QtWidgets.QDialog):
    def __init__(self, main_app):
        super().__init__(parent=main_app)
        self.main_app = main_app
        self.ref_pts = []
        self.ref_px = 0
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
        def _mm(default):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1000.0, 1000.0)
            sb.setDecimals(5)
            sb.setSingleStep(0.001)
            sb.setSuffix(" mm")
            sb.setValue(default)
            return sb
        self.in_su = _mm(0.0)
        self.in_sv = _mm(0.0)
        self.in_sz = _mm(0.0)
        self.in_px = QtWidgets.QDoubleSpinBox()
        self.in_px.setRange(0.01, 100000.0); self.in_px.setDecimals(2)
        self.in_px.setValue(150.0); self.in_px.setSuffix(" nm")
        self.in_final_px = QtWidgets.QDoubleSpinBox()
        self.in_final_px.setRange(0.01, 100000.0); self.in_final_px.setDecimals(2)
        self.in_final_px.setValue(100.0); self.in_final_px.setSuffix(" nm")
        fl_conf.addRow("su:", self.in_su)
        fl_conf.addRow("sv:", self.in_sv)
        fl_conf.addRow("sz:", self.in_sz)
        fl_conf.addRow("Refscan Pixel Size:", self.in_px)
        fl_conf.addRow("Final Pixel Size:", self.in_final_px)
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
        self.table_results = QtWidgets.QTableWidget(0, 5)
        self.table_results.setHorizontalHeaderLabels(
            ["ID", "Refscan (Pixels)", "Transformed", "Error (µm)", "Motors (su, sv, sz mm)"]
        )
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        l_det.addWidget(self.table_results)
        gb_det.setLayout(l_det)
        layout.addWidget(gb_det)

        # Motor-limit summary — populated by update_results_ui. Stays visible
        # next to the SAVE button so out-of-range placement is obvious before
        # the file dialog opens.
        self.lbl_motor_warn = QtWidgets.QLabel("Motor limits: —")
        self.lbl_motor_warn.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.lbl_motor_warn)

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
        added = self.table_pre.rowCount()
        self.lbl_result.setText(f"Status: Pasted {added} of {len(lines)} rows.")

    def load_match_points_from_file(self):
        options = QtWidgets.QFileDialog.Options()
        start_dir = self.main_app.cfg.get('last_match_points_dir', '')
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Match Points", start_dir, "Text Files (*.txt)", options=options)
        if not fileName:
            return
        self.main_app._persist_session_pref('last_match_points_dir', os.path.dirname(fileName))
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
        # Spinboxes guarantee numeric values — no validation try/except.
        try:
            pre_px = self.main_app.cfg["prescan_pixel_size_xy"]
            su = self.in_su.value()
            sv = self.in_sv.value()
            sz = self.in_sz.value()
            self.ref_px = self.in_px.value()

            pre_pts, ref_pts_raw = self.get_points()
            if len(pre_pts) < 3:
                self.lbl_result.setText("Status: Need at least 3 matching points.")
                return

            optics = self.main_app.cfg.get('optics', {})

            # If motor coordinate mode, convert su/sv/sz → refscan pixels
            if self.combo_match_mode.currentIndex() == 1:
                final_px = self.in_final_px.value()
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

            progress = QtWidgets.QProgressDialog(
                "Computing registration...", None, 0, 0, self)
            progress.setWindowTitle("Registration")
            progress.setWindowModality(QtCore.Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setCancelButton(None)
            progress.show()
            QtWidgets.QApplication.processEvents()

            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                self.res_svd = self.vreg_svd.fitTransformationMatrix(rot_z_only=z_only, method='svd')
                self.res_opt = self.vreg_opt.fitTransformationMatrix(rot_z_only=z_only, method='optimizer')
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
                progress.close()

            n_pts = len(pre_pts)
            err_svd = np.mean(self.res_svd.distances * (self.ref_px / 1000.0))
            err_opt = np.mean(self.res_opt.distances * (self.ref_px / 1000.0))
            max_svd = np.max(self.res_svd.distances * (self.ref_px / 1000.0))
            max_opt = np.max(self.res_opt.distances * (self.ref_px / 1000.0))

            self.combo_result_select.blockSignals(True)
            self.combo_result_select.clear()
            self.combo_result_select.addItem(f"SVD (Kabsch) - Avg: {err_svd:.2f} \u00b5m, Max: {max_svd:.2f} \u00b5m ({n_pts} pts)")
            self.combo_result_select.addItem(f"Optimizer - Avg: {err_opt:.2f} \u00b5m, Max: {max_opt:.2f} \u00b5m ({n_pts} pts)")

            # Math (SVD/Kabsch) is the default; user can switch to Optimizer manually.
            self.combo_result_select.setCurrentIndex(0)
            self.combo_result_select.blockSignals(False)

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

        # Motor coords + limits column. Driven by the active set of cylinder
        # placements so the user sees overruns at registration time, not at
        # save time.
        self._update_motor_warnings(active_res)

    def _compute_motor_warnings(self, active_vreg):
        """Return (su, sv, sz arrays, list of overrun strings) for the
        currently active cylinder set.

        Returns (None, None, None, []) if the active set is empty.
        """
        pts = self.main_app.get_all_active_points()
        if len(pts) == 0:
            return None, None, None, []
        XYZcoords_refscan = active_vreg.transformToRefscan(pts)
        final_px = self.in_final_px.value()
        su, sv, sz = active_vreg.refscan_to_motors(XYZcoords_refscan, final_px)
        limits = self.main_app.cfg.get('motor_limits', {})
        warnings = []
        for axis_name, values in [("su", su), ("sv", sv), ("sz", sz)]:
            lim = limits.get(axis_name)
            if lim is None:
                continue
            lo, hi = lim
            for i, v in enumerate(values):
                if v < lo or v > hi:
                    warnings.append((i, axis_name, float(v), float(lo), float(hi)))
        return su, sv, sz, warnings

    def _update_motor_warnings(self, active_res):
        """Populate the motor column of the results table and the summary
        label below it. Out-of-range cells are tinted red."""
        idx = self.combo_result_select.currentIndex()
        active_vreg = self.vreg_svd if idx == 0 else self.vreg_opt
        if active_vreg is None:
            self.lbl_motor_warn.setText("Motor limits: registration not computed yet")
            self.lbl_motor_warn.setStyleSheet("font-weight: bold; padding: 4px; color: #777;")
            return

        try:
            su, sv, sz, warnings = self._compute_motor_warnings(active_vreg)
        except Exception as e:
            self.lbl_motor_warn.setText(f"Motor limits: error computing ({e})")
            self.lbl_motor_warn.setStyleSheet("font-weight: bold; padding: 4px; color: #b00;")
            return

        if su is None:
            self.lbl_motor_warn.setText("Motor limits: no active cylinders")
            self.lbl_motor_warn.setStyleSheet("font-weight: bold; padding: 4px; color: #777;")
            return

        bad_indices = {(i, axis) for i, axis, *_ in warnings}
        n_active = len(su)
        # Per-row motor column. Match-points table rows correspond to the
        # match-point set, not the active-cylinder set, so we render motor
        # info only when the row index lines up.
        for i in range(self.table_results.rowCount()):
            if i < n_active:
                txt = f"({su[i]:+.4f}, {sv[i]:+.4f}, {sz[i]:+.4f}) mm"
                item = QtWidgets.QTableWidgetItem(txt)
                row_bad = any((i, ax) in bad_indices for ax in ("su", "sv", "sz"))
                if row_bad:
                    item.setBackground(QtGui.QColor(255, 150, 150))
                    bad_axes = [ax for ax in ("su", "sv", "sz") if (i, ax) in bad_indices]
                    item.setToolTip("Out of range: " + ", ".join(bad_axes))
                else:
                    item.setBackground(QtGui.QColor(220, 255, 220))
                self.table_results.setItem(i, 4, item)
            else:
                self.table_results.setItem(i, 4, QtWidgets.QTableWidgetItem("\u2014"))

        if warnings:
            self.lbl_motor_warn.setText(
                f"Motor limits: {len({i for i,_,*_ in warnings})} of {n_active} cylinder(s) out of range"
            )
            self.lbl_motor_warn.setStyleSheet(
                "font-weight: bold; padding: 4px; "
                "background-color: #fbe5e5; color: #a40000; border: 1px solid #a40000;"
            )
        else:
            self.lbl_motor_warn.setText(f"Motor limits: all {n_active} cylinder(s) within range")
            self.lbl_motor_warn.setStyleSheet(
                "font-weight: bold; padding: 4px; "
                "background-color: #e3f7e3; color: #176317; border: 1px solid #176317;"
            )

    def save_machine_file(self):
        idx = self.combo_result_select.currentIndex()
        if idx == -1 or not self.vreg_svd:
            QtWidgets.QMessageBox.warning(self, "Error", "Please calculate registration first.")
            return

        active_vreg = self.vreg_svd if idx == 0 else self.vreg_opt

        pts = self.main_app.get_all_active_points()

        if len(pts) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No active cylinders.")
            return

        try:
            su, sv, sz, warnings = self._compute_motor_warnings(active_vreg)
            if su is None:
                QtWidgets.QMessageBox.warning(self, "Error", "No active cylinders.")
                return
            if warnings:
                lines = [
                    f"  Cylinder {i}: {axis} = {v:.4f} mm (limits: [{lo}, {hi}])"
                    for i, axis, v, lo, hi in warnings
                ]
                msg = (
                    f"{len(warnings)} cylinder(s) exceed motor travel limits "
                    f"(also shown in red in the Per-Point Analysis table):\n\n"
                    + "\n".join(lines)
                )
                QtWidgets.QMessageBox.warning(self, "Motor Limit Warning", msg)

            df = pd.DataFrame(np.array([su, sv, sz]).T, columns=["#su", "sv", "sz"])

            options = QtWidgets.QFileDialog.Options()
            start_dir = self.main_app.cfg.get('last_machine_save_dir', '')
            seed = os.path.join(start_dir, "tiles_motor_coords.txt") if start_dir else "tiles_motor_coords.txt"
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Machine Coordinates", seed, "Text Files (*.txt)", options=options)

            if fileName:
                self.main_app._persist_session_pref('last_machine_save_dir', os.path.dirname(fileName))
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
    def __init__(self, config, vol_grid, clim, config_path=None):
        super().__init__()
        self.cfg = config
        self.config_path = config_path

        # Keys persisted incidentally (last-used file dialog dirs, etc.).
        # Call _persist_session_pref('key', value) to save & flush.
        self._session_pref_keys = (
            'last_nml_load_dir',
            'last_nml_save_dir',
            'last_machine_save_dir',
            'last_match_points_dir',
        )
        self.rois = [r.copy() for r in config.get('rois', [])]
        self.vol_grid = vol_grid
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
        self.lines = []           # list of (p1, p2) tuples in prescan-pixel coords
        self.line_points = np.empty((0,3))
        self.line_density = 1.0
        self.active_line_cyl_mask = np.empty((0,), dtype=bool)

        # Single-slot undo stash per category. Each holds the most recent
        # set of items deleted by the corresponding "Delete Selected" action,
        # restorable via the matching "Restore last deleted" button. Cleared
        # on next mutation that isn't a restore.
        self._stash_rois = None       # list[(roi_dict)] | None
        self._stash_lines = None      # list[(line_tuple, active_bool)] | None
        self._stash_manual = None     # list[(np.array(3,), active_bool)] | None

        self.dims_std = (10,10)
        self.dims_exp = (10,10)
        self.active_mask = []
        self.active_manual_mask = []
        self.active_line_mask = []
        self.total_roi_shift = [0, 0, 0]
        self.current_scan_res = config['scan_pixel_size']

        self.roi_actors = []
        self.line_actors = []
        self.actor_std = None
        self.actor_exp = None
        self.actor_man = None
        self.actor_man_exp = None
        self.actor_line = None
        self.actor_line_exp = None
        self.vol_actor = None
        self.actor_labels = None
        self.ref_grid_actors = []  # list of actors so "Both" mode can stack two

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
        scroll.setFixedWidth(440)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        # Tighten group-box padding so the sidebar fits common laptop heights.
        panel.setStyleSheet(
            "QGroupBox { margin-top: 6px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 6px; padding: 0 3px; }"
            "QPushButton { padding: 3px 6px; }"
        )
        scroll.setWidget(panel)
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        layout.addWidget(self._create_appearance_group())
        layout.addWidget(self._create_add_cylinders_tabs())
        layout.addWidget(self._create_config_group())
        layout.addWidget(self._create_auto_grid_group())
        layout.addWidget(self._create_actions_group())

        layout.addStretch()
        main_layout.addWidget(scroll)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        # White, bold X/Y/Z labels with shadow so they're readable on both
        # dark volumes and bright slices. We pass the per-axis colors via
        # add_axes() *and* override the caption text properties because
        # different pyvista versions honor only one of these paths.
        try:
            axes_marker = self.plotter.add_axes(
                line_width=3,
                x_color="white", y_color="white", z_color="white",
                xlabel="X", ylabel="Y", zlabel="Z",
            )
        except TypeError:
            # Older pyvista signatures don't accept x_color/y_color/z_color.
            axes_marker = self.plotter.add_axes(line_width=3)
        try:
            ax_actor = axes_marker.GetOrientationMarker()
            for caption in (ax_actor.GetXAxisCaptionActor2D(),
                            ax_actor.GetYAxisCaptionActor2D(),
                            ax_actor.GetZAxisCaptionActor2D()):
                tp = caption.GetCaptionTextProperty()
                tp.SetColor(1.0, 1.0, 1.0)
                tp.SetBold(1)
                tp.SetItalic(0)
                tp.SetShadow(1)
                tp.SetShadowOffset(2, -2)
                tp.SetFontSize(16)
                # Some VTK builds hide labels until BackgroundOpacity > 0.
                tp.SetBackgroundColor(0.0, 0.0, 0.0)
                tp.SetBackgroundOpacity(0.0)
                # Force the caption to redraw with the new properties.
                caption.SetCaption(caption.GetCaption())
        except Exception:
            pass
        # Depth peeling lets opaque/translucent geometry composite correctly
        # over volume renderings; without it, "Average" blending tends to
        # wash cylinder/bbox colors into the volume's grayscale.
        try:
            self.plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
        except Exception:
            pass
        main_layout.addWidget(self.plotter)

    def _create_add_cylinders_tabs(self):
        """Group the three cylinder-source UIs (Bounding Boxes, Line Coverage,
        Manual Centers) into a single tabbed container so they live together."""
        grp = QtWidgets.QGroupBox("Add Cylinders")
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(4, 4, 4, 4)

        tabs = QtWidgets.QTabWidget()

        # --- Tab: Bounding Boxes (incl. shift sub-section) ---
        bbox_tab = QtWidgets.QWidget()
        bbox_lo = QtWidgets.QVBoxLayout(bbox_tab)
        bbox_lo.setContentsMargins(4, 4, 4, 4)
        bbox_lo.addWidget(self._create_roi_group())
        bbox_lo.addWidget(self._make_collapsible(
            "Shift Bounding Boxes", self._create_roi_shift_content()
        ))
        bbox_lo.addStretch()
        tabs.addTab(bbox_tab, "Bounding Boxes")

        # --- Tab: Line Coverage ---
        line_tab = QtWidgets.QWidget()
        line_lo = QtWidgets.QVBoxLayout(line_tab)
        line_lo.setContentsMargins(4, 4, 4, 4)
        line_lo.addWidget(self._create_line_content())
        line_lo.addStretch()
        tabs.addTab(line_tab, "Line Coverage")

        # --- Tab: Manual Centers ---
        man_tab = QtWidgets.QWidget()
        man_lo = QtWidgets.QVBoxLayout(man_tab)
        man_lo.setContentsMargins(4, 4, 4, 4)
        man_lo.addWidget(self._create_manual_content())
        man_lo.addStretch()
        tabs.addTab(man_tab, "Manual Centers")

        outer.addWidget(tabs)
        grp.setLayout(outer)
        return grp

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
        self.spin_res = QtWidgets.QDoubleSpinBox()
        self.spin_res.setRange(0.01, 100000.0)
        self.spin_res.setDecimals(2)
        self.spin_res.setValue(float(self.current_scan_res))
        self.spin_res.setSuffix(" nm")
        # editingFinished fires once on commit (Enter or focus-out), so
        # solve_global_union doesn't thrash mid-typing the way valueChanged
        # would.
        self.spin_res.editingFinished.connect(self.update_resolution)
        lbl_res = QtWidgets.QLabel("Scan Px:")
        lbl_res.setStyleSheet("font-weight: bold;")
        h.addWidget(lbl_res)
        h.addWidget(self.spin_res)
        lo.addLayout(h)

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
        self.combo_render.addItems(["Composite", "MIP (Maximum)", "MinIP (Minimum)", "Average", "Additive"])
        self.combo_render.currentTextChanged.connect(self.update_volume_render_mode)
        h_rend.addWidget(self.combo_render)
        lo.addLayout(h_rend)

        h_invert = QtWidgets.QHBoxLayout()
        self.chk_invert_volume = QtWidgets.QCheckBox("Invert Volume")
        self.chk_invert_volume.setToolTip(
            "Map dark voxels bright and vice-versa (cmap='gray_r').\n"
            "Often makes empty resin recede while dense material pops, "
            "especially in MIP / Composite blending. Volume blending logic "
            "is unchanged."
        )
        self.chk_invert_volume.toggled.connect(self.update_volume_render_mode)
        h_invert.addWidget(self.chk_invert_volume)
        h_invert.addStretch()
        lo.addLayout(h_invert)

        h_curve = QtWidgets.QHBoxLayout()
        h_curve.addWidget(QtWidgets.QLabel("Opacity Curve:"))
        self.combo_opacity_curve = QtWidgets.QComboBox()
        self.combo_opacity_curve.addItems([
            "Linear",
            "Sigmoid (gentle)", "Sigmoid (medium)", "Sigmoid (sharp)",
            "Threshold (low)", "Threshold (mid)", "Threshold (high)",
        ])
        self.combo_opacity_curve.setToolTip(
            "Linear: standard ramp from clim[0]→0 to clim[1]→Max.\n"
            "Sigmoid: S-curve transfer function — suppresses dark voxels "
            "(empty resin) while keeping the bright object visible.\n"
            "Threshold: hard cutoff at 30% / 50% / 70% of the dynamic range."
        )
        self.combo_opacity_curve.currentTextChanged.connect(self.update_opacity)
        h_curve.addWidget(self.combo_opacity_curve)
        lo.addLayout(h_curve)

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
        self.txt_vol_max.setFixedWidth(50)
        self.txt_vol_max.returnPressed.connect(self.update_opacity)
        h_vol.addWidget(self.txt_vol_max)
        # Up/down arrows that scale Max by 10 each click (order of magnitude).
        btn_max_up = QtWidgets.QToolButton()
        btn_max_up.setArrowType(QtCore.Qt.UpArrow)
        btn_max_up.setToolTip("Multiply Max by 10")
        btn_max_up.clicked.connect(lambda: self._scale_vol_max(10.0))
        btn_max_dn = QtWidgets.QToolButton()
        btn_max_dn.setArrowType(QtCore.Qt.DownArrow)
        btn_max_dn.setToolTip("Divide Max by 10")
        btn_max_dn.clicked.connect(lambda: self._scale_vol_max(0.1))
        v_arrows = QtWidgets.QVBoxLayout()
        v_arrows.setSpacing(0)
        v_arrows.setContentsMargins(0, 0, 0, 0)
        v_arrows.addWidget(btn_max_up)
        v_arrows.addWidget(btn_max_dn)
        arrows_w = QtWidgets.QWidget()
        arrows_w.setLayout(v_arrows)
        h_vol.addWidget(arrows_w)
        lo.addLayout(h_vol)

        # Reference grid controls — a floor grid in the volume's coordinate
        # frame, useful for gauging distances against the rendered scene.
        # Supports µm, px, or both (two distinguishable overlaid grids).
        h_grid_top = QtWidgets.QHBoxLayout()
        self.chk_ref_grid = QtWidgets.QCheckBox("Show Reference Grid")
        self.chk_ref_grid.toggled.connect(self._refresh_reference_grid)
        h_grid_top.addWidget(self.chk_ref_grid)
        h_grid_top.addWidget(QtWidgets.QLabel("Units:"))
        self.combo_grid_unit = QtWidgets.QComboBox()
        self.combo_grid_unit.addItems(["µm", "px", "Both"])
        self.combo_grid_unit.currentTextChanged.connect(self._on_grid_unit_changed)
        h_grid_top.addWidget(self.combo_grid_unit)
        h_grid_top.addStretch()
        lo.addLayout(h_grid_top)

        h_grid = QtWidgets.QHBoxLayout()
        # µm spacing (used in "µm" and "Both" modes)
        self.lbl_grid_um = QtWidgets.QLabel("µm:")
        h_grid.addWidget(self.lbl_grid_um)
        self.spin_grid_spacing = QtWidgets.QDoubleSpinBox()
        self.spin_grid_spacing.setRange(0.1, 100000.0)
        self.spin_grid_spacing.setDecimals(1)
        self.spin_grid_spacing.setValue(50.0)
        self.spin_grid_spacing.setSuffix(" µm")
        self.spin_grid_spacing.editingFinished.connect(self._refresh_reference_grid)
        h_grid.addWidget(self.spin_grid_spacing)
        # px spacing (used in "px" and "Both" modes)
        self.lbl_grid_px = QtWidgets.QLabel("px:")
        h_grid.addWidget(self.lbl_grid_px)
        self.spin_grid_spacing_px = QtWidgets.QSpinBox()
        self.spin_grid_spacing_px.setRange(1, 1_000_000)
        self.spin_grid_spacing_px.setValue(100)
        self.spin_grid_spacing_px.setSuffix(" px")
        self.spin_grid_spacing_px.editingFinished.connect(self._refresh_reference_grid)
        h_grid.addWidget(self.spin_grid_spacing_px)
        lo.addLayout(h_grid)

        h_grid_op = QtWidgets.QHBoxLayout()
        h_grid_op.addWidget(QtWidgets.QLabel("Opacity:"))
        self.slider_grid_op = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_grid_op.setRange(0, 100)
        self.slider_grid_op.setValue(40)
        self.slider_grid_op.valueChanged.connect(self._refresh_reference_grid)
        h_grid_op.addWidget(self.slider_grid_op)
        lo.addLayout(h_grid_op)

        # Initialize visibility for the default unit ("µm").
        self.lbl_grid_px.setVisible(False)
        self.spin_grid_spacing_px.setVisible(False)

        grp.setLayout(lo)
        return grp

    def _create_roi_shift_content(self):
        widget = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(widget)
        lo.setContentsMargins(10, 5, 10, 5)
        lo.setSpacing(2)
        h = QtWidgets.QHBoxLayout()
        self.spin_step = QtWidgets.QSpinBox()
        self.spin_step.setRange(1, 100000)
        self.spin_step.setValue(10)
        self.spin_step.setSuffix(" px")
        h.addWidget(QtWidgets.QLabel("Step:"))
        h.addWidget(self.spin_step)
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

        self.btn_reset_shift = QtWidgets.QPushButton("Reset Shift")
        self.btn_reset_shift.clicked.connect(self.reset_rois)
        lo.addWidget(self.btn_reset_shift)
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
        self.roi_list_widget.itemDoubleClicked.connect(self._on_roi_double_clicked)
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

        h_del = QtWidgets.QHBoxLayout()
        self.btn_del_rois = QtWidgets.QPushButton("Delete")
        self.btn_del_rois.setEnabled(False)
        self.btn_del_rois.clicked.connect(self.delete_selected_rois)
        self.btn_restore_rois = QtWidgets.QPushButton("Undo delete")
        self.btn_restore_rois.setEnabled(False)
        self.btn_restore_rois.clicked.connect(self.restore_last_deleted_rois)
        h_del.addWidget(self.btn_del_rois)
        h_del.addWidget(self.btn_restore_rois)
        lo.addLayout(h_del)

        # Fill Mode and Treatment live with Bounding Boxes because they
        # only affect bbox-derived cylinders.
        h_fill = QtWidgets.QHBoxLayout()
        h_fill.addWidget(QtWidgets.QLabel("Fill Mode:"))
        self.combo_mode = QtWidgets.QComboBox()
        self.combo_mode.addItems(["Strict", "Center", "Coverage"])
        self.combo_mode.setCurrentIndex(1)
        self.combo_mode.currentTextChanged.connect(self.recalculate_points)
        h_fill.addWidget(self.combo_mode)
        lo.addLayout(h_fill)

        h_treat = QtWidgets.QHBoxLayout()
        h_treat.addWidget(QtWidgets.QLabel("Treat boxes as:"))
        self.combo_treatment = QtWidgets.QComboBox()
        self.combo_treatment.addItems(["Single volume", "Separate boxes"])
        self.combo_treatment.setCurrentIndex(0)
        self.combo_treatment.setToolTip(
            "Single volume: one global grid spans the union of all boxes "
            "(adjacent boxes share cylinders).\n"
            "Separate boxes: each box gets its own grid (cylinders never "
            "cross box boundaries; some redundancy at overlaps)."
        )
        self.combo_treatment.currentTextChanged.connect(self.recalculate_points)
        h_treat.addWidget(self.combo_treatment)
        lo.addLayout(h_treat)

        grp.setLayout(lo)
        # Update gating when selection or list contents change.
        self.roi_list_widget.itemSelectionChanged.connect(self._update_action_states)
        return grp

    def _create_line_content(self):
        widget = QtWidgets.QWidget()
        lo = QtWidgets.QVBoxLayout(widget)
        lo.setContentsMargins(10, 5, 10, 5)
        lo.setSpacing(2)

        info = QtWidgets.QLabel("Define a line by two points; cylinders will tile along it without overlap.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        lo.addWidget(info)

        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(QtWidgets.QLabel("P1 (x,y,z):"))
        self.txt_line_p1 = QtWidgets.QLineEdit()
        self.txt_line_p1.setPlaceholderText("e.g. 100, 100, 50")
        h1.addWidget(self.txt_line_p1)
        lo.addLayout(h1)

        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(QtWidgets.QLabel("P2 (x,y,z):"))
        self.txt_line_p2 = QtWidgets.QLineEdit()
        self.txt_line_p2.setPlaceholderText("e.g. 400, 300, 50")
        h2.addWidget(self.txt_line_p2)
        lo.addLayout(h2)

        h3 = QtWidgets.QHBoxLayout()
        b_add = QtWidgets.QPushButton("Add Line")
        b_add.clicked.connect(self.add_line_from_text)
        b_clr = QtWidgets.QPushButton("Clear Inputs")
        b_clr.clicked.connect(lambda: (self.txt_line_p1.clear(), self.txt_line_p2.clear()))
        h3.addWidget(b_add)
        h3.addWidget(b_clr)
        lo.addLayout(h3)

        h_dens = QtWidgets.QHBoxLayout()
        h_dens.addWidget(QtWidgets.QLabel("Density:"))
        self.spin_line_density = QtWidgets.QDoubleSpinBox()
        self.spin_line_density.setRange(0.1, 10.0)
        self.spin_line_density.setSingleStep(0.1)
        self.spin_line_density.setDecimals(2)
        self.spin_line_density.setValue(self.line_density)
        self.spin_line_density.setToolTip(
            "1.0 = cylinders touching, no overlap\n"
            ">1.0 = closer/overlapping (denser coverage)\n"
            "<1.0 = sparser, with gaps")
        self.spin_line_density.valueChanged.connect(self._on_line_density_changed)
        h_dens.addWidget(self.spin_line_density)
        h_dens.addWidget(QtWidgets.QLabel("(1.0 = touching)"))
        lo.addLayout(h_dens)

        self.chk_show_lines = QtWidgets.QCheckBox("Show Line Coverage Cylinders")
        self.chk_show_lines.setChecked(True)
        self.chk_show_lines.toggled.connect(self.update_3d_scene)
        lo.addWidget(self.chk_show_lines)

        self.line_list_widget = QtWidgets.QListWidget()
        self.line_list_widget.itemChanged.connect(self.on_line_item_changed)
        self.line_list_widget.itemDoubleClicked.connect(self._on_line_double_clicked)
        lo.addWidget(self.line_list_widget)

        h_del = QtWidgets.QHBoxLayout()
        self.btn_del_lines = QtWidgets.QPushButton("Delete")
        self.btn_del_lines.setEnabled(False)
        self.btn_del_lines.clicked.connect(self.delete_selected_lines)
        self.btn_restore_lines = QtWidgets.QPushButton("Undo delete")
        self.btn_restore_lines.setEnabled(False)
        self.btn_restore_lines.clicked.connect(self.restore_last_deleted_lines)
        h_del.addWidget(self.btn_del_lines)
        h_del.addWidget(self.btn_restore_lines)
        lo.addLayout(h_del)
        self.line_list_widget.itemSelectionChanged.connect(self._update_action_states)
        return widget

    def add_line_from_text(self):
        try:
            p1_parts = [float(x) for x in self.txt_line_p1.text().replace(',', ' ').split()]
            p2_parts = [float(x) for x in self.txt_line_p2.text().replace(',', ' ').split()]
        except ValueError:
            self.statusBar().showMessage("Line points must be numeric: x y z", 5000)
            return
        if len(p1_parts) != 3 or len(p2_parts) != 3:
            self.statusBar().showMessage("Each line point needs exactly 3 numbers (x, y, z)", 5000)
            return
        p1 = tuple(p1_parts)
        p2 = tuple(p2_parts)
        if p1 == p2:
            self.statusBar().showMessage("P1 and P2 must differ", 5000)
            return
        self.lines.append((p1, p2))
        self.active_line_mask.append(True)
        self.txt_line_p1.clear()
        self.txt_line_p2.clear()
        self.refresh_line_list()
        self.recalculate_line_points()

    def delete_selected_lines(self):
        rows = sorted({item.row() for item in self.line_list_widget.selectedIndexes()}, reverse=True)
        if not rows:
            return
        deleted = []
        for i in rows:
            deleted.append((self.lines[i], bool(self.active_line_mask[i])))
            del self.lines[i]
            del self.active_line_mask[i]
        self._stash_lines = list(deleted)
        if hasattr(self, 'btn_restore_lines'):
            self.btn_restore_lines.setEnabled(True)
        self.refresh_line_list()
        self.recalculate_line_points()

    def restore_last_deleted_lines(self):
        if not self._stash_lines:
            return
        for line, active in reversed(self._stash_lines):
            self.lines.append(line)
            self.active_line_mask.append(active)
        self._stash_lines = None
        if hasattr(self, 'btn_restore_lines'):
            self.btn_restore_lines.setEnabled(False)
        self.refresh_line_list()
        self.recalculate_line_points()

    def refresh_line_list(self):
        self.line_list_widget.blockSignals(True)
        self.line_list_widget.clear()
        for i, (p1, p2) in enumerate(self.lines):
            label = (f"Line #{i}: "
                     f"({int(p1[0])},{int(p1[1])},{int(p1[2])}) -> "
                     f"({int(p2[0])},{int(p2[1])},{int(p2[2])})")
            item = QtWidgets.QListWidgetItem(label)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if self.active_line_mask[i] else QtCore.Qt.Unchecked)
            self.line_list_widget.addItem(item)
        self.line_list_widget.blockSignals(False)
        self._update_action_states()

    def on_line_item_changed(self, item):
        self.active_line_mask[self.line_list_widget.row(item)] = (item.checkState() == QtCore.Qt.Checked)
        self.recalculate_line_points()

    def recalculate_line_points(self):
        active_lines = [ln for ln, ok in zip(self.lines, self.active_line_mask) if ok]
        pts, dims_std, dims_exp = solve_line_coverage(
            active_lines, self.current_scan_res, self.cfg, density=self.line_density
        )
        self.line_points = pts
        self.active_line_cyl_mask = np.ones(len(pts), dtype=bool)
        # If there are no ROIs, line coverage drives cylinder dims.
        if not self.rois:
            self.dims_std = dims_std
            self.dims_exp = dims_exp
        if hasattr(self, 'cyl_list_widget'):
            self.refresh_cyl_list()
        self.update_3d_scene()

    def _on_line_density_changed(self, val):
        self.line_density = float(val)
        self.recalculate_line_points()

    def _on_line_double_clicked(self, item):
        i = self.line_list_widget.row(item)
        if 0 <= i < len(self.lines):
            p1, p2 = self.lines[i]
            self.txt_line_p1.setText(
                f"{int(p1[0])}, {int(p1[1])}, {int(p1[2])}"
            )
            self.txt_line_p2.setText(
                f"{int(p2[0])}, {int(p2[1])}, {int(p2[2])}"
            )

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
        self.chk_show_manual.toggled.connect(self._on_show_manual_toggled)
        lo.addWidget(self.chk_show_manual)

        info = QtWidgets.QLabel("Manual cylinders appear in the unified Cylinders list below.")
        info.setStyleSheet("color: #555;")
        info.setWordWrap(True)
        lo.addWidget(info)
        return widget

    def _create_auto_grid_group(self):
        grp = QtWidgets.QGroupBox("Cylinders (Auto / Line / Manual)")
        lo = QtWidgets.QVBoxLayout()

        self.lbl_cyl_dims = QtWidgets.QLabel("Cylinder ⌀ × H: —")
        self.lbl_cyl_dims.setStyleSheet("color: #555; font-size: 10pt;")
        self.lbl_cyl_dims.setWordWrap(True)
        lo.addWidget(self.lbl_cyl_dims)

        self.cyl_list_widget = QtWidgets.QListWidget()
        self.cyl_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.cyl_list_widget.itemChanged.connect(self.on_combined_item_changed)
        lo.addWidget(self.cyl_list_widget)

        h = QtWidgets.QHBoxLayout()
        ba = QtWidgets.QPushButton("All")
        ba.clicked.connect(lambda: self.set_all_cyls(True))
        bn = QtWidgets.QPushButton("None")
        bn.clicked.connect(lambda: self.set_all_cyls(False))
        self.btn_del_manual = QtWidgets.QPushButton("Delete (M)")
        self.btn_del_manual.setEnabled(False)
        self.btn_del_manual.setToolTip("Delete selected manual cylinders. Auto and Line cylinders are derived and not user-deletable here.")
        self.btn_del_manual.clicked.connect(self.delete_selected_combined)
        self.btn_restore_manual = QtWidgets.QPushButton("Undo delete")
        self.btn_restore_manual.setEnabled(False)
        self.btn_restore_manual.clicked.connect(self.restore_last_deleted_manual)
        h.addWidget(ba)
        h.addWidget(bn)
        h.addWidget(self.btn_del_manual)
        h.addWidget(self.btn_restore_manual)
        lo.addLayout(h)
        self.cyl_list_widget.itemSelectionChanged.connect(self._update_action_states)
        grp.setLayout(lo)
        return grp

    def _create_actions_group(self):
        grp = QtWidgets.QGroupBox("Export & Registration")
        lo = QtWidgets.QVBoxLayout()

        self.btn_export_nml = QtWidgets.QPushButton("EXPORT NML (TILES)")
        self.btn_export_nml.clicked.connect(self.export_nml_tiles)
        self.btn_export_nml.setStyleSheet("background-color: #f0ad4e; color: white;")
        lo.addWidget(self.btn_export_nml)

        self.btn_print_console = QtWidgets.QPushButton("PRINT CONSOLE")
        self.btn_print_console.clicked.connect(self.export_coordinates)
        self.btn_print_console.setStyleSheet("background-color: blue; color: white;")
        lo.addWidget(self.btn_print_console)

        self.btn_register = QtWidgets.QPushButton("REGISTER COORDINATES")
        self.btn_register.clicked.connect(self.open_registration_dialog)
        self.btn_register.setStyleSheet("background-color: purple; color: white; font-weight: bold; padding: 6px;")
        lo.addWidget(self.btn_register)

        grp.setLayout(lo)
        return grp

    # Default Vol Opacity Max per blending mode. Average blending averages
    # all sample contributions, so its useful range is much smaller than
    # Composite's; MIP/MinIP collapse along the ray, so they sit between.
    _VOL_MAX_DEFAULTS = {
        "composite": 1.0,
        "maximum": 0.1,
        "minimum": 0.1,
        "average": 0.01,
    }

    def _on_grid_unit_changed(self, _):
        """Show/hide spacing controls based on the selected unit."""
        unit = self.combo_grid_unit.currentText()
        show_um = unit in ("µm", "Both")
        show_px = unit in ("px", "Both")
        self.lbl_grid_um.setVisible(show_um)
        self.spin_grid_spacing.setVisible(show_um)
        self.lbl_grid_px.setVisible(show_px)
        self.spin_grid_spacing_px.setVisible(show_px)
        self._refresh_reference_grid()

    def _build_ref_grid_polydata(self, spacing_px, ext_x, ext_y, ext_z, z_ratio=1.0):
        """Return a pv.PolyData with line segments forming a 3D wireframe
        cage: gridlines on each of the 6 faces of [0, ext_x] x [0, ext_y]
        x [0, ext_z * z_ratio]. Adjacent faces share their corner verts via
        tick crossings — the result reads as a 3D grid through the volume.

        z_ratio scales z values into display space (matches the rest of the
        renderer which multiplies prescan z by self.z_ratio).
        """
        if spacing_px <= 0 or ext_x <= 0 or ext_y <= 0 or ext_z <= 0:
            return None
        zmax = ext_z * z_ratio

        x_ticks = [min(i * spacing_px, ext_x) for i in range(int(ext_x // spacing_px) + 1)]
        y_ticks = [min(j * spacing_px, ext_y) for j in range(int(ext_y // spacing_px) + 1)]
        z_ticks_real = [min(k * spacing_px, ext_z) for k in range(int(ext_z // spacing_px) + 1)]
        z_ticks = [zt * z_ratio for zt in z_ticks_real]

        if not x_ticks or not y_ticks or not z_ticks:
            return None

        pts = []
        lines = []
        def seg(a, b):
            i0 = len(pts); pts.append(a); pts.append(b)
            lines.extend([2, i0, i0 + 1])

        # XY faces (z = 0 and z = zmax) — full grid
        for z_face in (0.0, zmax):
            for y in y_ticks:
                seg([0.0, y, z_face], [ext_x, y, z_face])
            for x in x_ticks:
                seg([x, 0.0, z_face], [x, ext_y, z_face])
        # XZ faces (y = 0 and y = ext_y)
        for y_face in (0.0, ext_y):
            for z in z_ticks:
                seg([0.0, y_face, z], [ext_x, y_face, z])
            for x in x_ticks:
                seg([x, y_face, 0.0], [x, y_face, zmax])
        # YZ faces (x = 0 and x = ext_x)
        for x_face in (0.0, ext_x):
            for z in z_ticks:
                seg([x_face, 0.0, z], [x_face, ext_y, z])
            for y in y_ticks:
                seg([x_face, y, 0.0], [x_face, y, zmax])

        poly = pv.PolyData()
        poly.points = np.array(pts, dtype=float)
        poly.lines = np.array(lines, dtype=int)
        return poly

    def _build_ref_grid_label_data(self, spacing_px, ext_x, ext_y, ext_z, value_unit, z_ratio=1.0):
        """Return (point_array, label_strings) for tick labels on the
        front-bottom edges of the cage — one per X tick (on the y=0,
        z=0 edge), Y tick (x=0, z=0), Z tick (x=0, y=0).

        *value_unit* is "µm" or "px" — labels are formatted in that unit.
        """
        zmax = ext_z * z_ratio
        # Sub-sampling: if there are too many ticks, label every Nth so the
        # scene doesn't drown in text.
        def thin(positions, max_labels=12):
            n = len(positions)
            if n <= max_labels:
                return list(range(n))
            step = max(1, n // max_labels)
            return list(range(0, n, step))

        px_xy_nm = float(self.cfg.get('prescan_pixel_size_xy', 150)) or 150.0
        px_z_nm = float(self.cfg.get('prescan_z_step', 150)) or 150.0

        # VTK string arrays are ASCII-only — use "um" (not "µm") in the
        # actual label text. The Qt UI elements that say "µm" stay as-is
        # because Qt handles Unicode fine.
        def fmt_xy(value_px):
            if value_unit == "µm":
                return f"{value_px * px_xy_nm / 1000.0:g} um"
            return f"{int(round(value_px))} px"

        def fmt_z(value_px):
            if value_unit == "µm":
                return f"{value_px * px_z_nm / 1000.0:g} um"
            return f"{int(round(value_px))} px"

        x_ticks = [min(i * spacing_px, ext_x) for i in range(int(ext_x // spacing_px) + 1)]
        y_ticks = [min(j * spacing_px, ext_y) for j in range(int(ext_y // spacing_px) + 1)]
        z_ticks_real = [min(k * spacing_px, ext_z) for k in range(int(ext_z // spacing_px) + 1)]

        labels = []
        positions = []
        for i in thin(x_ticks):
            x = x_ticks[i]
            positions.append([x, -spacing_px * 0.3, 0.0])
            labels.append(fmt_xy(x))
        for i in thin(y_ticks):
            y = y_ticks[i]
            positions.append([-spacing_px * 0.3, y, 0.0])
            labels.append(fmt_xy(y))
        for i in thin(z_ticks_real):
            zr = z_ticks_real[i]
            positions.append([-spacing_px * 0.3, 0.0, zr * z_ratio])
            labels.append(fmt_z(zr))

        return np.array(positions, dtype=float) if positions else None, labels

    def _refresh_reference_grid(self):
        """Draw or remove the reference grid based on the Appearance controls.

        The grid is a 3D wireframe cage at the volume bounds, with grid
        lines on each of the 6 faces at the chosen spacing, and value
        labels on the front-bottom X / Y / Z tick edges.

        Supports three unit modes:
          - "µm":   one grid, spacing entered in µm (converted via
                    prescan_pixel_size_xy).
          - "px":   one grid, spacing entered directly in prescan pixels.
          - "Both": two distinguishable grids overlaid (µm in light grey,
                    px in cyan) so you can read both scales at once.
        """
        if not hasattr(self, 'plotter') or self.plotter is None:
            return
        # Remove previous actors first; spacing/opacity/unit changes require
        # a full rebuild.
        for act in self.ref_grid_actors:
            try:
                self.plotter.remove_actor(act, reset_camera=False)
            except Exception:
                pass
        self.ref_grid_actors = []

        if not getattr(self, 'chk_ref_grid', None) or not self.chk_ref_grid.isChecked():
            self.plotter.render()
            return

        # Cage extents (prescan pixel coordinates).
        if self.vol_grid is not None:
            ext_x = self.max_dims[0] if self.max_dims[0] > 0 else 1000
            ext_y = self.max_dims[1] if self.max_dims[1] > 0 else 1000
            ext_z = self.max_dims[2] if self.max_dims[2] > 0 else 1000
        elif self.rois:
            ext_x = max(r['x'] + r['w'] for r in self.rois)
            ext_y = max(r['y'] + r['h'] for r in self.rois)
            ext_z = max(r['z'] + r['d'] for r in self.rois)
        else:
            ext_x = ext_y = ext_z = 1000

        unit = self.combo_grid_unit.currentText() if hasattr(self, 'combo_grid_unit') else "µm"
        opacity = max(0.0, min(1.0, self.slider_grid_op.value() / 100.0))
        px_xy_nm = float(self.cfg.get('prescan_pixel_size_xy', 150)) or 150.0

        # (label_unit, color, spacing_px) tuples — one per concurrent grid.
        grids = []
        if unit in ("µm", "Both"):
            spacing_um = float(self.spin_grid_spacing.value())
            spacing_px = max(1.0, spacing_um * 1000.0 / px_xy_nm)
            grids.append(("µm", "#bbbbbb", spacing_px))
        if unit in ("px", "Both"):
            spacing_px = float(self.spin_grid_spacing_px.value())
            grids.append(("px", "#33ccff", spacing_px))

        for label_unit, color, spacing_px in grids:
            poly = self._build_ref_grid_polydata(
                spacing_px, ext_x, ext_y, ext_z, z_ratio=self.z_ratio
            )
            if poly is None:
                continue
            actor = self.plotter.add_mesh(
                poly, color=color, line_width=1, opacity=opacity,
                lighting=False, reset_camera=False, pickable=False,
            )
            self.ref_grid_actors.append(actor)

            # Tick value labels along the front-bottom X / Y / Z edges.
            label_pts, label_strs = self._build_ref_grid_label_data(
                spacing_px, ext_x, ext_y, ext_z, label_unit, z_ratio=self.z_ratio
            )
            if label_pts is None or len(label_pts) == 0:
                continue
            label_poly = pv.PolyData(label_pts)
            label_poly["labels"] = label_strs
            try:
                lbl_actor = self.plotter.add_point_labels(
                    label_poly, "labels",
                    font_size=10, text_color=color, shadow=True,
                    show_points=False, always_visible=True,
                    shape_opacity=0.0, reset_camera=False,
                )
                self.ref_grid_actors.append(lbl_actor)
            except Exception:
                pass

        self.plotter.render()

    def _scale_vol_max(self, factor):
        try:
            cur = float(self.txt_vol_max.text())
        except ValueError:
            cur = 1.0
        new_val = cur * factor
        # Avoid drifting to absurd values; clamp to a sensible window.
        if new_val < 1e-6:
            new_val = 1e-6
        elif new_val > 1e6:
            new_val = 1e6
        # Keep the field tidy: scientific notation when the value is far
        # from 1 in either direction, plain otherwise.
        if new_val >= 100 or (0 < new_val < 0.01):
            txt = f"{new_val:.2e}"
        else:
            txt = f"{new_val:g}"
        self.txt_vol_max.setText(txt)
        self.update_opacity()

    def update_volume_render_mode(self):
        if self.vol_grid is None: return
        if self.vol_actor is not None:
            self.plotter.remove_actor(self.vol_actor, reset_camera=False)
            self.vol_actor = None

        mode_str = self.combo_render.currentText()
        blend_mode = "composite"
        if "MIP" in mode_str: blend_mode = "maximum"
        elif "MinIP" in mode_str: blend_mode = "minimum"
        elif "Average" in mode_str: blend_mode = "average"
        elif "Additive" in mode_str: blend_mode = "additive"

        # Auto-set Vol Opacity Max on mode change so the volume is visible
        # without manual fiddling. The user can still override afterwards.
        default_max = self._VOL_MAX_DEFAULTS.get(blend_mode, 1.0)
        self.txt_vol_max.setText(f"{default_max:g}")

        cmap = "gray_r" if (
            hasattr(self, 'chk_invert_volume') and self.chk_invert_volume.isChecked()
        ) else "gray"
        self.vol_actor = self.plotter.add_volume(
            self.vol_grid, cmap=cmap, clim=self.clim,
            opacity="linear", blending=blend_mode, reset_camera=False,
        )
        # Re-add cylinder/ROI/label actors on top so the new volume actor's
        # shader state doesn't clobber their colors. update_3d_scene also
        # invokes _apply_cylinder_blend_compensation, so cylinder material
        # properties match the new blend mode.
        self.update_3d_scene()

    def export_nml_tiles(self):
        pts = self.get_all_active_points()
        if len(pts) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No active cylinders to export.")
            return

        options = QtWidgets.QFileDialog.Options()
        start_dir = self.cfg.get('last_nml_save_dir', '')
        seed = os.path.join(start_dir, "tiles.nml") if start_dir else "tiles.nml"
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save NML Tiles", seed, "NML Files (*.nml)", options=options)
        if fileName:
            self._persist_session_pref('last_nml_save_dir', os.path.dirname(fileName))
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
        self.refresh_cyl_list()
        self.update_3d_scene()
        self.txt_man_input.clear()

    def refresh_manual_list(self):
        # Kept for backward-compat callers; manual cylinders now show in the
        # unified cylinder list, so just rebuild that.
        self.refresh_cyl_list()

    def _on_show_manual_toggled(self, _):
        self.refresh_cyl_list()
        self.update_3d_scene()

    def get_all_active_points(self):
        final_points = []
        idx_auto = np.where(self.active_mask)[0]
        if len(idx_auto) > 0:
            final_points.append(self.all_points[idx_auto])
        if (getattr(self, 'chk_show_lines', None) and self.chk_show_lines.isChecked()
                and len(self.line_points) > 0):
            mask = self.active_line_cyl_mask
            if len(mask) == len(self.line_points):
                idx_line = np.where(mask)[0]
                if len(idx_line) > 0:
                    final_points.append(self.line_points[idx_line])
            else:
                final_points.append(self.line_points)
        if self.chk_show_manual.isChecked() and len(self.manual_points) > 0:
            man_active = [p for p, a in zip(self.manual_points, self.active_manual_mask) if a]
            if len(man_active) > 0:
                final_points.append(np.array(man_active))

        if len(final_points) == 0: return np.empty((0,3))
        return np.vstack(final_points)

    def open_registration_dialog(self):
        self.reg_dialog = RegistrationDialog(self)
        self.reg_dialog.in_final_px.setValue(float(self.current_scan_res))
        self.reg_dialog.show()

    def load_nml_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        start_dir = self.cfg.get('last_nml_load_dir', '')
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open NML", start_dir, "NML (*.nml)", options=options)
        if fileName:
            self._persist_session_pref('last_nml_load_dir', os.path.dirname(fileName))
            new = parse_nml(fileName)
            if new:
                self.rois = new
                self.refresh_roi_list()
                self.recalculate_points()

    def update_resolution(self):
        new_res = float(self.spin_res.value())
        if new_res != self.current_scan_res:
            self.current_scan_res = new_res
            self.recalculate_points()

    def nudge_rois(self, ax, d):
        s = self.spin_step.value()
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
        self._update_action_states()

    def _on_roi_double_clicked(self, item):
        i = self.roi_list_widget.row(item)
        if 0 <= i < len(self.rois):
            r = self.rois[i]
            self.txt_roi.setText(
                f"{r['x']},{r['y']},{r['z']},{r['w']},{r['h']},{r['d']}"
            )

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
        rows = sorted({item.row() for item in self.roi_list_widget.selectedIndexes()}, reverse=True)
        if not rows:
            return
        deleted = []
        for i in rows:
            deleted.append(dict(self.rois[i]))
            del self.rois[i]
        # Push to stash for restore (most-recent-first within the batch).
        self._stash_rois = list(deleted)
        if hasattr(self, 'btn_restore_rois'):
            self.btn_restore_rois.setEnabled(True)
        self.refresh_roi_list()
        self.recalculate_points()

    def restore_last_deleted_rois(self):
        if not self._stash_rois:
            return
        # Append back at the end; positional ordering after restore isn't
        # critical because solve_global_union doesn't depend on it.
        self.rois.extend(reversed(self._stash_rois))
        self._stash_rois = None
        if hasattr(self, 'btn_restore_rois'):
            self.btn_restore_rois.setEnabled(False)
        self.refresh_roi_list()
        self.recalculate_points()

    def recalculate_points(self):
        mode = self.combo_mode.currentText().lower()
        old_count = len(self.all_points)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            treatment = "separate" if (
                hasattr(self, 'combo_treatment') and self.combo_treatment.currentIndex() == 1
            ) else "union"
            self.all_points, self.dims_std, self.dims_exp = solve_bbox_grids(
                self.rois, self.current_scan_res, self.cfg, mode, treatment
            )
            active_lines = [ln for ln, ok in zip(self.lines, self.active_line_mask) if ok]
            self.line_points, _, _ = solve_line_coverage(
                active_lines, self.current_scan_res, self.cfg, density=self.line_density
            )
            self.active_line_cyl_mask = np.ones(len(self.line_points), dtype=bool)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        new_count = len(self.all_points)
        if new_count != old_count or len(self.active_mask) != new_count:
            self.active_mask = np.ones(new_count, dtype=bool)
        self._update_cyl_dims_label()
        self.refresh_cyl_list()
        self.update_3d_scene()
        self._update_action_states()

    def _update_action_states(self):
        """Drive enabled/disabled state of action buttons from current data.

        Called from recalculate_points / refresh_cyl_list / list selection
        changes. Centralizes the gating logic so failure modes are visible
        as greyed-out buttons rather than post-click warnings.
        """
        # Export / Print: need at least one active cylinder anywhere.
        try:
            n_active = len(self.get_all_active_points())
        except Exception:
            n_active = 0
        if hasattr(self, 'btn_export_nml'):
            self.btn_export_nml.setEnabled(n_active > 0)
        if hasattr(self, 'btn_print_console'):
            self.btn_print_console.setEnabled(n_active > 0)
        if hasattr(self, 'btn_register'):
            # Registration always opens; the dialog itself gates Calculate.
            self.btn_register.setEnabled(n_active > 0)

        # Reset Shift: only when a shift has been applied.
        if hasattr(self, 'btn_reset_shift'):
            self.btn_reset_shift.setEnabled(self.total_roi_shift != [0, 0, 0])

        # Delete Selected (ROI/Line/Manual): need a selection of the right kind.
        if hasattr(self, 'btn_del_rois'):
            self.btn_del_rois.setEnabled(
                len(self.roi_list_widget.selectedIndexes()) > 0
            )
        if hasattr(self, 'btn_del_lines'):
            self.btn_del_lines.setEnabled(
                len(self.line_list_widget.selectedIndexes()) > 0
            )
        if hasattr(self, 'btn_del_manual'):
            has_manual_sel = any(
                (item.data(QtCore.Qt.UserRole) or (None,))[0] == 'manual'
                for item in self.cyl_list_widget.selectedItems()
            )
            self.btn_del_manual.setEnabled(has_manual_sel)

    def _persist_session_pref(self, key, value):
        """Set *key* on self.cfg and write back to the user config file."""
        self.cfg[key] = value
        if self.config_path:
            update_user_config_keys(self.config_path, {key: value})

    def _update_cyl_dims_label(self):
        if not hasattr(self, 'lbl_cyl_dims'):
            return
        d_std, h_std = self.dims_std
        d_exp, h_exp = self.dims_exp
        px_xy = float(self.cfg.get('prescan_pixel_size_xy', 0))
        px_z = float(self.cfg.get('prescan_z_step', 0))
        if px_xy <= 0 or px_z <= 0 or d_std <= 0:
            self.lbl_cyl_dims.setText("Cylinder ⌀ × H: —")
            return
        # Convert prescan-pixel dims to micrometers.
        d_std_um = d_std * px_xy / 1000.0
        h_std_um = h_std * px_z / 1000.0
        d_exp_um = d_exp * px_xy / 1000.0
        h_exp_um = h_exp * px_z / 1000.0
        self.lbl_cyl_dims.setText(
            f"Cylinder ⌀ × H:  std {d_std_um:.1f} × {h_std_um:.1f} µm   "
            f"exp {d_exp_um:.1f} × {h_exp_um:.1f} µm"
        )

    SECTION_BG = QtGui.QColor(60, 60, 70)
    SECTION_FG = QtGui.QColor(255, 255, 255)

    def _make_section_header(self, label, all_checked, kind):
        item = QtWidgets.QListWidgetItem(f"▼ {label}")
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        item.setBackground(self.SECTION_BG)
        item.setForeground(self.SECTION_FG)
        item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked if all_checked else QtCore.Qt.Unchecked)
        item.setData(QtCore.Qt.UserRole, ('section', kind))
        return item

    def _make_cyl_item(self, label, kind, idx, checked, deletable=False):
        item = QtWidgets.QListWidgetItem("    " + label)
        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable
        item.setFlags(flags)
        item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        item.setData(QtCore.Qt.UserRole, (kind, idx, bool(deletable)))
        return item

    def refresh_cyl_list(self):
        if not hasattr(self, 'cyl_list_widget'):
            return
        self.cyl_list_widget.blockSignals(True)
        self.cyl_list_widget.clear()

        n_auto = len(self.all_points)
        auto_states = [bool(self.active_mask[i]) if i < len(self.active_mask) else True
                       for i in range(n_auto)]
        auto_active = sum(auto_states)
        auto_all = bool(n_auto) and all(auto_states)
        self.cyl_list_widget.addItem(
            self._make_section_header(
                f"Auto from BBoxes — {auto_active} / {n_auto} active", auto_all, 'auto'
            )
        )
        seq = 0
        for i, p in enumerate(self.all_points):
            checked = auto_states[i]
            tag = f"A{seq}" if checked else "---"
            self.cyl_list_widget.addItem(
                self._make_cyl_item(f"{tag}  (Orig {i}): {p.astype(int)}", 'auto', i, checked)
            )
            if checked:
                seq += 1

        n_line = len(self.line_points)
        line_show = (getattr(self, 'chk_show_lines', None) is None
                     or self.chk_show_lines.isChecked())
        line_states = [bool(self.active_line_cyl_mask[i]) if i < len(self.active_line_cyl_mask) else True
                       for i in range(n_line)]
        line_active = sum(s and line_show for s in line_states)
        line_all = line_show and bool(n_line) and all(line_states)
        self.cyl_list_widget.addItem(
            self._make_section_header(
                f"Line Coverage — {line_active} / {n_line} active", line_all, 'line'
            )
        )
        seq = 0
        for i, p in enumerate(self.line_points):
            checked = line_show and line_states[i]
            tag = f"L{seq}" if checked else "---"
            self.cyl_list_widget.addItem(
                self._make_cyl_item(f"{tag}: {p.astype(int)}", 'line', i, checked)
            )
            if checked:
                seq += 1

        n_man = len(self.manual_points)
        man_show = self.chk_show_manual.isChecked() if hasattr(self, 'chk_show_manual') else True
        man_states = [bool(self.active_manual_mask[i]) for i in range(n_man)]
        man_active = sum(s and man_show for s in man_states)
        man_all = man_show and bool(n_man) and all(man_states)
        self.cyl_list_widget.addItem(
            self._make_section_header(
                f"Manual — {man_active} / {n_man} active", man_all, 'manual'
            )
        )
        seq = 0
        for i, p in enumerate(self.manual_points):
            checked = man_show and man_states[i]
            tag = f"M{seq}" if checked else "---"
            self.cyl_list_widget.addItem(
                self._make_cyl_item(f"{tag}: {p.astype(int)}", 'manual', i, checked, deletable=True)
            )
            if checked:
                seq += 1

        self.cyl_list_widget.blockSignals(False)
        self._update_action_states()

    def on_combined_item_changed(self, item):
        data = item.data(QtCore.Qt.UserRole)
        if not data:
            return
        checked = (item.checkState() == QtCore.Qt.Checked)
        kind = data[0]
        if kind == 'section':
            section = data[1]
            if section == 'auto':
                self.active_mask[:] = checked
            elif section == 'line':
                if hasattr(self, 'chk_show_lines'):
                    self.chk_show_lines.blockSignals(True)
                    self.chk_show_lines.setChecked(checked)
                    self.chk_show_lines.blockSignals(False)
                if len(self.active_line_cyl_mask):
                    self.active_line_cyl_mask[:] = checked
            elif section == 'manual':
                self.chk_show_manual.blockSignals(True)
                self.chk_show_manual.setChecked(checked)
                self.chk_show_manual.blockSignals(False)
                self.active_manual_mask = [checked] * len(self.active_manual_mask)
            self.refresh_cyl_list()
            self.update_3d_scene()
            return

        idx = data[1]
        if kind == 'auto':
            if 0 <= idx < len(self.active_mask):
                self.active_mask[idx] = checked
        elif kind == 'line':
            if 0 <= idx < len(self.active_line_cyl_mask):
                self.active_line_cyl_mask[idx] = checked
        elif kind == 'manual':
            if 0 <= idx < len(self.active_manual_mask):
                self.active_manual_mask[idx] = checked
        self.refresh_cyl_list()
        self.update_3d_scene()

    def set_all_cyls(self, s):
        if len(self.active_mask):
            self.active_mask[:] = s
        if len(self.active_line_cyl_mask):
            self.active_line_cyl_mask[:] = s
        self.active_manual_mask = [s] * len(self.active_manual_mask)
        if hasattr(self, 'chk_show_lines'):
            self.chk_show_lines.blockSignals(True)
            self.chk_show_lines.setChecked(s)
            self.chk_show_lines.blockSignals(False)
        self.chk_show_manual.blockSignals(True)
        self.chk_show_manual.setChecked(s)
        self.chk_show_manual.blockSignals(False)
        self.refresh_cyl_list()
        self.update_3d_scene()

    def delete_selected_combined(self):
        # Only manual cylinders are user-deletable. Auto cylinders are
        # derived from ROIs and line cylinders from lines.
        manual_indices = []
        for item in self.cyl_list_widget.selectedItems():
            data = item.data(QtCore.Qt.UserRole)
            if data and data[0] == 'manual':
                manual_indices.append(data[1])
        if not manual_indices:
            self.statusBar().showMessage(
                "Only manual cylinders can be deleted. Use the Bounding Boxes/Line lists for the others.",
                5000,
            )
            return
        deleted = []
        for i in sorted(set(manual_indices), reverse=True):
            if 0 <= i < len(self.manual_points):
                deleted.append((np.array(self.manual_points[i], copy=True),
                                bool(self.active_manual_mask[i])))
                del self.manual_points[i]
                del self.active_manual_mask[i]
        self._stash_manual = list(deleted)
        if hasattr(self, 'btn_restore_manual'):
            self.btn_restore_manual.setEnabled(bool(deleted))
        self.refresh_cyl_list()
        self.update_3d_scene()

    def restore_last_deleted_manual(self):
        if not self._stash_manual:
            return
        for pt, active in reversed(self._stash_manual):
            self.manual_points.append(pt)
            self.active_manual_mask.append(active)
        self._stash_manual = None
        if hasattr(self, 'btn_restore_manual'):
            self.btn_restore_manual.setEnabled(False)
        self.refresh_cyl_list()
        self.update_3d_scene()

    def update_3d_scene(self):
        self.plotter.suppress_rendering = True
        if self.actor_std: self.plotter.remove_actor(self.actor_std, reset_camera=False)
        if self.actor_exp: self.plotter.remove_actor(self.actor_exp, reset_camera=False)
        if self.actor_man: self.plotter.remove_actor(self.actor_man, reset_camera=False)
        if self.actor_man_exp: self.plotter.remove_actor(self.actor_man_exp, reset_camera=False)
        if self.actor_line: self.plotter.remove_actor(self.actor_line, reset_camera=False)
        if self.actor_line_exp: self.plotter.remove_actor(self.actor_line_exp, reset_camera=False)
        if self.actor_labels: self.plotter.remove_actor(self.actor_labels, reset_camera=False)
        self.actor_std = None
        self.actor_exp = None
        self.actor_man = None
        self.actor_man_exp = None
        self.actor_line = None
        self.actor_line_exp = None
        self.actor_labels = None

        for act in self.roi_actors:
            self.plotter.remove_actor(act, reset_camera=False)
        self.roi_actors.clear()
        for act in self.line_actors:
            self.plotter.remove_actor(act, reset_camera=False)
        self.line_actors.clear()

        for r in self.rois:
            o = np.array([r['x'], r['y'], r['z']*self.z_ratio])
            va=np.array([r['w'], 0, 0])
            vb=np.array([0, r['h'], 0])
            vc=np.array([0, 0, r['d']*self.z_ratio])
            cube = pv.Cube(bounds=(o[0], o[0]+va[0], o[1], o[1]+vb[1], o[2], o[2]+vc[2]))
            act = self.plotter.add_mesh(cube, style='wireframe', color='cyan', line_width=2, lighting=False, reset_camera=False)
            self.roi_actors.append(act)

        # Draw the user-defined line segments (always visible, like ROI wireframes)
        for (p1, p2), active in zip(self.lines, self.active_line_mask):
            a = np.array([p1[0], p1[1], p1[2] * self.z_ratio])
            b = np.array([p2[0], p2[1], p2[2] * self.z_ratio])
            line_mesh = pv.Line(a, b)
            color = "orange" if active else "#666666"
            act = self.plotter.add_mesh(line_mesh, color=color, line_width=3, lighting=False, reset_camera=False)
            self.line_actors.append(act)
            pts_mesh = pv.PolyData(np.vstack([a, b]))
            act_pts = self.plotter.add_mesh(pts_mesh, color=color, point_size=10, render_points_as_spheres=True, lighting=False, reset_camera=False)
            self.line_actors.append(act_pts)

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
            self.actor_std = self.plotter.add_mesh(pv.PolyData(vp).glyph(geom=c1, scale=False), color='cyan', opacity=self.slider_cyl.value()/100, lighting=False, reset_camera=False)

            c2 = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=d_exp/2, height=vis_H_exp)
            self.actor_exp = self.plotter.add_mesh(pv.PolyData(vp).glyph(geom=c2, scale=False), color='magenta', opacity=self.slider_cyl.value()/100, lighting=False, reset_camera=False)

            for pt in vp:
                label_points.append(pt)
                label_ids.append(str(seq))
                seq += 1

        # Line-coverage cylinders (orange) — share dims with std cylinders
        if (getattr(self, 'chk_show_lines', None) and self.chk_show_lines.isChecked()
                and len(self.line_points) > 0):
            mask = self.active_line_cyl_mask
            if len(mask) == len(self.line_points):
                lp_src = self.line_points[np.where(mask)[0]]
            else:
                lp_src = self.line_points
            if len(lp_src) > 0:
                lp = lp_src.copy()
                lp[:, 2] *= self.z_ratio
                d_std, h_std = self.dims_std
                d_exp, h_exp = self.dims_exp
                if d_std > 0 and h_std > 0:
                    c_l = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                                      radius=d_std/2, height=h_std*self.z_ratio)
                    self.actor_line = self.plotter.add_mesh(
                        pv.PolyData(lp).glyph(geom=c_l, scale=False),
                        color='orange', opacity=self.slider_cyl.value()/100,
                        lighting=False, reset_camera=False)
                if d_exp > 0 and h_exp > 0:
                    c_le = pv.Cylinder(center=(0,0,0), direction=(0,0,1),
                                       radius=d_exp/2, height=h_exp*self.z_ratio)
                    self.actor_line_exp = self.plotter.add_mesh(
                        pv.PolyData(lp).glyph(geom=c_le, scale=False),
                        color='#ff6600', opacity=self.slider_cyl.value()/100,
                        lighting=False, reset_camera=False)
                for pt in lp:
                    label_points.append(pt)
                    label_ids.append(f"L{seq}")
                    seq += 1

        if self.chk_show_manual.isChecked() and len(self.manual_points) > 0:
            man_active = [p for p, a in zip(self.manual_points, self.active_manual_mask) if a]
            if len(man_active) > 0:
                mp = np.array(man_active).copy()
                mp[:, 2] *= self.z_ratio
                vis_H_std = self.dims_std[1] * self.z_ratio

                c_man = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=self.dims_std[0]/2, height=vis_H_std)
                self.actor_man = self.plotter.add_mesh(pv.PolyData(mp).glyph(geom=c_man, scale=False), color='yellow', opacity=self.slider_cyl.value()/100, lighting=False, reset_camera=False)

                vis_H_exp = self.dims_exp[1] * self.z_ratio
                c_man_exp = pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=self.dims_exp[0]/2, height=vis_H_exp)
                self.actor_man_exp = self.plotter.add_mesh(pv.PolyData(mp).glyph(geom=c_man_exp, scale=False), color='yellow', opacity=self.slider_cyl.value()/100, lighting=False, reset_camera=False)

                for pt in mp:
                    label_points.append(pt)
                    label_ids.append(f"M{seq}")
                    seq += 1

        if label_points:
            label_poly = pv.PolyData(np.array(label_points))
            label_poly["labels"] = label_ids
            self.actor_labels = self.plotter.add_point_labels(
                label_poly, "labels", font_size=12, text_color="white",
                shadow=True, show_points=False,
                always_visible=True, shape_opacity=0.4,
                reset_camera=False,
            )

        self.plotter.suppress_rendering = False

        self.update_visibility()
        self.update_opacity()
        self._apply_cylinder_blend_compensation()

    def _apply_cylinder_blend_compensation(self):
        """Tune cylinder material properties so they survive Average and
        Additive volume blends.

        VTK's ray integrator combines volume samples with surface fragments;
        in Average/Additive modes a translucent cyan cylinder gets averaged
        toward the volume's grey or summed into white-out. Forcing the
        actor's ambient up + diffuse down makes the cylinder read as a
        flat-shaded, self-illuminated patch that holds its color even when
        the integrator pushes neighbours toward grey.
        """
        if not hasattr(self, 'combo_render'):
            return
        mode_str = self.combo_render.currentText()
        is_avg = "Average" in mode_str
        is_add = "Additive" in mode_str
        # In Composite/MIP/MinIP, default phong material looks fine.
        ambient = 0.0; diffuse = 1.0; specular = 0.0
        if is_avg:
            # Average pulls everything toward mean grey — push cylinders
            # to almost-flat self-illumination so their hue dominates.
            ambient = 0.95; diffuse = 0.05; specular = 0.0
        elif is_add:
            # Additive sums; cylinders that were diffusely lit get washed
            # to white. Lower diffuse, modest ambient.
            ambient = 0.7; diffuse = 0.2; specular = 0.0

        for actor in (self.actor_std, self.actor_exp,
                      self.actor_man, self.actor_man_exp,
                      self.actor_line, self.actor_line_exp):
            if actor is None:
                continue
            try:
                prop = actor.GetProperty()
                prop.SetAmbient(ambient)
                prop.SetDiffuse(diffuse)
                prop.SetSpecular(specular)
            except Exception:
                pass

    def update_visibility(self):
        show = self.chk_4th.isChecked()
        if self.actor_std: self.actor_std.SetVisibility(not show)
        if self.actor_exp: self.actor_exp.SetVisibility(show)
        if self.actor_man: self.actor_man.SetVisibility(not show)
        if self.actor_man_exp: self.actor_man_exp.SetVisibility(show)
        if self.actor_line: self.actor_line.SetVisibility(not show)
        if self.actor_line_exp: self.actor_line_exp.SetVisibility(show)

    def update_opacity(self):
        cyl_op = self.slider_cyl.value() / 100.0
        if self.actor_std: self.actor_std.GetProperty().SetOpacity(cyl_op)
        if self.actor_exp: self.actor_exp.GetProperty().SetOpacity(cyl_op)
        if self.actor_man: self.actor_man.GetProperty().SetOpacity(cyl_op)
        if self.actor_man_exp: self.actor_man_exp.GetProperty().SetOpacity(cyl_op)
        if self.actor_line: self.actor_line.GetProperty().SetOpacity(cyl_op)
        if self.actor_line_exp: self.actor_line_exp.GetProperty().SetOpacity(cyl_op)

        if self.vol_actor:
            try: vol_max = float(self.txt_vol_max.text())
            except ValueError: vol_max = 1.0

            vol_op = (self.slider_vol.value() / 100.0) * vol_max
            otf = self.vol_actor.GetProperty().GetScalarOpacity()
            otf.RemoveAllPoints()
            curve = (self.combo_opacity_curve.currentText()
                     if hasattr(self, 'combo_opacity_curve') else "Linear")
            for v, op in self._build_opacity_points(curve, self.clim, vol_op):
                otf.AddPoint(v, op)

        self.plotter.render()

    @staticmethod
    def _build_opacity_points(curve, clim, vol_op):
        """Return a list of (scalar_value, opacity) tuples for the volume's
        scalar opacity transfer function.

        - Linear: ramp from clim[0]→0 to clim[1]→vol_op (matches old behavior).
        - Sigmoid (gentle/medium/sharp): S-curve sampled at 32 points; center
          at the dynamic-range midpoint, sharper variants suppress more of
          the lower range — useful for hiding empty resin around an object.
        - Threshold (low/mid/high): hard cutoff at 30% / 50% / 70% of the
          range; below the cutoff opacity is 0, above it is vol_op.
        """
        lo, hi = float(clim[0]), float(clim[1])
        if hi <= lo:
            return [(lo, 0.0), (lo + 1.0, vol_op)]

        if curve == "Linear":
            return [(lo, 0.0), (hi, vol_op)]

        if curve.startswith("Threshold"):
            frac = {"Threshold (low)": 0.30,
                    "Threshold (mid)": 0.50,
                    "Threshold (high)": 0.70}.get(curve, 0.50)
            cut = lo + frac * (hi - lo)
            eps = max(1e-9, (hi - lo) * 1e-3)
            return [(lo, 0.0),
                    (cut - eps, 0.0),
                    (cut + eps, vol_op),
                    (hi, vol_op)]

        # Sigmoid family: f(x) = 1 / (1 + exp(-k*(x - x0)))
        # Steeper k → sharper transition; x0 sits at 50% of range.
        k_map = {"Sigmoid (gentle)": 4.0,
                 "Sigmoid (medium)": 8.0,
                 "Sigmoid (sharp)":  16.0}
        k = k_map.get(curve, 8.0)
        n = 32
        xs = np.linspace(lo, hi, n)
        norm = (xs - lo) / (hi - lo) - 0.5  # in [-0.5, 0.5]
        y = 1.0 / (1.0 + np.exp(-k * norm))
        # Anchor the curve so opacity is exactly 0 at lo and vol_op at hi.
        y = (y - y[0]) / max(y[-1] - y[0], 1e-12)
        y *= vol_op
        return list(zip(xs.tolist(), y.tolist()))

    def export_coordinates(self):
        pts = self.get_all_active_points()
        if len(pts) == 0:
            print("\n=== SCAN EXPORT ===\nNo active cylinders.\n")
            return

        print("\n=== SCAN EXPORT ===")
        print(f"Total Active (Auto + Manual): {len(pts)}")
        print("ID, X, Y, Z")
        for i,p in enumerate(pts):
            print(f"{i},{int(p[0])},{int(p[1])},{int(p[2])}")
