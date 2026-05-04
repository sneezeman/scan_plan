"""
scan_plan - Advanced GUI for Cylinder Packing Strategy in high-resolution tomography.

Entry point only. All logic lives in io.py, solver.py, gui.py, volume_registration.py,
and nml_exporter.py.
"""

import json
import os
import sys
import argparse
import datetime
import logging
import warnings

from PyQt5 import QtWidgets

warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")

from scan_plan.io import load_volume, load_config, detect_tiff_dims
from scan_plan.solver import calculate_contrast_limits
from scan_plan.gui import CylinderApp, ConfigDialog

logger = logging.getLogger(__name__)


def _update_user_config(config_path, updates):
    """Read the user's config file, apply *updates*, and write it back.

    Only the keys present in *updates* are touched; every other key the
    user had (and nothing they didn't have) is preserved.  This avoids
    injecting instrument-default keys (optics, motor_limits, etc.) into
    the user's file.
    """
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, 'r') as f:
            user_cfg = json.load(f)
    except Exception:
        return
    user_cfg.update(updates)
    with open(config_path, 'w') as f:
        json.dump(user_cfg, f, indent=4)


def _save_config_history(config_path):
    """Write a timestamped copy of *config_path* into a sibling history dir.

    Layout:
      <config_dir>/scan_plan_config_history/<basename>_<YYYYMMDD-HHMMSS>.json

    Errors are logged but not raised — history is best-effort.
    """
    if not os.path.exists(config_path):
        return
    try:
        cfg_dir = os.path.dirname(os.path.abspath(config_path)) or "."
        history_dir = os.path.join(cfg_dir, "scan_plan_config_history")
        os.makedirs(history_dir, exist_ok=True)
        base = os.path.basename(config_path)
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".json"
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(history_dir, f"{stem}_{ts}{ext}")
        with open(config_path, "r") as src, open(dest, "w") as dst:
            dst.write(src.read())
        logger.info("Saved config history copy: %s", dest)
    except (OSError, PermissionError) as e:
        logger.warning("Could not save config history copy: %s", e)


def main():
    parser = argparse.ArgumentParser(
        description="Advanced GUI for Cylinder Packing Strategy in high-resolution tomography."
    )
    parser.add_argument("config", nargs="?", default="scan_plan_config.json",
                        help="Path to JSON configuration file (default: scan_plan_config.json)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format='[%(levelname)s] %(message)s'
    )

    app = QtWidgets.QApplication(sys.argv)
    cfg = load_config(args.config)

    # Startup wizard — replaces manual JSON editing as the primary entry point.
    dlg = ConfigDialog(cfg, args.config)
    if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.accepted_proceed:
        updates = dlg.get_updates()
        cfg.update(updates)
        _update_user_config(args.config, updates)
        _save_config_history(args.config)

    fp = cfg['volume_path']
    detected = detect_tiff_dims(fp)
    if detected is not None:
        new_dims, new_dtype = detected
        changed = {}
        if cfg['raw_dims'] != new_dims:
            changed['raw_dims'] = new_dims
        if cfg['raw_dtype'] != new_dtype:
            changed['raw_dtype'] = new_dtype
        if changed:
            cfg.update(changed)
            _update_user_config(args.config, changed)

    zr = cfg['prescan_z_step'] / cfg['prescan_pixel_size_xy']
    try:
        g, d = load_volume(
            fp,
            tuple(cfg['raw_dims']),
            cfg['raw_dtype'],
            cfg['binning'],
            z_ratio=zr,
            header_bytes=cfg.get('raw_header_bytes', 0)
        )
    except (FileNotFoundError, ValueError) as e:
        logger.warning("%s", e)
        g, d = None, None

    c = calculate_contrast_limits(d)
    del d  # free raw volume data now that contrast limits are computed
    w = CylinderApp(cfg, g, c, config_path=args.config)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
