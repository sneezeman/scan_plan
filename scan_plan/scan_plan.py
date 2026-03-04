"""
scan_plan - Advanced GUI for Cylinder Packing Strategy in high-resolution tomography.

Entry point only. All logic lives in io.py, solver.py, gui.py, volume_registration.py,
and nml_exporter.py.
"""

import sys
import argparse
import logging
import warnings

from PyQt5 import QtWidgets

warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")

from scan_plan.io import load_volume, load_config, detect_tiff_dims
from scan_plan.solver import calculate_contrast_limits
from scan_plan.gui import CylinderApp

logger = logging.getLogger(__name__)


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

    fp = cfg['volume_path']
    detect_tiff_dims(fp, cfg, args.config)

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
    w = CylinderApp(cfg, g, d, c)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
