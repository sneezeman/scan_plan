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

from scan_plan.io import parse_nml, load_volume, load_config, detect_tiff_dims
from scan_plan.solver import calculate_contrast_limits
from scan_plan.gui import CylinderApp

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scan_plan_config.json", help="Path to JSON configuration file.")
    parser.add_argument("--volume", help="Override volume path.")
    parser.add_argument("--nml", help="Override ROI with NML file.")
    parser.add_argument("--binning", type=int, help="Override binning factor.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose memory shape logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format='[%(levelname)s] %(message)s'
    )

    app = QtWidgets.QApplication(sys.argv)
    cfg = load_config(args.config)

    if args.volume: cfg['volume_path'] = args.volume
    if args.binning: cfg['binning'] = args.binning
    if args.nml: cfg['rois'] = parse_nml(args.nml)

    fp = cfg['volume_path']
    detect_tiff_dims(fp, cfg, args.config)

    zr = cfg['prescan_z_step'] / cfg['prescan_pixel_size_xy']
    g, d = load_volume(
        fp,
        tuple(cfg['raw_dims']),
        cfg['raw_dtype'],
        cfg['binning'],
        z_ratio=zr,
        header_bytes=cfg.get('raw_header_bytes', 0)
    )

    c = calculate_contrast_limits(d)
    w = CylinderApp(cfg, g, d, c)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
