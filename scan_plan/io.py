"""
I/O helpers: config loading, volume loading, NML parsing, TIFF dimension detection.
"""

import os
import json
import logging
import xml.etree.ElementTree as ET

import numpy as np
import tifffile
import pyvista as pv

logger = logging.getLogger(__name__)


def parse_nml(file_path):
    if not os.path.exists(file_path):
        return []
    rois = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for bbox in root.iter('userBoundingBox'):
            try:
                rois.append({
                    'x': int(bbox.get('topLeftX')),
                    'y': int(bbox.get('topLeftY')),
                    'z': int(bbox.get('topLeftZ')),
                    'w': int(bbox.get('width')),
                    'h': int(bbox.get('height')),
                    'd': int(bbox.get('depth'))
                })
            except (ValueError, TypeError):
                continue
        return rois
    except Exception:
        return []


def load_volume(filepath, dims, dtype_str, binning, z_ratio=1.0, header_bytes=0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Volume file not found: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ['.tif', '.tiff', '.raw', '.vol']:
        raise ValueError(f"Unsupported volume format: '{ext}'. Use .tif, .tiff, .raw, or .vol")
    try:
        if ext in ['.tif', '.tiff']:
            with tifffile.TiffFile(filepath) as tif:
                data = tif.asarray()
        else:
            dtype = np.dtype(dtype_str)
            x, y, z = dims
            expected_elements = x * y * z
            with open(filepath, 'rb') as f:
                if header_bytes > 0: f.seek(header_bytes)
                data = np.fromfile(f, dtype=dtype, count=expected_elements)
            data = data.reshape((z, y, x))

        data = np.squeeze(data)
        if data.ndim > 3:
            data = data[..., 0]
            data = np.squeeze(data)
        while data.ndim < 3:
            data = data[np.newaxis, ...]

        if data.ndim != 3:
            raise ValueError(f"Array cannot be formatted as 3D. Final shape: {data.shape}")

        data = np.transpose(data, (2, 1, 0))

        max_display_voxels = 100_000_000
        render_bin = 1
        while (data.size / (render_bin**3)) > max_display_voxels:
            render_bin += 1

        if render_bin > 1:
            data = data[::render_bin, ::render_bin, ::render_bin]
            display_spacing = (binning * render_bin, binning * render_bin, binning * z_ratio * render_bin)
        else:
            display_spacing = (binning, binning, binning * z_ratio)

        grid = pv.ImageData()
        grid.dimensions = data.shape
        grid.origin = (0, 0, 0)
        grid.spacing = display_spacing
        grid.point_data["values"] = data.flatten(order="F")
        return grid, data

    except Exception as e:
        logger.error("Failed to load volume: %s", e, exc_info=True)
        return None, None


def load_config(filepath):
    default_config = {
        "volume_path": "/path/to/your/scan/example_norec_.vol",
        "binning": 1,
        "raw_dims": [2048, 2048, 2048],
        "raw_dtype": "float32",
        "raw_header_bytes": 0,
        "prescan_pixel_size_xy": 180,
        "prescan_z_step": 180,
        "scan_pixel_size": 20,
        "rois": [],
        "optics": {
            "beam_pitch_rad": -0.015396,
            "optics_pixel_size_um": 2.952,
            "z12": 1281,
            "sx0_mm": 1.28,
            "rotation_offset_deg": -21.5
        }
    }

    if not os.path.exists(filepath):
        logger.info("Config file not found. Creating default: %s", filepath)
        with open(filepath, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except Exception as e:
            logger.error("Failed to parse config JSON: %s", e)
            return default_config


def detect_tiff_dims(filepath, config, config_path):
    """Auto-detect TIFF dimensions and update config if they differ."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ['.tif', '.tiff'] or not os.path.exists(filepath):
        return

    try:
        with tifffile.TiffFile(filepath) as tif:
            if tif.series:
                t_shape = tif.series[0].shape
                t_dtype = tif.series[0].dtype.name
            else:
                t_shape = (len(tif.pages), tif.pages[0].shape[0], tif.pages[0].shape[1])
                t_dtype = tif.pages[0].dtype.name

            if isinstance(t_shape, tuple):
                t_shape = tuple(d for d in t_shape if d > 1)
                while len(t_shape) < 3: t_shape = (1,) + t_shape
                t_shape = t_shape[-3:]

            new_dims = [t_shape[2], t_shape[1], t_shape[0]]
            if config['raw_dims'] != new_dims or config['raw_dtype'] != str(t_dtype):
                config['raw_dims'] = new_dims
                config['raw_dtype'] = str(t_dtype)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
    except Exception as e:
        logger.debug(f"TIFF shape parse fallback used: {e}")
