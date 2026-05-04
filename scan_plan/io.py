"""
I/O helpers: config loading, volume loading, NML parsing, TIFF dimension detection.
"""

import os
import json
import shutil
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
            data = np.memmap(filepath, dtype=dtype, mode='r',
                             offset=header_bytes, shape=(z, y, x))
            if data.size != expected_elements:
                raise ValueError(
                    f"Expected {expected_elements} elements but got {data.size}"
                )

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
        grid.point_data["values"] = data.ravel(order="F")
        return grid, data

    except (IOError, OSError, ValueError, MemoryError, tifffile.TiffFileError) as e:
        logger.warning("Failed to load volume: %s", e, exc_info=True)
        return None, None


def _load_instrument_defaults(target_dir=None):
    """Load instrument defaults (optics, motor limits).

    If *target_dir* is given, the bundled JSON is copied there (if not
    already present) and loaded from the copy. This lets users tweak
    per-session instrument parameters without touching the package —
    edits persist across runs but the user can always delete the copy
    to regenerate a fresh one from the package defaults.
    """
    bundled = os.path.join(os.path.dirname(__file__), 'instrument_defaults.json')

    if target_dir is not None:
        user_copy = os.path.join(target_dir, 'instrument_defaults.json')
        if not os.path.exists(user_copy):
            try:
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy(bundled, user_copy)
                logger.info("Copied instrument_defaults.json to %s", user_copy)
            except (OSError, PermissionError) as e:
                logger.warning("Could not copy instrument_defaults.json to %s: %s — using bundled version", target_dir, e)
                user_copy = bundled
        path = user_copy
    else:
        path = bundled

    with open(path, 'r') as f:
        return json.load(f)


def _deep_merge(defaults, overrides):
    """Recursively merge *defaults* under *overrides*.

    For every key in *defaults*:
    - If the key is missing from *overrides*, use the default value.
    - If both values are dicts, recurse so that partial user dicts still
      inherit missing sub-keys from the defaults.
    - Otherwise the user value in *overrides* wins.

    Returns a new dict (neither input is mutated).
    """
    merged = dict(defaults)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(filepath):
    default_config = {
        "volume_path": "/path/to/your/scan/example_norec_.vol",
        "binning": 1,
        "raw_dims": [2048, 2048, 2048],
        "raw_dtype": "float32",
        "raw_header_bytes": 0,
        "prescan_pixel_size_xy": 150,
        "prescan_z_step": 150,
        "scan_pixel_size": 20,
        "rois": []
    }

    if not os.path.exists(filepath):
        logger.info("Config file not found. Creating default: %s", filepath)
        with open(filepath, 'w') as f:
            json.dump(default_config, f, indent=4)
        cfg = default_config
    else:
        with open(filepath, 'r') as f:
            try:
                cfg = json.load(f)
            except Exception as e:
                logger.error("Failed to parse config JSON: %s", e)
                cfg = default_config

    # Copy instrument_defaults.json next to the user's config file
    # (created on first run; user can edit it to tweak per-session values).
    target_dir = os.path.dirname(os.path.abspath(filepath))
    instrument = _load_instrument_defaults(target_dir)
    cfg = _deep_merge(instrument, cfg)

    return cfg


def detect_tiff_dims(filepath):
    """Auto-detect TIFF dimensions and dtype.

    Returns ``(dims, dtype_str)`` where *dims* is ``[x, y, z]`` and
    *dtype_str* is the NumPy dtype name (e.g. ``'uint16'``), or ``None``
    if *filepath* is not a TIFF or cannot be read.

    This is a **pure** function: it reads the file but never mutates
    external state or writes to disk.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in ['.tif', '.tiff'] or not os.path.exists(filepath):
        return None

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
            return new_dims, str(t_dtype)
    except Exception as e:
        logger.debug(f"TIFF shape parse fallback used: {e}")
        return None
