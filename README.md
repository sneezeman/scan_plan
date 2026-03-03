# scan_plan

Advanced GUI for Cylinder Packing Strategy in high-resolution synchrotron tomography.

Plan cylindrical scan regions within a 3D prescan volume, pack them efficiently using bounding-box ROIs, and register coordinates between prescan and machine reference frames using dual SVD/Optimizer transformation models.

## Features

- 3D volume rendering with adjustable blending modes (Composite, MIP, MinIP, Average)
- Automatic cylinder grid generation with three fill modes: Strict, Center, Coverage
- Manual cylinder placement with bulk paste support
- ROI management from WebKnossos NML files
- Dual coordinate registration (SVD Kabsch + SciPy Optimizer) with per-point error analysis
- NML export for standard and expanded bounding boxes
- Machine coordinate export with motor position calculation

## Installation

```bash
pip install -e .
```

For development with tests:

```bash
pip install -e ".[test]"
```

### Environment

Requires Python >= 3.10 and scipy >= 1.13 (for `scipy.spatial.transform.RigidTransform`).

Recommended setup with conda:

```bash
conda create -n scan_plan python=3.13 -y
conda activate scan_plan
pip install -e ".[test]"
```

## Usage

```bash
scan-plan [OPTIONS]
```

### CLI Arguments

| Argument    | Description                          | Default                    |
|-------------|--------------------------------------|----------------------------|
| `--config`  | Path to JSON configuration file      | `scan_plan_config.json`    |
| `--volume`  | Override volume file path            | (from config)              |
| `--nml`     | Load ROIs from an NML file           | (from config)              |
| `--binning` | Override binning factor              | (from config)              |
| `--debug`   | Enable verbose debug logging         | off                        |

### Examples

```bash
# Launch with default config
scan-plan

# Load a specific volume with debug logging
scan-plan --volume /data/sample.tif --debug

# Use a custom config and NML ROIs
scan-plan --config my_config.json --nml rois.nml --binning 2
```

## Configuration

The JSON config file controls volume loading and scan parameters. See `scan_plan_config.example.json` for a full template.

### Core Settings

| Key                    | Type     | Description                              |
|------------------------|----------|------------------------------------------|
| `volume_path`          | string   | Path to .tif/.tiff/.raw/.vol volume      |
| `binning`              | int      | Prescan binning factor                   |
| `raw_dims`             | [x,y,z]  | Volume dimensions (for .raw/.vol)        |
| `raw_dtype`            | string   | Data type (e.g. `"float32"`)             |
| `raw_header_bytes`     | int      | Header offset in raw files               |
| `prescan_pixel_size_xy`| number   | Prescan pixel size in nm (XY)            |
| `prescan_z_step`       | number   | Prescan Z step size in nm                |
| `scan_pixel_size`      | number   | Target scan pixel size in nm             |
| `rois`                 | array    | Pre-defined ROI bounding boxes           |

### Optics Settings (optional)

| Key                    | Type   | Default    | Description                       |
|------------------------|--------|------------|-----------------------------------|
| `beam_pitch_rad`       | float  | -0.015396  | Beam pitch angle in radians       |
| `optics_pixel_size_um` | float  | 2.952      | Optics pixel size in micrometers  |
| `z12`                  | int    | 1281       | Optics z12 parameter              |
| `sx0_mm`               | float  | 1.28       | sx0 offset in mm                  |
| `rotation_offset_deg`  | float  | -21.5      | Stage rotation offset in degrees  |

## Running Tests

```bash
pip install -e ".[test]"
pytest
```
