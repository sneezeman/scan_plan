# scan_plan

Advanced GUI for Cylinder Packing Strategy in high-resolution synchrotron tomography.

Plan cylindrical scan regions within a 3D prescan volume, pack them efficiently using bounding-box ROIs, and register coordinates between prescan and machine reference frames using dual SVD/Optimizer transformation models.

## Features

- 3D volume rendering with adjustable blending modes (Composite, MIP, MinIP, Average)
- Automatic cylinder grid generation with three fill modes: Strict, Center, Coverage
- Manual cylinder placement with bulk paste support
- ROI management from WebKnossos NML files or manual entry
- Dual coordinate registration (SVD Kabsch + SciPy Optimizer) with per-point error analysis
- Motor coordinate registration mode — match prescan pixels directly against su/sv/sz motor positions without a refscan
- NML export for standard and expanded bounding boxes
- Machine coordinate export with motor positions, Fiji-compatible coordinates, and WebKnossos NML bounding boxes
- Match point file loading — reload previously saved `_match_pairs.txt` files into the registration dialog

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

## Workflow

### 1. Prepare bounding boxes

Choose one of two paths to define regions of interest (ROIs) in your prescan volume:

**WebKnossos path:**

1. Upload prescan volume to WebKnossos
2. Draw bounding boxes around regions of interest
3. Export the annotation as an NML file (Download → NML)
4. Load the NML into scan-plan via the **Load NML** button

**Fiji / manual path:**

1. Open the prescan volume in Fiji (or any image viewer)
2. Identify bounding box coordinates manually as `x, y, z, w, h, d` (top-left corner + dimensions in pixels)
3. Enter each ROI directly in the scan-plan UI text field and click **Add**

### 2. Configure and launch scan-plan

1. Edit your JSON config file (see [Configuration](#configuration)):
   - Set `volume_path` to your prescan TIFF/RAW volume
   - Set pixel sizes (`prescan_pixel_size_xy`, `prescan_z_step`, `scan_pixel_size`)
   - Optionally pre-define `rois` in the config, or load them at runtime
2. Launch the GUI:
   ```bash
   scan-plan my_config.json
   ```

### 3. Adjust cylinder grid

1. Select a **Fill Mode** (Strict / Center / Coverage)
2. Use the **Shift Bounding Boxes** controls to nudge ROIs if needed
3. Toggle individual cylinders on/off in the **Auto Grid Cylinders** list
4. Add manual cylinders if needed via the **Manual Cylinders** section

### 4. Register coordinates

1. Click **REGISTER COORDINATES** to open the Registration Dialog
2. Enter matching point pairs (prescan pixels ↔ refscan pixels or motor coords):
   - **Paste Clipboard** — paste 6-column data (prescan X Y Z, refscan X Y Z)
   - **Load File** — load a previously saved `_match_pairs.txt` file
   - Or enter values manually in the tables
3. Set the machine reference parameters (su, sv, sz, pixel sizes)
4. Click **Calculate & Verify Models** to fit both SVD and Optimizer models
5. Review per-point errors in the Results tab
6. Click **SAVE MACHINE COORDINATES** to export all output files

## CLI Usage

```bash
scan-plan [CONFIG_PATH] [--debug]
```

### CLI Arguments

| Argument      | Description                          | Default                    |
|---------------|--------------------------------------|----------------------------|
| `CONFIG_PATH` | Path to JSON configuration file      | `scan_plan_config.json`    |
| `--debug`     | Enable verbose debug logging         | off                        |

All settings (volume path, binning, ROIs, optics, etc.) are specified in the JSON config file. If the config file does not exist, a default example is auto-generated.

### Examples

```bash
# Launch with default config (scan_plan_config.json)
scan-plan

# Use a custom config
scan-plan my_experiment.json

# Debug mode
scan-plan my_experiment.json --debug
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

## Motor Coordinate Registration

In the Registration Dialog, the right-table mode can be switched between **Refscan Pixels** (default) and **Motor Coordinates (su/sv/sz)**. Motor coordinate mode is useful when you identify features directly on the physical sample by moving the stage and recording su/sv/sz motor positions, without performing a refscan. The motor coordinates are automatically converted to virtual refscan pixel equivalents before fitting.

## Output Files

When you click **SAVE MACHINE COORDINATES** in the Registration Dialog, four files are saved (assuming the chosen filename is `tiles_motor_coords.txt`):

| File | Description |
|------|-------------|
| `tiles_motor_coords.txt` | Motor positions (su, sv, sz) for each active cylinder — the primary output for the beamline control system |
| `tiles_motor_coords_fiji.txt` | Prescan pixel coordinates (x, y, z) with cylinder dimensions (D_std, H_std) — for verification in Fiji |
| `tiles_motor_coords_webknossos.nml` | WebKnossos NML with active cylinders as bounding boxes — for verification in WebKnossos |
| `tiles_motor_coords_match_pairs.txt` | Saved match point pairs (prescan ↔ refscan) — can be reloaded via the **Load File** button |

Additionally, the main window **EXPORT NML (TILES)** button exports two NML files with standard and expanded bounding boxes for all active cylinders (before coordinate registration).

## Running Tests

```bash
pip install -e ".[test]"
pytest
```
