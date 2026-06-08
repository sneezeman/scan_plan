# Beam-energy Presets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 33keV/17keV beam-energy presets to the `scan_planner` startup wizard, with an editor window and locally-persisted user overrides.

**Architecture:** A preset is an optics-only dict. Bundled presets ship as `scan_plan/presets/*.json`; user edits persist to `~/.config/scan_planner/presets.json` (XDG-aware), deep-merged over bundled. `cli.main` resolves the active preset into `cfg['optics']` — a single override point, since every consumer already reads `cfg.get('optics')`. The wizard gains a dropdown + "Edit presets…" button opening a new `PresetEditorDialog`.

**Tech Stack:** Python 3.13, PyQt5, pytest, conda env `scan_plan`.

---

## Environment note

All test commands assume the `scan_plan` conda env (Python 3.13). Prefix with:

```bash
source activate scan_plan
```

The base env is Python 3.8 and will fail. Run tests from the repo root
`/home/artem1706/GoogleDrive/Projects_work/scan_plan`.

## File structure

- Create: `scan_plan/presets/33keV.json` — bundled 33keV optics block.
- Create: `scan_plan/presets/17keV.json` — bundled 17keV optics block.
- Modify: `scan_plan/io.py` — add `user_config_dir()`, `load_bundled_presets()`, `load_presets()`, `save_presets()`.
- Modify: `pyproject.toml` — ship `presets/*.json` as package data.
- Modify: `scan_plan/cli.py` — resolve `active_preset` → `cfg['optics']`.
- Modify: `scan_plan/gui.py` — add `PresetEditorDialog`; add dropdown + button to `ConfigDialog`; persist `active_preset` in `get_updates()`.
- Test: `tests/test_io.py` — preset helpers.

---

### Task 1: Bundled preset JSON files

**Files:**
- Create: `scan_plan/presets/33keV.json`
- Create: `scan_plan/presets/17keV.json`

- [ ] **Step 1: Create the 33keV preset file**

Create `scan_plan/presets/33keV.json`:

```json
{
    "beam_pitch_rad": -0.015396,
    "optics_pixel_size_um": 2.952,
    "z12": 1282,
    "sx0_mm": 1.292,
    "rotation_offset_deg": -21.5
}
```

- [ ] **Step 2: Create the 17keV preset file**

Create `scan_plan/presets/17keV.json`:

```json
{
    "beam_pitch_rad": -0.015396,
    "optics_pixel_size_um": 2.952,
    "z12": 1213,
    "sx0_mm": -3.113,
    "rotation_offset_deg": -21.5
}
```

- [ ] **Step 3: Commit**

```bash
git add scan_plan/presets/33keV.json scan_plan/presets/17keV.json
git commit -m "Add bundled 33keV/17keV optics preset files"
```

---

### Task 2: Ship presets as package data

**Files:**
- Modify: `pyproject.toml:28-29`

- [ ] **Step 1: Add presets glob to package-data**

In `pyproject.toml`, change the `[tool.setuptools.package-data]` block from:

```toml
[tool.setuptools.package-data]
scan_plan = ["instrument_defaults.json"]
```

to:

```toml
[tool.setuptools.package-data]
scan_plan = ["instrument_defaults.json", "presets/*.json"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "Ship presets/*.json as package data"
```

---

### Task 3: `user_config_dir()` helper

**Files:**
- Modify: `scan_plan/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_io.py` (update the import line at the top from
`from scan_plan.io import parse_nml, load_config, load_volume` to
`from scan_plan.io import parse_nml, load_config, load_volume, user_config_dir, load_bundled_presets, load_presets, save_presets`):

```python
class TestUserConfigDir:
    def test_respects_xdg_config_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert user_config_dir() == os.path.join(str(tmp_path), "scan_planner")

    def test_falls_back_to_home_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert user_config_dir() == os.path.join(str(tmp_path), ".config", "scan_planner")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestUserConfigDir -v`
Expected: FAIL with `ImportError: cannot import name 'user_config_dir'`.

- [ ] **Step 3: Write minimal implementation**

In `scan_plan/io.py`, add after `_load_instrument_defaults` (anywhere at module
scope is fine):

```python
def user_config_dir():
    """Directory for user-writable scan_planner config (XDG-aware).

    Honours ``$XDG_CONFIG_HOME`` when set, otherwise ``~/.config``. Under
    Apptainer the host ``$HOME`` is bind-mounted, so this persists across
    container runs.
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return os.path.join(base, "scan_planner")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestUserConfigDir -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scan_plan/io.py tests/test_io.py
git commit -m "Add user_config_dir() XDG-aware helper"
```

---

### Task 4: `load_bundled_presets()`

**Files:**
- Modify: `scan_plan/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_io.py`:

```python
class TestLoadBundledPresets:
    def test_contains_both_energies(self):
        presets = load_bundled_presets()
        assert set(presets) == {"33keV", "17keV"}

    def test_seed_values(self):
        presets = load_bundled_presets()
        assert presets["33keV"]["z12"] == 1282
        assert presets["33keV"]["sx0_mm"] == 1.292
        assert presets["17keV"]["z12"] == 1213
        assert presets["17keV"]["sx0_mm"] == -3.113
        # shared constants identical across energies
        for name in ("33keV", "17keV"):
            assert presets[name]["beam_pitch_rad"] == -0.015396
            assert presets[name]["optics_pixel_size_um"] == 2.952
            assert presets[name]["rotation_offset_deg"] == -21.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestLoadBundledPresets -v`
Expected: FAIL with `ImportError: cannot import name 'load_bundled_presets'`.

- [ ] **Step 3: Write minimal implementation**

In `scan_plan/io.py`, add:

```python
def load_bundled_presets():
    """Load the bundled energy presets shipped in ``scan_plan/presets/``.

    Returns ``{name: optics_dict}``, keyed by the JSON filename stem
    (e.g. ``"33keV"``).
    """
    presets_dir = os.path.join(os.path.dirname(__file__), "presets")
    presets = {}
    for fname in os.listdir(presets_dir):
        if not fname.endswith(".json"):
            continue
        name = os.path.splitext(fname)[0]
        with open(os.path.join(presets_dir, fname), "r") as f:
            presets[name] = json.load(f)
    return presets
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestLoadBundledPresets -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scan_plan/io.py tests/test_io.py
git commit -m "Add load_bundled_presets()"
```

---

### Task 5: `load_presets()` / `save_presets()` round-trip + merge

**Files:**
- Modify: `scan_plan/io.py`
- Test: `tests/test_io.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_io.py`:

```python
class TestUserPresets:
    def test_load_returns_bundled_when_no_user_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        presets = load_presets()
        assert set(presets) == {"33keV", "17keV"}
        assert presets["33keV"]["z12"] == 1282

    def test_save_then_load_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        presets = load_presets()
        presets["33keV"]["z12"] = 9999
        save_presets(presets)
        reloaded = load_presets()
        assert reloaded["33keV"]["z12"] == 9999

    def test_user_file_deep_merged_over_bundled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # User file overrides only one field of 33keV; missing fields and the
        # whole 17keV preset must still come from bundled.
        cfg_dir = os.path.join(str(tmp_path), "scan_planner")
        os.makedirs(cfg_dir, exist_ok=True)
        with open(os.path.join(cfg_dir, "presets.json"), "w") as f:
            json.dump({"33keV": {"sx0_mm": 0.0}}, f)
        presets = load_presets()
        assert presets["33keV"]["sx0_mm"] == 0.0       # user wins
        assert presets["33keV"]["z12"] == 1282          # inherited from bundled
        assert presets["17keV"]["z12"] == 1213          # bundled preset preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestUserPresets -v`
Expected: FAIL with `ImportError: cannot import name 'load_presets'`.

- [ ] **Step 3: Write minimal implementation**

In `scan_plan/io.py`, add. Note `_deep_merge(defaults, overrides)` already
exists in this module (used by `load_config`) and merges defaults *under*
overrides, recursing into nested dicts — reuse it:

```python
def _user_presets_path():
    return os.path.join(user_config_dir(), "presets.json")


def load_presets():
    """Energy presets: bundled defaults deep-merged under the user's file.

    The user file (``~/.config/scan_planner/presets.json``) wins per field;
    any preset or field it omits is inherited from the bundled set, so new
    bundled presets/fields still appear after an upgrade.
    """
    presets = load_bundled_presets()
    path = _user_presets_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                user = json.load(f)
        except Exception as e:
            logger.warning("Could not read user presets: %s", e)
            user = {}
        presets = _deep_merge(presets, user)
    return presets


def save_presets(presets):
    """Write the full preset set to the user's config file.

    Creates the config directory if needed. Errors are logged, never raised.
    """
    path = _user_presets_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(presets, f, indent=4)
    except (OSError, PermissionError) as e:
        logger.warning("Could not write user presets: %s", e)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source activate scan_plan && python -m pytest tests/test_io.py::TestUserPresets -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full io test module**

Run: `source activate scan_plan && python -m pytest tests/test_io.py -v`
Expected: PASS (all green, including pre-existing tests).

- [ ] **Step 6: Commit**

```bash
git add scan_plan/io.py tests/test_io.py
git commit -m "Add load_presets()/save_presets() with bundled deep-merge"
```

---

### Task 6: Resolve active preset into cfg['optics'] in cli.main

**Files:**
- Modify: `scan_plan/cli.py:89` (just after `cfg = load_config(args.config)`)

- [ ] **Step 1: Add preset resolution after load_config**

In `scan_plan/cli.py`, update the import line:

```python
from scan_plan.io import load_volume, load_config, detect_tiff_dims
```

to:

```python
from scan_plan.io import load_volume, load_config, detect_tiff_dims, load_presets
```

Then, immediately after the line `cfg = load_config(args.config)` (currently
`cli.py:89`) and **before** the `ConfigDialog` is constructed, insert:

```python
    # Resolve the active beam-energy preset into cfg['optics']. The user
    # config carries only the preset name; the optics values themselves live
    # in the (bundled + user) preset store.
    presets = load_presets()
    active_preset = cfg.get("active_preset")
    if active_preset not in presets:
        active_preset = next(iter(presets))  # deterministic fallback
        cfg["active_preset"] = active_preset
    cfg["optics"] = presets[active_preset]
```

- [ ] **Step 2: Verify the app still imports and starts the wizard logic**

Run: `source activate scan_plan && python -c "import scan_plan.cli; print('ok')"`
Expected: prints `ok` with no import error.

- [ ] **Step 3: Commit**

```bash
git add scan_plan/cli.py
git commit -m "Resolve active beam-energy preset into cfg['optics']"
```

---

### Task 7: PresetEditorDialog

**Files:**
- Modify: `scan_plan/gui.py` (add new class; place it directly after the `ConfigDialog` class, before `RegistrationDialog` at line 220)

- [ ] **Step 1: Confirm io imports available in gui.py**

Check the existing imports at the top of `scan_plan/gui.py`. Ensure
`load_presets` and `save_presets` are imported from `scan_plan.io`. If the file
imports io helpers like `from scan_plan.io import detect_tiff_dims`, extend that
line; otherwise add:

```python
from scan_plan.io import load_presets, save_presets, load_bundled_presets
```

(Place alongside the other `scan_plan.io` imports near the top of the file.)

- [ ] **Step 2: Add the PresetEditorDialog class**

Insert this class in `scan_plan/gui.py` immediately before
`class RegistrationDialog` (currently line 220):

```python
class PresetEditorDialog(QtWidgets.QDialog):
    """Edit beam-energy optics presets; persists to the user presets file."""

    # (label, key, suffix, decimals)
    FIELDS = (
        ("Beam pitch", "beam_pitch_rad", " rad", 6),
        ("Optics pixel size", "optics_pixel_size_um", " µm", 4),
        ("z12 (z1+z2)", "z12", " mm", 3),
        ("sx0", "sx0_mm", " mm", 4),
        ("Rotation offset", "rotation_offset_deg", " deg", 4),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Beam-Energy Presets")
        self.resize(420, 0)

        # Working copy held in memory; written only on Save.
        self.presets = load_presets()
        self._current = None  # name of preset currently shown

        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Edit optics parameters per beam energy. Changes are saved to your "
            "local presets file and preserved across runs."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555;")
        layout.addWidget(intro)

        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Preset:"))
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(sorted(self.presets.keys()))
        self.combo.currentTextChanged.connect(self._on_preset_changed)
        sel_row.addWidget(self.combo, 1)
        layout.addLayout(sel_row)

        form_box = QtWidgets.QGroupBox("Optics")
        form = QtWidgets.QFormLayout()
        self.spins = {}
        for label, key, suffix, decimals in self.FIELDS:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(decimals)
            spin.setSuffix(suffix)
            # Commit edits to the in-memory dict as the user types.
            spin.valueChanged.connect(self._on_value_changed)
            self.spins[key] = spin
            form.addRow(label + ":", spin)
        form_box.setLayout(form)
        layout.addWidget(form_box)

        btn_row = QtWidgets.QHBoxLayout()
        btn_restore = QtWidgets.QPushButton("Restore bundled")
        btn_restore.clicked.connect(self._restore_bundled)
        btn_row.addWidget(btn_restore)
        btn_row.addStretch(1)
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.clicked.connect(self._save)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        if self.combo.count():
            self._on_preset_changed(self.combo.currentText())

    def _on_preset_changed(self, name):
        if not name:
            return
        self._current = name
        optics = self.presets.get(name, {})
        for key, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(float(optics.get(key, 0.0)))
            spin.blockSignals(False)

    def _on_value_changed(self, _value):
        if not self._current:
            return
        optics = self.presets.setdefault(self._current, {})
        for key, spin in self.spins.items():
            optics[key] = spin.value()

    def _restore_bundled(self):
        if not self._current:
            return
        bundled = load_bundled_presets()
        if self._current in bundled:
            self.presets[self._current] = dict(bundled[self._current])
            self._on_preset_changed(self._current)

    def _save(self):
        save_presets(self.presets)
        self.accept()
```

- [ ] **Step 3: Smoke-test the dialog headlessly**

Run:

```bash
source activate scan_plan && QT_QPA_PLATFORM=offscreen python -c "
from PyQt5 import QtWidgets
import sys
from scan_plan.gui import PresetEditorDialog
app = QtWidgets.QApplication(sys.argv)
d = PresetEditorDialog()
assert d.combo.count() == 2, d.combo.count()
d._on_preset_changed('33keV')
assert abs(d.spins['z12'].value() - 1282) < 1e-6, d.spins['z12'].value()
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add scan_plan/gui.py
git commit -m "Add PresetEditorDialog for editing beam-energy optics presets"
```

---

### Task 8: Wire dropdown + Edit button into ConfigDialog

**Files:**
- Modify: `scan_plan/gui.py` — `ConfigDialog.__init__` (the Session Settings form, around lines 52-67), `ConfigDialog._on_accept` (lines 196-214), `ConfigDialog.get_updates` (lines 216-217), and `ConfigDialog.EDITABLE_KEYS` (lines 28-32).

- [ ] **Step 1: Add active_preset to EDITABLE_KEYS**

In `ConfigDialog`, change `EDITABLE_KEYS` from:

```python
    EDITABLE_KEYS = (
        "volume_path", "binning",
        "raw_dims", "raw_dtype", "raw_header_bytes",
        "prescan_pixel_size_xy", "prescan_z_step", "scan_pixel_size",
    )
```

to:

```python
    EDITABLE_KEYS = (
        "volume_path", "binning",
        "raw_dims", "raw_dtype", "raw_header_bytes",
        "prescan_pixel_size_xy", "prescan_z_step", "scan_pixel_size",
        "active_preset",
    )
```

- [ ] **Step 2: Add the beam-energy row at the top of the Session Settings form**

In `ConfigDialog.__init__`, the form is built starting near line 52
(`form_box = QtWidgets.QGroupBox("Session Settings")` / `form = QtWidgets.QFormLayout()`).
Immediately after `form = QtWidgets.QFormLayout()` and **before** the
`# Volume file picker` block, insert:

```python
        # Beam-energy preset selector + editor launcher.
        from scan_plan.io import load_presets  # local import: avoids load at import time
        self._presets = load_presets()
        h_energy = QtWidgets.QHBoxLayout()
        self.combo_energy = QtWidgets.QComboBox()
        self.combo_energy.addItems(sorted(self._presets.keys()))
        active = config.get("active_preset")
        if active in self._presets:
            self.combo_energy.setCurrentText(active)
        btn_edit_presets = QtWidgets.QPushButton("Edit presets…")
        btn_edit_presets.clicked.connect(self._edit_presets)
        h_energy.addWidget(self.combo_energy, 1)
        h_energy.addWidget(btn_edit_presets)
        form.addRow("Beam energy:", h_energy)
```

- [ ] **Step 3: Add the _edit_presets handler method**

Add this method to `ConfigDialog` (e.g. directly after `__init__`, before
`_browse_volume`):

```python
    def _edit_presets(self):
        dlg = PresetEditorDialog(self)
        dlg.exec_()
        # Refresh the dropdown from the (possibly edited) preset store,
        # preserving the current selection where still valid.
        current = self.combo_energy.currentText()
        self._presets = load_presets()
        self.combo_energy.blockSignals(True)
        self.combo_energy.clear()
        self.combo_energy.addItems(sorted(self._presets.keys()))
        if current in self._presets:
            self.combo_energy.setCurrentText(current)
        self.combo_energy.blockSignals(False)
```

Note: `load_presets` is referenced here too. Add a module-level import at the
top of `gui.py` (alongside the other `scan_plan.io` imports):
`from scan_plan.io import load_presets, save_presets, load_bundled_presets`
(if not already added in Task 7 Step 1). The local import inside `__init__`
from Step 2 can then be removed — but leaving it is harmless. Prefer the
module-level import and delete the local one for cleanliness.

- [ ] **Step 4: Persist the selected preset in _on_accept**

In `ConfigDialog._on_accept`, the `updates` dict is built around lines 198-211.
Add `active_preset` to it. Change the closing of that dict from:

```python
            "scan_pixel_size": self.spin_scan_px.value(),
        }
        self.config.update(updates)
```

to:

```python
            "scan_pixel_size": self.spin_scan_px.value(),
            "active_preset": self.combo_energy.currentText(),
        }
        self.config.update(updates)
```

- [ ] **Step 5: Smoke-test ConfigDialog headlessly**

Run:

```bash
source activate scan_plan && QT_QPA_PLATFORM=offscreen python -c "
from PyQt5 import QtWidgets
import sys
from scan_plan.gui import ConfigDialog
app = QtWidgets.QApplication(sys.argv)
cfg = {'volume_path': '', 'binning': 1, 'raw_dims': [8,8,8], 'raw_dtype': 'float32',
       'raw_header_bytes': 0, 'prescan_pixel_size_xy': 150, 'prescan_z_step': 150,
       'scan_pixel_size': 20, 'active_preset': '17keV'}
d = ConfigDialog(cfg, 'dummy_config.json')
assert d.combo_energy.currentText() == '17keV', d.combo_energy.currentText()
d._on_accept()
assert d.get_updates()['active_preset'] == '17keV'
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add scan_plan/gui.py
git commit -m "Add beam-energy dropdown + Edit presets button to ConfigDialog"
```

---

### Task 9: Full regression run

**Files:** none (verification only)

- [ ] **Step 1: Run the whole test suite**

Run: `source activate scan_plan && python -m pytest tests/ -v`
Expected: all tests pass (pre-existing + the new preset tests).

- [ ] **Step 2: Sanity-check the end-to-end resolution**

Run:

```bash
source activate scan_plan && python -c "
import json, tempfile, os
from scan_plan.io import load_config, load_presets
d = tempfile.mkdtemp()
cfgp = os.path.join(d, 'scan_plan_config.json')
with open(cfgp, 'w') as f:
    json.dump({'active_preset': '17keV'}, f)
cfg = load_config(cfgp)
presets = load_presets()
active = cfg.get('active_preset')
cfg['optics'] = presets[active]
assert cfg['optics']['z12'] == 1213, cfg['optics']
assert cfg['optics']['sx0_mm'] == -3.113
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 3: Final commit (if any uncommitted changes remain)**

```bash
git status
# if clean, nothing to do
```

---

## Self-review notes

- **Spec coverage:** bundled presets (Task 1), package-data (Task 2),
  `user_config_dir`/`load_bundled_presets`/`load_presets`/`save_presets`
  (Tasks 3-5), runtime wiring into `cfg['optics']` (Task 6), `PresetEditorDialog`
  with Save/Restore-bundled/Close (Task 7), wizard dropdown + Edit button +
  `active_preset` persistence (Task 8), tests + regression (Tasks 3-5, 9).
  All spec sections covered.
- **Naming consistency:** `load_presets`, `save_presets`, `load_bundled_presets`,
  `user_config_dir`, `active_preset`, `cfg['optics']`, `combo_energy`,
  `PresetEditorDialog`, `_edit_presets` used identically across tasks.
- **`_deep_merge` reuse:** confirmed present in `io.py` (used by `load_config`),
  signature `_deep_merge(defaults, overrides)` with overrides winning — matches
  Task 5 usage.
