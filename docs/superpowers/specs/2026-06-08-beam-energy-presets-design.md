# Beam-energy presets — design

Date: 2026-06-08
Status: approved (pending spec review)

## Goal

Add beam-energy presets (33keV / 17keV) to the `scan_planner` startup
**Session Settings** wizard. The user picks an energy from a dropdown; its
optics parameters are applied to the session. A **"Edit presets…"** button
opens a separate window to manually edit every relevant value. Bundled preset
values ship with the package; user modifications are persisted locally and
survive across runs (and across Apptainer container runs).

## Background — what an "optics preset" is

The values that distinguish a beam energy live in the `optics` block currently
held in `instrument_defaults.json`:

| field | physical meaning (per `volume_registration.py`) | energy-dependent? |
|---|---|---|
| `optics_pixel_size_um` | magnified detector/scintillator pixel | No (camera optic) |
| `z12` | focus→detector distance `z1+z2` (mm) | **Yes** |
| `sx0_mm` | sample-x motor offset at focus | **Yes** |
| `beam_pitch_rad` | vertical beam tilt; corrects `sz` vs x-travel | mildly |
| `rotation_offset_deg` | somega frame offset (geometry) | No |

A preset is **optics-only**. `motor_limits` stays global in
`instrument_defaults.json`, untouched.

## Bundled preset values

Extracted from the ID16A commissioning logbook 2026-I (33keV RT runs and the
17keV cryo run BLC17144, 02 Jun 2026):

```json
{
  "33keV": {
    "beam_pitch_rad": -0.015396,
    "optics_pixel_size_um": 2.952,
    "z12": 1282,
    "sx0_mm": 1.292,
    "rotation_offset_deg": -21.5
  },
  "17keV": {
    "beam_pitch_rad": -0.015396,
    "optics_pixel_size_um": 2.952,
    "z12": 1213,
    "sx0_mm": -3.113,
    "rotation_offset_deg": -21.5
  }
}
```

Provenance:
- `z12`: 33keV `z1+z2 = 1282` (31 Mar 2026); 17keV `z12 = 1213` (re-derived
  10 Mar / 05 May / 12 May after `align_crot`).
- `sx0_mm`: 33keV `sx0 = 1.292` (31 Mar, KB focus); 17keV `sx0 = -3.113`
  (02 Jun cryo).
- `beam_pitch_rad`, `optics_pixel_size_um`, `rotation_offset_deg`: no clear
  per-energy values in the logbook — kept at current defaults for both
  (detector/geometry constants). 33keV beam-tracker `sz_angle = -0.015294`
  (31 Mar) confirms `-0.015396` is already correct for 33keV.

Only `z12` and `sx0_mm` differ between energies, which matches the physics.

## Architecture

### Storage / persistence

- **Bundled defaults**: a new `scan_plan/presets/` package directory with
  `33keV.json` and `17keV.json`, shipped read-only via `package-data`.
- **User overrides**: `~/.config/scan_planner/presets.json` (XDG-aware:
  honour `$XDG_CONFIG_HOME`, fall back to `~/.config`). Created on first edit.
  Holds the full preset set; once it exists it is authoritative, deep-merged
  *over* the bundled set so future bundled additions still appear.
- **Apptainer**: works as-is. `apptainer run`/`exec` bind-mounts the host
  `$HOME`, so writes to `~/.config/scan_planner/presets.json` land on the host
  and persist. (Only `--contain`/`--no-home` would break this; the project's
  `.def`/runscript does neither.)

New `io.py` helpers:
- `user_config_dir()` → `Path` to `$XDG_CONFIG_HOME/scan_planner` or
  `~/.config/scan_planner`.
- `load_bundled_presets()` → `dict[name -> optics dict]` read from
  `scan_plan/presets/*.json`.
- `load_presets()` → bundled deep-merged under user file (user wins).
- `save_presets(presets)` → write user file (creating the dir; errors logged,
  not raised).

### Runtime wiring (minimal blast radius)

The user config gains one new key: `active_preset` (a name string). In
`cli.main`, after `load_config`:

1. `presets = load_presets()`
2. `active = cfg.get('active_preset')` (fall back to first preset name if unset
   or unknown)
3. `cfg['optics'] = presets[active]` (override the instrument-default optics)

Everything downstream already reads `cfg.get('optics', {})`
(`RegistrationDialog`, `VolumeRegistration`), so no other call site changes.
`motor_limits` is left as merged by `load_config`.

### Wizard change — `ConfigDialog`

Add one row at the top of the **Session Settings** group:
- a **"Beam energy"** `QComboBox` populated with preset names, pre-selecting
  `active_preset`;
- an **"Edit presets…"** `QPushButton` that opens `PresetEditorDialog`.

On *Save & Continue*:
- the chosen preset name is written to user config as `active_preset` (via the
  existing `get_updates()` / `_update_user_config` path);
- `cfg['optics']` is set to the chosen preset's optics so the running session
  uses it immediately.

If the editor is used and saves, the combo is refreshed from `load_presets()`
so a just-edited preset applies cleanly.

### New `PresetEditorDialog`

Opened by "Edit presets…". A separate modal window:
- a preset selector (combo or list) for the available presets;
- for the selected preset, one labelled spinbox per optics field with units:
  - `beam_pitch_rad` — rad (QDoubleSpinBox, enough decimals, e.g. 6)
  - `optics_pixel_size_um` — µm
  - `z12` — mm
  - `sx0_mm` — mm
  - `rotation_offset_deg` — deg
- buttons:
  - **Save** → `save_presets(...)` to `~/.config/scan_planner/presets.json`;
  - **Restore bundled** → re-seed the *selected* preset from its shipped
    `scan_plan/presets/<name>.json`;
  - **Close**.

Scope: edit the two existing presets' values only. No add / rename / delete
(YAGNI; can be added later). Editing one preset and switching to another before
saving keeps unsaved edits in memory until Save/Close per the implementation
plan's choice (simplest: edits are committed to the in-memory dict on field
change; Save writes the whole dict; Restore bundled replaces the selected
preset in the in-memory dict).

## Testing

`tests/test_io.py`:
- `user_config_dir()` respects `$XDG_CONFIG_HOME` and falls back to `~/.config`.
- `load_bundled_presets()` returns both `33keV` and `17keV` with the seed values
  above.
- `load_presets()` / `save_presets()` round-trip through a tmp config dir.
- `load_presets()` deep-merges bundled under a partial user file (user wins,
  missing keys inherited).

Optionally a small check that `cfg['optics']` resolves to the active preset
given an `active_preset` key.

## Out of scope (YAGNI)

- Adding / renaming / deleting presets in the UI.
- Per-RT/cryo variants (single value per energy for now).
- Editing `motor_limits` through the preset editor.
