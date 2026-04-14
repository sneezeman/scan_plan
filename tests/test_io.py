import json
import os
import pytest
from scan_plan.io import parse_nml, load_config, load_volume


FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestParseNml:
    def test_valid_nml(self):
        rois = parse_nml(os.path.join(FIXTURES, "sample.nml"))
        assert len(rois) == 2
        assert rois[0] == {"x": 100, "y": 200, "z": 300, "w": 500, "h": 600, "d": 700}
        assert rois[1] == {"x": 800, "y": 900, "z": 1000, "w": 400, "h": 300, "d": 200}

    def test_missing_file(self):
        assert parse_nml("/nonexistent/path.nml") == []

    def test_invalid_xml(self, tmp_path):
        bad_file = tmp_path / "bad.nml"
        bad_file.write_text("this is not xml")
        assert parse_nml(str(bad_file)) == []

    def test_empty_nml(self, tmp_path):
        empty = tmp_path / "empty.nml"
        empty.write_text('<?xml version="1.0"?><things><parameters></parameters></things>')
        assert parse_nml(str(empty)) == []


class TestLoadConfig:
    def test_creates_default(self, tmp_path):
        cfg_path = tmp_path / "new_config.json"
        cfg = load_config(str(cfg_path))
        assert cfg_path.exists()
        assert cfg["binning"] == 1
        assert "volume_path" in cfg

    def test_loads_existing(self, tmp_path):
        cfg_path = tmp_path / "existing.json"
        cfg_path.write_text(json.dumps({"binning": 4, "volume_path": "/test", "prescan_pixel_size_xy": 100,
                                         "prescan_z_step": 100, "scan_pixel_size": 10, "raw_dims": [1,1,1],
                                         "raw_dtype": "float32", "rois": []}))
        cfg = load_config(str(cfg_path))
        assert cfg["binning"] == 4

    def test_bad_json_returns_default(self, tmp_path):
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text("not valid json {{{")
        cfg = load_config(str(cfg_path))
        assert cfg["binning"] == 1  # default

    def test_copies_instrument_defaults_next_to_config(self, tmp_path):
        """instrument_defaults.json should be copied to the config's dir on first run."""
        cfg_path = tmp_path / "new_config.json"
        instrument_path = tmp_path / "instrument_defaults.json"
        assert not instrument_path.exists()
        cfg = load_config(str(cfg_path))
        assert instrument_path.exists(), "instrument_defaults.json was not copied"
        # Values from instrument_defaults.json should be merged into the config
        assert "optics" in cfg
        assert "motor_limits" in cfg

    def test_user_edits_to_instrument_defaults_are_respected(self, tmp_path):
        """Editing the working-dir instrument_defaults.json should override bundled values."""
        cfg_path = tmp_path / "cfg.json"
        instrument_path = tmp_path / "instrument_defaults.json"
        # First run creates the copy
        load_config(str(cfg_path))
        # Edit the local copy
        edited = {"optics": {"rotation_offset_deg": -99.9}, "motor_limits": {"su": [-1.0, 1.0]}}
        instrument_path.write_text(json.dumps(edited))
        # Second run should pick up edited values
        cfg = load_config(str(cfg_path))
        assert cfg["optics"]["rotation_offset_deg"] == -99.9
        assert cfg["motor_limits"]["su"] == [-1.0, 1.0]


class TestLoadVolume:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_volume("/nonexistent/file.vol", (10,10,10), "float32", 1)

    def test_bad_extension_raises(self, tmp_path):
        bad = tmp_path / "data.xyz"
        bad.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported"):
            load_volume(str(bad), (10,10,10), "float32", 1)
