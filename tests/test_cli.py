from scan_plan.cli import _resolve_optics


PRESETS = {
    "33keV": {"z12": 1282, "sx0_mm": 1.292},
    "17keV": {"z12": 1213, "sx0_mm": -3.113},
}


class TestResolveOptics:
    def test_uses_active_preset(self):
        cfg = {"active_preset": "17keV"}
        active = _resolve_optics(cfg, PRESETS)
        assert active == "17keV"
        assert cfg["optics"] == PRESETS["17keV"]

    def test_missing_active_falls_back_to_sorted_first(self):
        cfg = {}
        active = _resolve_optics(cfg, PRESETS)
        # sorted(["33keV", "17keV"])[0] == "17keV"
        assert active == "17keV"
        assert cfg["active_preset"] == "17keV"
        assert cfg["optics"] == PRESETS["17keV"]

    def test_unknown_active_falls_back_and_is_rewritten(self):
        cfg = {"active_preset": "99keV"}
        active = _resolve_optics(cfg, PRESETS)
        assert active == "17keV"
        assert cfg["active_preset"] == "17keV"

    def test_resolves_freshly_after_preset_change(self):
        # Simulates the wizard switching the active preset: a re-resolve must
        # swap cfg['optics'] to match the new selection.
        cfg = {"active_preset": "17keV"}
        _resolve_optics(cfg, PRESETS)
        cfg["active_preset"] = "33keV"  # user changed it in the wizard
        _resolve_optics(cfg, PRESETS)
        assert cfg["optics"] == PRESETS["33keV"]
