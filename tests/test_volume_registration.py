import numpy as np
import pytest
from scan_plan.volume_registration import VolumeRegistration


def _make_vreg(optics=None):
    vreg = VolumeRegistration(180, optics=optics)
    vreg.addReferenceVolume(-0.5, 0.1, 1.5, 180)
    return vreg


def _add_test_points(vreg):
    """Add 4 matching points with known offsets for testing."""
    points = [
        ((100, 200, 300), (100, 200, 300)),
        ((400, 500, 600), (400, 500, 600)),
        ((700, 100, 400), (700, 100, 400)),
        ((200, 800, 150), (200, 800, 150)),
    ]
    for p, r in points:
        vreg.addMatchPoint(p, r, 0)
    return points


class TestSuvSaxyRoundtrip:
    def test_roundtrip_default_offset(self):
        vreg = _make_vreg()
        su_in, sv_in = 1.5, -0.3
        sax, say = vreg._suv2saxy(su_in, sv_in)
        su_out, sv_out = vreg._saxy2suv(sax, say)
        np.testing.assert_allclose(su_out, su_in, atol=1e-10)
        np.testing.assert_allclose(sv_out, sv_in, atol=1e-10)

    def test_roundtrip_custom_offset(self):
        vreg = _make_vreg(optics={"rotation_offset_deg": -10.0})
        su_in, sv_in = 2.0, 0.5
        sax, say = vreg._suv2saxy(su_in, sv_in)
        su_out, sv_out = vreg._saxy2suv(sax, say)
        np.testing.assert_allclose(su_out, su_in, atol=1e-10)
        np.testing.assert_allclose(sv_out, sv_in, atol=1e-10)


class TestOpticsConfig:
    def test_default_values(self):
        vreg = VolumeRegistration(180)
        assert vreg._rotation_offset == -21.5
        assert vreg._VolumeRegistration__optics_pixel_size == 2.952
        assert vreg._VolumeRegistration__z12 == 1281
        assert vreg._VolumeRegistration__sx0 == 1.28

    def test_custom_values(self):
        optics = {
            "optics_pixel_size_um": 3.0,
            "z12": 1500,
            "sx0_mm": 1.5,
            "rotation_offset_deg": -15.0,
            "beam_pitch_rad": -0.02,
        }
        vreg = VolumeRegistration(180, optics=optics)
        assert vreg._rotation_offset == -15.0
        assert vreg._VolumeRegistration__optics_pixel_size == 3.0
        assert vreg._VolumeRegistration__z12 == 1500
        assert vreg._VolumeRegistration__sx0 == 1.5
        assert vreg._VolumeRegistration__beam_pitch == -0.02


class TestMotorsToRefscan:
    def test_roundtrip(self):
        """refscan_to_motors -> motors_to_refscan should recover original coords."""
        vreg = _make_vreg()
        refscan_coords = np.array([
            [500, 600, 700],
            [800, 300, 400],
            [1024, 1024, 1024],
        ], dtype=float)
        scan_px = 100.0
        su, sv, sz = vreg.refscan_to_motors(refscan_coords, scan_px)
        recovered = vreg.motors_to_refscan(su, sv, sz, scan_px)
        np.testing.assert_allclose(recovered, refscan_coords, atol=1e-6)

    def test_roundtrip_custom_optics(self):
        """Roundtrip with custom optics parameters."""
        optics = {
            "optics_pixel_size_um": 3.0,
            "z12": 1500,
            "sx0_mm": 1.5,
            "rotation_offset_deg": -15.0,
            "beam_pitch_rad": -0.02,
        }
        vreg = _make_vreg(optics=optics)
        refscan_coords = np.array([
            [200, 1800, 100],
            [1500, 500, 1900],
        ], dtype=float)
        scan_px = 50.0
        su, sv, sz = vreg.refscan_to_motors(refscan_coords, scan_px)
        recovered = vreg.motors_to_refscan(su, sv, sz, scan_px)
        np.testing.assert_allclose(recovered, refscan_coords, atol=1e-6)

    def test_single_point_at_center(self):
        """A point at the refscan center should map to the reference volume motor coords."""
        vreg = _make_vreg()
        center = np.array([[1024, 1024, 1024]], dtype=float)
        scan_px = 100.0
        su, sv, sz = vreg.refscan_to_motors(center, scan_px)
        # At center, su/sv should be close to ref volume su/sv
        np.testing.assert_allclose(su, -0.5, atol=1e-4)
        np.testing.assert_allclose(sv, 0.1, atol=1e-4)
        # Roundtrip
        recovered = vreg.motors_to_refscan(su, sv, sz, scan_px)
        np.testing.assert_allclose(recovered, center, atol=1e-6)


class TestFitTransformation:
    def test_svd_identity_transform(self):
        """When prescan == refscan, transform should be near-identity."""
        vreg = _make_vreg()
        _add_test_points(vreg)
        result = vreg.fitTransformationMatrix(rot_z_only=True, method='svd')
        # Distances should be near zero for identity
        np.testing.assert_allclose(result.distances, 0, atol=1e-6)

    def test_optimizer_identity_transform(self):
        vreg = _make_vreg()
        _add_test_points(vreg)
        result = vreg.fitTransformationMatrix(rot_z_only=True, method='optimizer')
        np.testing.assert_allclose(result.distances, 0, atol=0.1)

    def test_svd_vs_optimizer_agreement(self):
        """Both methods should agree on mean error for the same data."""
        vreg_svd = _make_vreg()
        vreg_opt = _make_vreg()
        _add_test_points(vreg_svd)
        _add_test_points(vreg_opt)
        res_svd = vreg_svd.fitTransformationMatrix(rot_z_only=True, method='svd')
        res_opt = vreg_opt.fitTransformationMatrix(rot_z_only=True, method='optimizer')
        # Both should have very low error for identity-like transform
        assert np.mean(res_svd.distances) < 1.0
        assert np.mean(res_opt.distances) < 1.0
