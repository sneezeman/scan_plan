import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
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

    # ---- New tests: non-trivial transforms ----

    def _make_rotated_points(self, rotation, n_points=8, rng=None):
        """Generate random prescan points and apply a rotation to create refscan points.

        Returns (prescan_list, refscan_list) as lists of tuples, ready for addMatchPoint.
        The rotation is applied about the centroid so that centering in the
        fitting routine recovers the correct rotation.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        prescan = rng.uniform(200, 1800, size=(n_points, 3))
        refscan = rotation.apply(prescan)
        prescan_tuples = [tuple(row) for row in prescan]
        refscan_tuples = [tuple(row) for row in refscan]
        return prescan_tuples, refscan_tuples

    def test_svd_known_z_rotation(self):
        """SVD should recover a known 45-degree Z rotation to within 0.1 degrees."""
        angle_deg = 45.0
        rotation = R.from_euler('xyz', [0, 0, angle_deg], degrees=True)
        rng = np.random.default_rng(42)

        vreg = _make_vreg()
        prescan_pts, refscan_pts = self._make_rotated_points(rotation, n_points=8, rng=rng)
        for p, r in zip(prescan_pts, refscan_pts):
            vreg.addMatchPoint(p, r, 0)

        result = vreg.fitTransformationMatrix(rot_z_only=True, method='svd')

        # rotation_angles is [yaw, pitch, roll]; for rot_z_only, roll is the Z angle
        recovered_angle = result.rotation_angles[2]
        np.testing.assert_allclose(recovered_angle, angle_deg, atol=0.1,
                                   err_msg=f"SVD recovered {recovered_angle}, expected {angle_deg}")
        # Residuals should be near-zero for perfect data
        assert np.max(result.distances) < 1.0

    def test_optimizer_known_z_rotation(self):
        """Optimizer should recover a known 45-degree Z rotation to within 0.5 degrees."""
        angle_deg = 45.0
        rotation = R.from_euler('xyz', [0, 0, angle_deg], degrees=True)
        rng = np.random.default_rng(42)

        vreg = _make_vreg()
        prescan_pts, refscan_pts = self._make_rotated_points(rotation, n_points=8, rng=rng)
        for p, r in zip(prescan_pts, refscan_pts):
            vreg.addMatchPoint(p, r, 0)

        result = vreg.fitTransformationMatrix(rot_z_only=True, method='optimizer')

        recovered_angle = result.rotation_angles[2]
        np.testing.assert_allclose(recovered_angle, angle_deg, atol=0.5,
                                   err_msg=f"Optimizer recovered {recovered_angle}, expected {angle_deg}")
        assert np.max(result.distances) < 1.0

    def test_svd_x_flip_detection(self):
        """SVD should detect an X-flip and report low residual."""
        rng = np.random.default_rng(42)

        vreg = _make_vreg()
        prescan = rng.uniform(200, 1800, size=(8, 3))
        refscan = prescan.copy()
        refscan[:, 0] = -refscan[:, 0]  # flip X axis

        for i in range(len(prescan)):
            vreg.addMatchPoint(tuple(prescan[i]), tuple(refscan[i]), 0)

        result = vreg.fitTransformationMatrix(rot_z_only=True, method='svd')

        # Total residual should be low (X-flip is a clean transform)
        assert np.sum(result.distances) < 1.0, (
            f"Total residual {np.sum(result.distances)} too high for X-flip"
        )
        # Solution message should indicate X-flip was detected
        assert "X-flip" in result.solution.message, (
            f"Expected 'X-flip' in solution message, got: {result.solution.message}"
        )

    def test_optimizer_x_flip_detection(self):
        """Optimizer should detect an X-flip and report low residual."""
        rng = np.random.default_rng(42)

        vreg = _make_vreg()
        prescan = rng.uniform(200, 1800, size=(8, 3))
        refscan = prescan.copy()
        refscan[:, 0] = -refscan[:, 0]  # flip X axis

        for i in range(len(prescan)):
            vreg.addMatchPoint(tuple(prescan[i]), tuple(refscan[i]), 0)

        result = vreg.fitTransformationMatrix(rot_z_only=True, method='optimizer')

        assert np.sum(result.distances) < 1.0, (
            f"Total residual {np.sum(result.distances)} too high for X-flip"
        )
        assert "X-flip" in result.solution.message, (
            f"Expected 'X-flip' in solution message, got: {result.solution.message}"
        )

    def test_svd_full_3d_rotation(self):
        """SVD with rot_z_only=False should recover a full 3D rotation."""
        angles_deg = [10.0, 20.0, 30.0]  # yaw, pitch, roll
        rotation = R.from_euler('xyz', angles_deg, degrees=True)
        rng = np.random.default_rng(42)

        vreg = _make_vreg()
        prescan_pts, refscan_pts = self._make_rotated_points(rotation, n_points=10, rng=rng)
        for p, r in zip(prescan_pts, refscan_pts):
            vreg.addMatchPoint(p, r, 0)

        result = vreg.fitTransformationMatrix(rot_z_only=False, method='svd')

        # Check each angle is recovered approximately
        for i, (recovered, expected) in enumerate(zip(result.rotation_angles, angles_deg)):
            np.testing.assert_allclose(
                recovered, expected, atol=0.5,
                err_msg=f"Angle {i} (xyz): recovered {recovered}, expected {expected}"
            )
        # Residuals should be near-zero for perfect data
        assert np.max(result.distances) < 1.0

    def test_noisy_registration(self):
        """Both SVD and optimizer should tolerate moderate noise and still recover angles."""
        angle_deg = 30.0
        rotation = R.from_euler('xyz', [0, 0, angle_deg], degrees=True)
        rng = np.random.default_rng(42)
        noise_sigma = 2.0  # pixels

        n_points = 20  # more points to average out noise
        prescan = rng.uniform(200, 1800, size=(n_points, 3))
        refscan_clean = rotation.apply(prescan)
        refscan_noisy = refscan_clean + rng.normal(0, noise_sigma, size=refscan_clean.shape)

        for method in ('svd', 'optimizer'):
            vreg = _make_vreg()
            for i in range(n_points):
                vreg.addMatchPoint(tuple(prescan[i]), tuple(refscan_noisy[i]), 0)

            result = vreg.fitTransformationMatrix(rot_z_only=True, method=method)
            recovered_angle = result.rotation_angles[2]

            np.testing.assert_allclose(
                recovered_angle, angle_deg, atol=5.0,
                err_msg=f"{method}: recovered {recovered_angle}, expected ~{angle_deg} (noise sigma={noise_sigma})"
            )
            # Mean residual should be reasonable (same order as noise)
            assert np.mean(result.distances) < 10.0, (
                f"{method}: mean residual {np.mean(result.distances)} unreasonably large"
            )

    def test_minimum_points_raises(self):
        """Calling fitTransformationMatrix with fewer than 2 points should raise ValueError."""
        # Zero points
        vreg0 = _make_vreg()
        with pytest.raises(ValueError):
            vreg0.fitTransformationMatrix(rot_z_only=True, method='svd')

        # One point
        vreg1 = _make_vreg()
        vreg1.addMatchPoint((500, 500, 500), (500, 500, 500), 0)
        with pytest.raises(ValueError):
            vreg1.fitTransformationMatrix(rot_z_only=True, method='svd')
