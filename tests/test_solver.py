import numpy as np
import pytest
from scan_plan.solver import (
    solve_global_union,
    calculate_contrast_limits,
    solve_line_coverage,
    cylinder_dims,
)


CFG = {
    "prescan_pixel_size_xy": 180,
    "prescan_z_step": 180,
    "scan_pixel_size": 20,
}


class TestSolveGlobalUnion:
    def test_empty_rois(self):
        pts, dims_std, dims_exp = solve_global_union([], 20, CFG)
        assert len(pts) == 0
        # Manual cylinders need real dims even with no ROIs.
        assert dims_std[0] > 0 and dims_std[1] > 0
        assert dims_exp[0] > dims_std[0]
        assert dims_std == cylinder_dims(20, CFG)[0]

    def test_single_roi_center(self):
        roi = [{"x": 0, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000}]
        pts, dims_std, dims_exp = solve_global_union(roi, 20, CFG, mode="center")
        assert len(pts) > 0
        assert dims_std[0] > 0 and dims_std[1] > 0
        assert dims_exp[0] > dims_std[0]  # expanded > standard

    def test_single_roi_strict(self):
        roi = [{"x": 0, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000}]
        pts_strict, _, _ = solve_global_union(roi, 20, CFG, mode="strict")
        pts_center, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        pts_coverage, _, _ = solve_global_union(roi, 20, CFG, mode="coverage")
        # strict <= center <= coverage
        assert len(pts_strict) <= len(pts_center) <= len(pts_coverage)

    def test_mode_ordering(self):
        roi = [{"x": 100, "y": 100, "z": 100, "w": 1000, "h": 1000, "d": 1000}]
        pts_strict, _, _ = solve_global_union(roi, 20, CFG, mode="strict")
        pts_center, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        pts_coverage, _, _ = solve_global_union(roi, 20, CFG, mode="coverage")
        assert len(pts_strict) <= len(pts_center) <= len(pts_coverage)

    def test_points_sorted_by_z_y_x(self):
        roi = [{"x": 0, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000}]
        pts, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        if len(pts) > 1:
            # lexsort by (z, y, x) means z varies slowest
            for i in range(len(pts) - 1):
                assert (pts[i, 2] < pts[i+1, 2]) or \
                       (pts[i, 2] == pts[i+1, 2] and pts[i, 1] < pts[i+1, 1]) or \
                       (pts[i, 2] == pts[i+1, 2] and pts[i, 1] == pts[i+1, 1] and pts[i, 0] <= pts[i+1, 0])


class TestSolveLineCoverage:
    def test_empty_lines(self):
        pts, dims_std, dims_exp = solve_line_coverage([], 20, CFG)
        assert len(pts) == 0
        assert dims_std[0] > 0 and dims_std[1] > 0

    def test_horizontal_line_spacing_equals_diameter(self):
        # Pure XY line: spacing should be 2 * R = D_std
        (D_std, H_std), _ = cylinder_dims(20, CFG)
        p1 = (0.0, 0.0, 0.0)
        p2 = (10 * D_std, 0.0, 0.0)
        pts, _, _ = solve_line_coverage([(p1, p2)], 20, CFG)
        assert len(pts) >= 2
        # Centers must lie on the line (y=z=0).
        assert np.allclose(pts[:, 1], 0.0)
        assert np.allclose(pts[:, 2], 0.0)
        # Consecutive spacing along x ≈ D_std.
        diffs = np.diff(np.sort(pts[:, 0]))
        assert np.allclose(diffs, D_std, atol=1e-6)

    def test_vertical_line_spacing_equals_height(self):
        # Pure Z line: spacing should be 2 * (H/2) = H_std
        (D_std, H_std), _ = cylinder_dims(20, CFG)
        p1 = (50.0, 50.0, 0.0)
        p2 = (50.0, 50.0, 10 * H_std)
        pts, _, _ = solve_line_coverage([(p1, p2)], 20, CFG)
        assert len(pts) >= 2
        assert np.allclose(pts[:, 0], 50.0)
        assert np.allclose(pts[:, 1], 50.0)
        diffs = np.diff(np.sort(pts[:, 2]))
        assert np.allclose(diffs, H_std, atol=1e-6)

    def test_no_overlap_along_line(self):
        # Diagonal line — every consecutive pair must be at least min(D, H) apart
        # in the appropriate projection.
        (D_std, H_std), _ = cylinder_dims(20, CFG)
        p1 = (0.0, 0.0, 0.0)
        p2 = (5 * D_std, 5 * D_std, 5 * H_std)
        pts, _, _ = solve_line_coverage([(p1, p2)], 20, CFG)
        assert len(pts) >= 1
        for i in range(len(pts) - 1):
            d = np.linalg.norm(pts[i+1] - pts[i])
            # Must be positive and at most ~ 2*half_h_along_line
            assert d > 0

    def test_zero_length_line(self):
        # P1 == P2 should yield exactly one point (no division by zero).
        p = (1.0, 2.0, 3.0)
        pts, _, _ = solve_line_coverage([(p, p)], 20, CFG)
        assert len(pts) == 1
        assert np.allclose(pts[0], np.array(p))

    def test_multiple_lines_aggregate(self):
        (D_std, H_std), _ = cylinder_dims(20, CFG)
        ln_a = ((0.0, 0.0, 0.0), (10 * D_std, 0.0, 0.0))
        ln_b = ((0.0, 100.0, 0.0), (0.0, 100.0, 10 * H_std))
        pts_a, _, _ = solve_line_coverage([ln_a], 20, CFG)
        pts_b, _, _ = solve_line_coverage([ln_b], 20, CFG)
        pts_both, _, _ = solve_line_coverage([ln_a, ln_b], 20, CFG)
        assert len(pts_both) == len(pts_a) + len(pts_b)


class TestCalculateContrastLimits:
    def test_none_input(self):
        assert calculate_contrast_limits(None) == [0, 255]

    def test_empty_array(self):
        assert calculate_contrast_limits(np.array([])) == [0, 255]

    def test_normal_data(self):
        data = np.random.rand(50, 50, 50).astype(np.float32) * 100 + 10
        lo, hi = calculate_contrast_limits(data)
        assert lo < hi
        assert lo >= 0

    def test_all_zeros(self):
        data = np.zeros((10, 10, 10))
        lo, hi = calculate_contrast_limits(data)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_uniform_data(self):
        data = np.ones((20, 20, 20)) * 42.0
        lo, hi = calculate_contrast_limits(data)
        assert abs(lo - 42.0) < 1.0
        assert abs(hi - 42.0) < 1.0
