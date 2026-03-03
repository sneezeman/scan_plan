import numpy as np
import pytest
from scan_plan.solver import solve_global_union, calculate_contrast_limits


CFG = {
    "prescan_pixel_size_xy": 180,
    "prescan_z_step": 180,
    "scan_pixel_size": 20,
}


class TestSolveGlobalUnion:
    def test_empty_rois(self):
        pts, dims_std, dims_exp = solve_global_union([], 20, CFG)
        assert len(pts) == 0
        assert dims_std == (0, 0)

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
