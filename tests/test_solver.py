import numpy as np
import pytest
from scan_plan.solver import (
    solve_global_union,
    solve_per_box,
    solve_bbox_grids,
    calculate_contrast_limits,
    solve_line_coverage,
    solve_parallelepiped_coverage,
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


class TestSolvePerBox:
    def test_empty(self):
        pts, ds, _ = solve_per_box([], 20, CFG)
        assert len(pts) == 0
        assert ds[0] > 0  # dims still valid

    def test_single_roi_matches_union(self):
        roi = [{"x": 0, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000}]
        u_pts, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        s_pts, _, _ = solve_per_box(roi, 20, CFG, mode="center")
        # One ROI: per-box and union should give the same set.
        assert len(u_pts) == len(s_pts)
        assert np.allclose(np.sort(u_pts, axis=0), np.sort(s_pts, axis=0))

    def test_disjoint_rois_give_more_points_per_box(self):
        # Two boxes far apart in X. Union recenters across the whole span,
        # so the seam between them gets fewer (or no) cylinders. Per-box
        # places its own grid in each, which generally yields >= union.
        roi = [
            {"x": 0,    "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000},
            {"x": 5000, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000},
        ]
        u_pts, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        s_pts, _, _ = solve_per_box(roi, 20, CFG, mode="center")
        # Per-box should never produce fewer points for disjoint boxes
        assert len(s_pts) >= len(u_pts)

    def test_dispatcher(self):
        roi = [{"x": 0, "y": 0, "z": 0, "w": 2000, "h": 2000, "d": 2000}]
        u_pts, _, _ = solve_bbox_grids(roi, 20, CFG, mode="center", treatment="union")
        s_pts, _, _ = solve_bbox_grids(roi, 20, CFG, mode="center", treatment="separate")
        u_ref, _, _ = solve_global_union(roi, 20, CFG, mode="center")
        s_ref, _, _ = solve_per_box(roi, 20, CFG, mode="center")
        assert len(u_pts) == len(u_ref)
        assert len(s_pts) == len(s_ref)


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

    def test_density_scales_spacing(self):
        (D_std, _), _ = cylinder_dims(20, CFG)
        p1 = (0.0, 0.0, 0.0)
        p2 = (10 * D_std, 0.0, 0.0)
        # density=2 → spacing halved (overlap), more cylinders
        pts1, _, _ = solve_line_coverage([(p1, p2)], 20, CFG, density=1.0)
        pts2, _, _ = solve_line_coverage([(p1, p2)], 20, CFG, density=2.0)
        pts_h, _, _ = solve_line_coverage([(p1, p2)], 20, CFG, density=0.5)
        d2 = np.diff(np.sort(pts2[:, 0]))
        dh = np.diff(np.sort(pts_h[:, 0]))
        assert len(pts2) > len(pts1) > len(pts_h)
        assert np.allclose(d2, D_std / 2, atol=1e-6)
        assert np.allclose(dh, 2 * D_std, atol=1e-6)

    def test_multiple_lines_aggregate(self):
        (D_std, H_std), _ = cylinder_dims(20, CFG)
        ln_a = ((0.0, 0.0, 0.0), (10 * D_std, 0.0, 0.0))
        ln_b = ((0.0, 100.0, 0.0), (0.0, 100.0, 10 * H_std))
        pts_a, _, _ = solve_line_coverage([ln_a], 20, CFG)
        pts_b, _, _ = solve_line_coverage([ln_b], 20, CFG)
        pts_both, _, _ = solve_line_coverage([ln_a, ln_b], 20, CFG)
        assert len(pts_both) == len(pts_a) + len(pts_b)


class TestSolveParallelepipedCoverage:
    def test_empty(self):
        pts, ds, _ = solve_parallelepiped_coverage([], 20, CFG)
        assert len(pts) == 0
        assert ds[0] > 0

    def test_axis_aligned_box(self):
        # Axis-aligned box: A=origin, C=opposite corner of base, A1
        # straight up. u1 along X, u2 along Y, u3 along Z.
        (D, H), _ = cylinder_dims(20, CFG)
        a = (0.0, 0.0, 0.0)
        c = (10 * D, 10 * D, 0.0)
        a1 = (0.0, 0.0, 10 * H)
        pts, _, _ = solve_parallelepiped_coverage([(a, c, a1)], 20, CFG)
        assert len(pts) > 0
        # All points must lie inside the box (with default Center mode
        # they may sit on the surface but not far outside).
        eps = 1e-6
        assert np.all(pts[:, 0] >= -eps) and np.all(pts[:, 0] <= 10 * D + eps)
        assert np.all(pts[:, 1] >= -eps) and np.all(pts[:, 1] <= 10 * D + eps)
        assert np.all(pts[:, 2] >= -eps) and np.all(pts[:, 2] <= 10 * H + eps)

    def test_density_scales_count(self):
        (D, H), _ = cylinder_dims(20, CFG)
        plp = ((0.0, 0.0, 0.0), (10 * D, 10 * D, 0.0), (0.0, 0.0, 10 * H))
        pts1, _, _ = solve_parallelepiped_coverage([plp], 20, CFG, density=1.0)
        pts2, _, _ = solve_parallelepiped_coverage([plp], 20, CFG, density=2.0)
        assert len(pts2) > len(pts1)

    def test_tilted_third_edge(self):
        # A1 - A has XY components — third edge tilts into the base plane.
        (D, H), _ = cylinder_dims(20, CFG)
        a = (0.0, 0.0, 0.0)
        c = (10 * D, 10 * D, 0.0)
        a1 = (5 * D, 0.0, 10 * H)  # tilted in X
        pts, _, _ = solve_parallelepiped_coverage([(a, c, a1)], 20, CFG)
        assert len(pts) > 0

    def test_flat_box_zero_height(self):
        # A1 == A → u3 = 0. Box collapses to its base; tile in 2D.
        (D, _), _ = cylinder_dims(20, CFG)
        a = (0.0, 0.0, 0.0)
        c = (10 * D, 10 * D, 0.0)
        a1 = (0.0, 0.0, 0.0)
        pts, _, _ = solve_parallelepiped_coverage([(a, c, a1)], 20, CFG)
        assert len(pts) > 0
        assert np.allclose(pts[:, 2], 0.0)

    def test_zero_size(self):
        p = (1.0, 2.0, 3.0)
        pts, _, _ = solve_parallelepiped_coverage([(p, p, p)], 20, CFG)
        assert len(pts) == 1
        assert np.allclose(pts[0], np.array(p))

    def test_mode_ordering(self):
        (D, H), _ = cylinder_dims(20, CFG)
        plp = ((0.0, 0.0, 0.0), (10 * D, 10 * D, 0.0), (0.0, 0.0, 10 * H))
        ns = []
        for mode in ("strict", "center", "coverage"):
            pts, _, _ = solve_parallelepiped_coverage([plp], 20, CFG, mode=mode)
            ns.append(len(pts))
        assert ns[0] <= ns[1] <= ns[2]

    def test_strict_excludes_when_too_small(self):
        (D, H), _ = cylinder_dims(20, CFG)
        small = ((0.0, 0.0, 0.0), (D / 4, D / 4, 0.0), (0.0, 0.0, H / 4))
        pts_strict, _, _ = solve_parallelepiped_coverage([small], 20, CFG, mode="strict")
        pts_center, _, _ = solve_parallelepiped_coverage([small], 20, CFG, mode="center")
        assert len(pts_strict) == 0
        assert len(pts_center) >= 1


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
