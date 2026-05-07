"""
Pure math helpers: cylinder grid solving and contrast limit calculation.
"""

import math
import numpy as np


def calculate_contrast_limits(data_array, fraction=0.5):
    if data_array is None or data_array.size == 0:
        return [0, 255]
    s = data_array.shape
    c = [int(d * (1 - fraction) / 2) for d in s]
    e = [max(c[i] + 1, int(s[i] * (1 + fraction) / 2)) for i in range(len(s))]
    crop = data_array[c[0]:e[0], c[1]:e[1], c[2]:e[2]]
    if crop.size == 0: crop = data_array
    try:
        nz = crop[crop > 0]
        if nz.size == 0: nz = crop
        return [float(np.percentile(nz, 2)), float(np.percentile(nz, 98))]
    except Exception:
        return [0.0, 255.0]


def cylinder_dims(scan_res_nm, config):
    """Return ((D_std, H_std), (D_exp, H_exp)) in prescan-pixel units."""
    size_std = scan_res_nm * 2048
    size_exp = scan_res_nm * 3216
    D_std = int(np.floor(size_std / config["prescan_pixel_size_xy"]))
    D_exp = int(np.floor(size_exp / config["prescan_pixel_size_xy"]))
    H_std = int(np.floor(size_std / config["prescan_z_step"]))
    H_exp = int(np.floor(size_exp / config["prescan_z_step"]))
    return (D_std, H_std), (D_exp, H_exp)


def _solve_grid(roi_list, scan_res_nm, config, mode="center"):
    """Compute cylinder grid points whose union covers *roi_list*.

    All ROIs share a single grid; the centering logic uses the union's
    bounding span. This is the per-batch primitive used by both
    solve_global_union (one batch = all ROIs) and solve_per_box (one
    batch per ROI).
    """
    (D_std, H_std), (D_exp, H_exp) = cylinder_dims(scan_res_nm, config)
    if not roi_list:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)
    if D_std == 0 or H_std == 0:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    R = D_std / 2.0

    min_x = min(r['x'] for r in roi_list)
    min_y = min(r['y'] for r in roi_list)
    min_z = min(r['z'] for r in roi_list)
    max_x = max(r['x'] + r['w'] for r in roi_list)
    max_y = max(r['y'] + r['h'] for r in roi_list)
    max_z = max(r['z'] + r['d'] for r in roi_list)

    span_x, span_y, span_z = max_x - min_x, max_y - min_y, max_z - min_z
    count_x, count_y, count_z = math.floor(span_x/D_std), math.floor(span_y/D_std), math.floor(span_z/H_std)
    center_start_x = min_x + (span_x - (count_x*D_std))/2.0
    center_start_y = min_y + (span_y - (count_y*D_std))/2.0
    center_start_z = min_z + (span_z - (count_z*H_std))/2.0

    extra = 4
    x_coords = center_start_x + (np.arange(-extra, count_x + extra + 1) * D_std)
    y_coords = center_start_y + (np.arange(-extra, count_y + extra + 1) * D_std)
    z_coords = center_start_z + (np.arange(-extra, count_z + extra + 1) * H_std)

    if len(x_coords)==0 or len(y_coords)==0 or len(z_coords)==0:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    candidates = np.stack([gx + D_std/2, gy + D_std/2, gz + H_std/2], axis=-1).reshape(-1, 3)
    mask = np.zeros(len(candidates), dtype=bool)
    cx, cy, cz = candidates[:,0], candidates[:,1], candidates[:,2]
    cyl_min_x, cyl_max_x = cx - R, cx + R
    cyl_min_y, cyl_max_y = cy - R, cy + R
    cyl_min_z, cyl_max_z = cz - H_std/2.0, cz + H_std/2.0

    for roi in roi_list:
        rx, ry, rz = roi['x'], roi['y'], roi['z']
        roi_max_x, roi_max_y, roi_max_z = rx + roi['w'], ry + roi['h'], rz + roi['d']

        if mode == "strict":
            in_roi = (cyl_min_x >= rx) & (cyl_max_x <= roi_max_x) & \
                     (cyl_min_y >= ry) & (cyl_max_y <= roi_max_y) & \
                     (cyl_min_z >= rz) & (cyl_max_z <= roi_max_z)
        elif mode == "coverage":
            in_roi = (cyl_max_x >= rx) & (cyl_min_x <= roi_max_x) & \
                     (cyl_max_y >= ry) & (cyl_min_y <= roi_max_y) & \
                     (cyl_max_z >= rz) & (cyl_min_z <= roi_max_z)
        else:  # center
            in_roi = (cx >= rx) & (cx <= roi_max_x) & \
                     (cy >= ry) & (cy <= roi_max_y) & \
                     (cz >= rz) & (cz <= roi_max_z)
        mask = mask | in_roi

    final_points = candidates[mask]
    if len(final_points) > 0:
        sort_indices = np.lexsort((final_points[:, 0], final_points[:, 1], final_points[:, 2]))
        final_points = final_points[sort_indices]

    return final_points, (D_std, H_std), (D_exp, H_exp)


def solve_global_union(roi_list, scan_res_nm, config, mode="center"):
    """One global grid that covers the union of all ROIs."""
    return _solve_grid(roi_list, scan_res_nm, config, mode)


def solve_per_box(roi_list, scan_res_nm, config, mode="center"):
    """Independent grid per ROI; cylinders never cross box boundaries.

    Adjacent or overlapping ROIs may produce duplicate cylinders; the
    output is deduplicated on (x, y, z) at integer-pixel granularity.
    """
    (D_std, H_std), (D_exp, H_exp) = cylinder_dims(scan_res_nm, config)
    if not roi_list:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    chunks = []
    for roi in roi_list:
        pts, _, _ = _solve_grid([roi], scan_res_nm, config, mode)
        if len(pts) > 0:
            chunks.append(pts)
    if not chunks:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    pts = np.vstack(chunks)
    # Dedupe at sub-pixel granularity so two boxes meeting along a face
    # don't both contribute the same border cylinder.
    rounded = np.round(pts * 1000.0).astype(np.int64)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    pts = pts[np.sort(unique_idx)]
    sort_idx = np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))
    return pts[sort_idx], (D_std, H_std), (D_exp, H_exp)


def solve_bbox_grids(roi_list, scan_res_nm, config, mode="center", treatment="union"):
    """Dispatcher between solve_global_union and solve_per_box.

    *treatment* is "union" (single global grid) or "separate" (per-ROI grids).
    """
    if treatment == "separate":
        return solve_per_box(roi_list, scan_res_nm, config, mode)
    return solve_global_union(roi_list, scan_res_nm, config, mode)


def solve_line_coverage(line_list, scan_res_nm, config, density=1.0):
    """Distribute cylinder centers along each (P1, P2) line.

    Each line is a tuple (p1, p2) where p1, p2 are length-3 sequences in
    prescan-pixel coordinates. Cylinders are anisotropic: radius R = D_std/2
    in xy and half-height H_std/2 in z. Along a unit direction (dx,dy,dz)
    a cylinder centered on the line covers a half-length of
        h = min( R / sqrt(dx^2 + dy^2),  (H_std/2) / |dz| )
    so adjacent centers at density=1 are spaced by 2*h (touching, no overlap).

    *density* > 1 packs cylinders tighter (they overlap); *density* < 1 leaves
    gaps. Spacing along the line is (2*h)/density.

    Returns (points, (D_std, H_std), (D_exp, H_exp)).
    """
    (D_std, H_std), (D_exp, H_exp) = cylinder_dims(scan_res_nm, config)
    if not line_list or D_std <= 0 or H_std <= 0:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    R = D_std / 2.0
    half_h_z = H_std / 2.0
    density = max(float(density), 1e-6)

    out = []
    for p1, p2 in line_list:
        a = np.asarray(p1, dtype=float)
        b = np.asarray(p2, dtype=float)
        seg = b - a
        L = np.linalg.norm(seg)
        if L == 0:
            out.append(a)
            continue
        u = seg / L
        dx, dy, dz = u
        radial = np.hypot(dx, dy)
        cand = []
        if radial > 1e-12:
            cand.append(R / radial)
        if abs(dz) > 1e-12:
            cand.append(half_h_z / abs(dz))
        half_len = min(cand) if cand else L
        spacing = (2.0 * half_len) / density
        if spacing <= 0:
            out.append(a)
            continue
        # Start one half-length in from p1 so the first cylinder's far edge
        # tangents p1; tile until the far edge passes p2.
        n = max(1, int(np.ceil((L - half_len) / spacing)) + 1)
        # Recenter the chain so it's symmetric on the segment when possible.
        used = (n - 1) * spacing
        offset = (L - used) / 2.0
        if offset < 0:
            offset = 0.0
        for k in range(n):
            t = offset + k * spacing
            out.append(a + u * t)

    if not out:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)
    pts = np.array(out)
    sort_idx = np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))
    return pts[sort_idx], (D_std, H_std), (D_exp, H_exp)


def _half_reach_along(unit_vec, R, half_h_z):
    """Half-length a cylinder-on-line covers along *unit_vec* before exiting
    the cylinder surface. unit_vec must already be unit-length.
    """
    dx, dy, dz = unit_vec
    radial = np.hypot(dx, dy)
    cand = []
    if radial > 1e-12:
        cand.append(R / radial)
    if abs(dz) > 1e-12:
        cand.append(half_h_z / abs(dz))
    return min(cand) if cand else float("inf")


def _tile_axis(L, half_reach, spacing, mode):
    """Return (n, first_offset) for placing N cylinders along a 1D edge of
    length *L* with anisotropic half-reach *half_reach* and packing
    *spacing*. *mode* matches the bbox semantics:

      strict   — cylinder fully inside the edge (centers >= half_reach
                 from each end). Yields 0 if the edge is shorter than 2h.
      center   — cylinder centers must lie on [0, L] (some surface area
                 of the cylinders overhangs by up to half_reach).
      coverage — cylinder partially intersects the edge (centers may be
                 up to half_reach beyond either end).
    """
    if mode == "strict":
        L_eff = L - 2.0 * half_reach
        base = half_reach
    elif mode == "coverage":
        L_eff = L + 2.0 * half_reach
        base = -half_reach
    else:  # center / default
        L_eff = L
        base = 0.0
    if spacing <= 0 or L_eff < 0:
        return 0, 0.0
    n = max(1, int(L_eff // spacing) + 1)
    used = (n - 1) * spacing
    extra = max(0.0, L_eff - used)
    return n, base + extra / 2.0


def solve_parallelogram_coverage(plg_list, scan_res_nm, config, density=1.0, mode="center"):
    """Tile cylinders across each parallelogram defined by three corners.

    Each entry is a tuple (p0, p1, p2). The parallelogram has corners
    P0, P1, P1+(P2-P0), P2 — i.e. P0–P1 is one edge, P0–P2 is the other.
    Cylinders are placed on a regular 2D grid spanning the (u1, u2) plane.

    Spacing along each edge follows the same anisotropic logic as
    solve_line_coverage. *density* > 1 packs tighter (overlap), *density*
    < 1 leaves gaps. *mode* mirrors the bbox Fill Mode (strict / center /
    coverage), applied independently to each edge axis.

    Returns (points, (D_std, H_std), (D_exp, H_exp)).
    """
    (D_std, H_std), (D_exp, H_exp) = cylinder_dims(scan_res_nm, config)
    if not plg_list or D_std <= 0 or H_std <= 0:
        return np.empty((0, 3)), (D_std, H_std), (D_exp, H_exp)

    R = D_std / 2.0
    half_h_z = H_std / 2.0
    density = max(float(density), 1e-6)
    mode = (mode or "center").lower()

    out = []
    for p0, p1, p2 in plg_list:
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        c = np.asarray(p2, dtype=float)
        u1 = b - a
        u2 = c - a
        L1 = float(np.linalg.norm(u1))
        L2 = float(np.linalg.norm(u2))
        if L1 == 0 and L2 == 0:
            out.append(a)
            continue
        if L1 == 0:
            # Degenerates to a line along u2.
            half2 = _half_reach_along(u2 / L2, R, half_h_z)
            spacing2 = (2.0 * half2) / density
            n2, off2 = _tile_axis(L2, half2, spacing2, mode)
            for j in range(n2):
                out.append(a + (off2 + j * spacing2) * (u2 / L2))
            continue
        if L2 == 0:
            half1 = _half_reach_along(u1 / L1, R, half_h_z)
            spacing1 = (2.0 * half1) / density
            n1, off1 = _tile_axis(L1, half1, spacing1, mode)
            for i in range(n1):
                out.append(a + (off1 + i * spacing1) * (u1 / L1))
            continue

        u1h = u1 / L1
        u2h = u2 / L2
        half1 = _half_reach_along(u1h, R, half_h_z)
        half2 = _half_reach_along(u2h, R, half_h_z)
        spacing1 = (2.0 * half1) / density
        spacing2 = (2.0 * half2) / density
        n1, off1 = _tile_axis(L1, half1, spacing1, mode)
        n2, off2 = _tile_axis(L2, half2, spacing2, mode)
        if n1 == 0 or n2 == 0:
            continue

        for j in range(n2):
            for i in range(n1):
                t1 = off1 + i * spacing1
                t2 = off2 + j * spacing2
                out.append(a + t1 * u1h + t2 * u2h)

    if not out:
        return np.empty((0, 3)), (D_std, H_std), (D_exp, H_exp)
    pts = np.array(out)
    sort_idx = np.lexsort((pts[:, 0], pts[:, 1], pts[:, 2]))
    return pts[sort_idx], (D_std, H_std), (D_exp, H_exp)
