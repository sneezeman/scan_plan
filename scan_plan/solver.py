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


def solve_global_union(roi_list, scan_res_nm, config, mode="center"):
    (D_std, H_std), (D_exp, H_exp) = cylinder_dims(scan_res_nm, config)
    if not roi_list:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

    R = D_std / 2.0

    min_x = min(r['x'] for r in roi_list)
    min_y = min(r['y'] for r in roi_list)
    min_z = min(r['z'] for r in roi_list)
    max_x = max(r['x'] + r['w'] for r in roi_list)
    max_y = max(r['y'] + r['h'] for r in roi_list)
    max_z = max(r['z'] + r['d'] for r in roi_list)

    if D_std == 0 or H_std == 0:
        return np.empty((0,3)), (D_std, H_std), (D_exp, H_exp)

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
