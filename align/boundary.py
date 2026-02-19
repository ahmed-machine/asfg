import numpy as np

def _generate_boundary_gcps(gcps, M_geo, img_width, img_height, spacing_px=500):
    if M_geo is None:
        return []
    if len(gcps) < 3:
        return []
    px_coords = np.array([(g[0], g[1]) for g in gcps])
    geo_coords = np.array([(g[2], g[3]) for g in gcps])
    n = len(gcps)
    A = np.zeros((2 * n, 6))
    b = np.zeros(2 * n)
    for i in range(n):
        x, y = px_coords[i]
        A[2 * i] = [x, y, 1, 0, 0, 0]
        A[2 * i + 1] = [0, 0, 0, x, y, 1]
        b[2 * i] = geo_coords[i, 0]
        b[2 * i + 1] = geo_coords[i, 1]
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    M_px2geo = np.array([[result[0], result[1], result[2]],
                          [result[3], result[4], result[5]]])
    def px_to_geo(px, py):
        gx = M_px2geo[0, 0] * px + M_px2geo[0, 1] * py + M_px2geo[0, 2]
        gy = M_px2geo[1, 0] * px + M_px2geo[1, 1] * py + M_px2geo[1, 2]
        return gx, gy
    edge_points = []
    for x in range(0, img_width, spacing_px): edge_points.append((float(x), 0.0))
    for x in range(0, img_width, spacing_px): edge_points.append((float(x), float(img_height - 1)))
    for y in range(spacing_px, img_height - spacing_px, spacing_px): edge_points.append((0.0, float(y)))
    for y in range(spacing_px, img_height - spacing_px, spacing_px): edge_points.append((float(img_width - 1), float(y)))
    corners = [(0.0, 0.0), (float(img_width - 1), 0.0), (0.0, float(img_height - 1)), (float(img_width - 1), float(img_height - 1))]
    for c in corners:
        if c not in edge_points: edge_points.append(c)
    real_px = np.array([(g[0], g[1]) for g in gcps])
    boundary_gcps = []
    min_dist = spacing_px * 0.5
    for px, py in edge_points:
        dists = np.sqrt((real_px[:, 0] - px) ** 2 + (real_px[:, 1] - py) ** 2)
        if np.min(dists) > min_dist:
            gx, gy = px_to_geo(px, py)
            boundary_gcps.append((px, py, gx, gy))
    return boundary_gcps