"""TIN-TARR topological filtering and FPP Accuracy Difference optimization.

Implements the methods from:
  Guo et al. (2022) "Outlier removal and feature point pairs optimization for
  piecewise linear transformation in the co-registration of very high-resolution
  optical remote sensing imagery", ISPRS J. Photogrammetry & Remote Sensing.

Adapted for the declass-process pipeline which works in geographic coordinate
space (meters) rather than pixel space.

Phase A -- ``filter_by_tin_tarr``
    Removes matched pairs whose local Delaunay triangles exhibit extreme area
    distortion between the reference and offset coordinate systems (TIN-TARR
    metric).  Both coordinate sets must be in the same unit system (e.g. UTM
    meters) for the area ratio to be meaningful.

Phase B -- ``optimize_fpps_accuracy``
    Iteratively removes GCPs whose deletion *improves* the local image
    similarity (ZNCC) of the piecewise-affine warp.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay, ConvexHull

from .types import MatchPair


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _triangle_area(tri_pts):
    """Unsigned area of a triangle from a (3, 2) array of vertices."""
    return abs(
        tri_pts[0, 0] * tri_pts[1, 1]
        + tri_pts[1, 0] * tri_pts[2, 1]
        + tri_pts[2, 0] * tri_pts[0, 1]
        - tri_pts[2, 0] * tri_pts[1, 1]
        - tri_pts[0, 0] * tri_pts[2, 1]
        - tri_pts[1, 0] * tri_pts[0, 1]
    ) / 2.0


def _tin_tarr_for_point(pt_idx, tin, ref_pts, off_pts):
    """Mean triangle-area representation ratio for a single point.

    Returns the mean ratio across all triangles adjacent to *pt_idx*.
    A ratio of 1.0 means no distortion; higher values mean more distortion.
    """
    connected = np.any(tin.simplices == pt_idx, axis=1)
    tris = tin.simplices[connected]
    if len(tris) == 0:
        return 1.0

    ratios = []
    for tri in tris:
        ref_area = _triangle_area(ref_pts[tri])
        off_area = _triangle_area(off_pts[tri])
        if off_area == 0 or ref_area == 0:
            ratios.append(np.inf)
            continue
        ratio = ref_area / off_area
        if ratio < 1.0:
            ratio = 1.0 / ratio
        ratios.append(ratio)

    return float(np.mean(ratios))


def _zncc_2d(a, b, mask=None):
    """Zero-mean normalized cross-correlation between two 2-D arrays.

    Returns a scalar in [-1, 1].  Handles masked/zero regions gracefully.
    """
    if mask is not None:
        valid = mask & (a != 0) & (b != 0)
    else:
        valid = (a != 0) & (b != 0)
    
    if not np.any(valid):
        return -1.0
        
    a_v = a[valid].astype(np.float64)
    b_v = b[valid].astype(np.float64)
    
    if len(a_v) < 16:
        return -1.0
        
    a_v -= np.mean(a_v)
    b_v -= np.mean(b_v)
    
    a_sq_sum = np.sum(a_v ** 2)
    b_sq_sum = np.sum(b_v ** 2)
    
    if a_sq_sum < 1e-12 or b_sq_sum < 1e-12:
        return 0.0
        
    return float(np.sum(a_v * b_v) / np.sqrt(a_sq_sum * b_sq_sum))


# ---------------------------------------------------------------------------
# Phase A: TIN-TARR topological filter
# ---------------------------------------------------------------------------

def filter_by_tin_tarr(matched_pairs, boundary_pairs, threshold=1.5):
    """Remove matched pairs that severely distort the local Delaunay mesh.

    The TIN-TARR (Triangle-Area Representation Ratio of TIN) compares the
    area of each Delaunay triangle in the *reference* coordinate space to
    its corresponding triangle in the *offset* coordinate space.  A large
    ratio indicates that the point is causing an anomalous local deformation.

    Parameters
    ----------
    matched_pairs : list of (ref_gx, ref_gy, off_gx, off_gy, quality, name)
        Interior matched pairs (candidates for deletion).  Both ref and
        offset coordinates must be in the same unit system (e.g. UTM metres).
    boundary_pairs : list of (ref_gx, ref_gy, off_gx, off_gy)
        Synthetic edge-anchor pairs (locked, never deleted).  Generated
        from the global affine model so both coordinate sets are in the
        same unit system.
    threshold : float
        Maximum allowable mean TIN-TARR ratio per point.  A value of 1.5
        means the average adjacent triangle area can differ by up to 50%
        between the reference and offset spaces before the point is rejected.

    Returns
    -------
    list
        Surviving matched pairs (boundary_pairs are NOT included).
    """
    if len(matched_pairs) < 4:
        return list(matched_pairs)

    n_internal = len(matched_pairs)

    # Build coordinate arrays: ref_pts and off_pts in the same units
    ref_internal = np.array([(p.ref_x, p.ref_y) for p in matched_pairs], dtype=np.float64)
    off_internal = np.array([(p.off_x, p.off_y) for p in matched_pairs], dtype=np.float64)
    ref_boundary = np.array([(p.ref_x, p.ref_y) for p in boundary_pairs], dtype=np.float64)
    off_boundary = np.array([(p.off_x, p.off_y) for p in boundary_pairs], dtype=np.float64)

    ref_pts = np.concatenate([ref_internal, ref_boundary], axis=0)
    off_pts = np.concatenate([off_internal, off_boundary], axis=0)

    active = np.ones(len(ref_pts), dtype=bool)

    while True:
        active_idx = np.where(active)[0]
        if np.sum(active[:n_internal]) < 4:
            break

        cur_ref = ref_pts[active_idx]
        cur_off = off_pts[active_idx]

        try:
            # QJ options jitters duplicate points to avoid Delaunay failure/omission
            tin = Delaunay(cur_ref, qhull_options="QJ")
        except Exception:
            break

        # Calculate TIN-TARR for each active internal point
        ratios = np.full(len(cur_ref), -1.0)
        for local_i in range(len(cur_ref)):
            global_i = active_idx[local_i]
            # Boundary points are locked
            if global_i >= n_internal:
                continue
            ratios[local_i] = _tin_tarr_for_point(local_i, tin, cur_ref, cur_off)

        worst_local = int(np.argmax(ratios))
        if ratios[worst_local] <= threshold:
            break

        worst_global = active_idx[worst_local]
        active[worst_global] = False

    surviving = [matched_pairs[i] for i in range(n_internal) if active[i]]
    return surviving


# ---------------------------------------------------------------------------
# Phase B: FPP Accuracy Difference optimization
# ---------------------------------------------------------------------------

def _local_piecewise_warp_cv2(ref_pts_local, off_pts_local, off_img, output_shape):
    """Warp *off_img* using piecewise affine with OpenCV remap. (Optimized)"""
    h, w = output_shape
    if len(ref_pts_local) < 3:
        return np.zeros((h, w), dtype=np.float32)

    try:
        tri = Delaunay(ref_pts_local)
    except Exception:
        return np.zeros((h, w), dtype=np.float32)

    # Initialize coordinate maps
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)

    for simplex in tri.simplices:
        dst_tri = ref_pts_local[simplex].astype(np.float32)
        src_tri = off_pts_local[simplex].astype(np.float32)

        # Compute affine transform for this triangle
        M = cv2.getAffineTransform(dst_tri, src_tri)
        
        # Determine bounding box of this triangle within the patch
        tx_min, ty_min = np.floor(dst_tri.min(axis=0)).astype(int)
        tx_max, ty_max = np.ceil(dst_tri.max(axis=0)).astype(int)
        
        # Clamp to patch bounds
        tx_min, ty_min = max(0, tx_min), max(0, ty_min)
        tx_max, ty_max = min(w, tx_max + 1), min(h, ty_max + 1)
        
        if tx_max <= tx_min or ty_max <= ty_min:
            continue
            
        # Create a local mask and grid for just this triangle's bounding box
        tw, th_box = tx_max - tx_min, ty_max - ty_min
        tri_mask = np.zeros((th_box, tw), dtype=np.uint8)
        # Shift triangle vertices to local box coords
        dst_tri_local = dst_tri - [tx_min, ty_min]
        cv2.fillConvexPoly(tri_mask, dst_tri_local.astype(np.int32), 1)
        
        ys_box, xs_box = np.where(tri_mask)
        if len(xs_box) == 0:
            continue
            
        # Coordinates relative to the patch
        xs_patch = xs_box + tx_min
        ys_patch = ys_box + ty_min
        
        pts = np.column_stack([xs_patch, ys_patch]).astype(np.float32)
        src_coords = (M[:, :2] @ pts.T).T + M[:, 2]
        
        map_x[ys_patch, xs_patch] = src_coords[:, 0]
        map_y[ys_patch, xs_patch] = src_coords[:, 1]
        mask[ys_patch, xs_patch] = 1

    # Apply remapping
    warped = cv2.remap(off_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, mask


def _influence_area_triangles(pt_idx, tin):
    """Return simplices that contain *pt_idx* as a vertex."""
    return tin.simplices[np.any(tin.simplices == pt_idx, axis=1)]


def _single_point_ad(pt_idx, ref_pts_px, off_pts_px, tin, hull_vertices_set,
                     ref_img, off_img):
    """Compute Accuracy Difference for one GCP. (Optimized)"""
    
    if pt_idx in hull_vertices_set:
        return -np.inf

    # Find influence area
    ia_tris = _influence_area_triangles(pt_idx, tin)
    if len(ia_tris) == 0:
        return -np.inf

    # Neighborhood expansion
    nbr_pts = np.setdiff1d(np.unique(ia_tris), pt_idx)
    all_ia_tris = [ia_tris]
    for nbr in nbr_pts:
        all_ia_tris.append(_influence_area_triangles(nbr, tin))
    ia_tris = np.unique(np.concatenate(all_ia_tris), axis=0)
    ia_pt_indices = np.unique(ia_tris)

    # Local pixel bounds
    pts_in_ia = ref_pts_px[ia_pt_indices]
    center_px = ref_pts_px[pt_idx]
    
    # Cap the evaluation area to a reasonable radius around the point
    # even if triangles are huge. 256px radius = 512px box.
    eval_radius = 256 
    
    x_min = int(max(pts_in_ia[:, 0].min() - 5, center_px[0] - eval_radius))
    x_max = int(min(pts_in_ia[:, 0].max() + 5, center_px[0] + eval_radius))
    y_min = int(max(pts_in_ia[:, 1].min() - 5, center_px[1] - eval_radius))
    y_max = int(min(pts_in_ia[:, 1].max() + 5, center_px[1] + eval_radius))
    
    # Clamp to image boundaries
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(ref_img.shape[1], x_max), min(ref_img.shape[0], y_max)

    if x_max - x_min < 16 or y_max - y_min < 16:
        return -np.inf

    ref_patch = ref_img[y_min:y_max, x_min:x_max]
    if np.mean(ref_patch > 0) < 0.2:
        return -np.inf

    # Local coordinates
    ref_local = (ref_pts_px[ia_pt_indices] - [x_min, y_min]).astype(np.float32)
    off_local = off_pts_px[ia_pt_indices].astype(np.float32)
    pt_local_idx = np.where(ia_pt_indices == pt_idx)[0][0]

    patch_shape = ref_patch.shape

    # Extract offset patch (generous bounding box)
    off_cols = off_local[:, 0].astype(np.int32)
    off_rows = off_local[:, 1].astype(np.int32)
    ox_min = max(0, off_cols.min() - 20)
    ox_max = min(off_img.shape[1], off_cols.max() + 20)
    oy_min = max(0, off_rows.min() - 20)
    oy_max = min(off_img.shape[0], off_rows.max() + 20)
    if ox_max - ox_min < 16 or oy_max - oy_min < 16:
        return -np.inf

    off_patch = off_img[oy_min:oy_max, ox_min:ox_max]
    if np.mean(off_patch > 0) < 0.2:
        return -np.inf

    # Adjust offset coords to local patch space
    off_local_adj = (off_local - [ox_min, oy_min]).astype(np.float32)

    patch_shape = ref_patch.shape

    # Warp WITH the point
    warped_use, mask_use = _local_piecewise_warp_cv2(ref_local, off_local_adj, off_patch, patch_shape)
    acc_use = _zncc_2d(ref_patch, warped_use, mask=mask_use.astype(bool))

    # Warp WITHOUT the point
    ref_local_del = np.delete(ref_local, pt_local_idx, axis=0)
    off_local_del = np.delete(off_local_adj, pt_local_idx, axis=0)
    
    warped_del, mask_del = _local_piecewise_warp_cv2(ref_local_del, off_local_del, off_patch, patch_shape)
    acc_del = _zncc_2d(ref_patch, warped_del, mask=mask_del.astype(bool))

    return acc_del - acc_use


def optimize_fpps_accuracy(matched_pairs, boundary_pairs, arr_ref, arr_off,
                           ref_transform, off_transform):
    """Iteratively remove matched pairs whose deletion improves local similarity.

    Optimized implementation:
    - Pre-calculates pixel coordinates.
    - Uses OpenCV for fast warping.
    - Only re-evaluates neighbors of deleted points.
    """
    if len(matched_pairs) < 6:
        return list(matched_pairs), 0

    n_internal = len(matched_pairs)
    
    # Combine internal and boundary points
    ref_geo = np.array([(p.ref_x, p.ref_y) for p in matched_pairs] + [(p.ref_x, p.ref_y) for p in boundary_pairs], dtype=np.float64)
    off_geo = np.array([(p.off_x, p.off_y) for p in matched_pairs] + [(p.off_x, p.off_y) for p in boundary_pairs], dtype=np.float64)

    # Pre-convert all geo coordinates to pixel coordinates for the provided arrays
    # arr_ref and arr_off are overlap arrays
    import rasterio.transform
    def _geo_to_px(geo_pts, transform):
        rows, cols = rasterio.transform.rowcol(transform, geo_pts[:, 0], geo_pts[:, 1])
        return np.column_stack([cols, rows]).astype(np.float32)

    ref_pts_px = _geo_to_px(ref_geo, ref_transform)
    off_pts_px = _geo_to_px(off_geo, off_transform)

    # Classify internal points for balanced shoreline/inland retention.
    land = (arr_ref > 0) & (arr_off > 0)
    k = np.ones((3, 3), np.uint8)
    shore = cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0
    shore = cv2.dilate(shore.astype(np.uint8), k, iterations=1) > 0

    point_class = np.array(["other"] * len(ref_geo), dtype=object)
    h, w = arr_ref.shape
    for i in range(n_internal):
        cx = int(round(ref_pts_px[i, 0]))
        cy = int(round(ref_pts_px[i, 1]))
        if 0 <= cx < w and 0 <= cy < h:
            if shore[cy, cx]:
                point_class[i] = "shore"
            elif land[cy, cx]:
                point_class[i] = "inland"

    n_shore_init = int(np.sum(point_class[:n_internal] == "shore"))
    n_inland_init = int(np.sum(point_class[:n_internal] == "inland"))
    min_shore_keep = max(4, int(0.55 * n_shore_init)) if n_shore_init > 0 else 0
    min_inland_keep = max(8, int(0.70 * n_inland_init)) if n_inland_init > 0 else 0
    max_shore_remove = int(0.45 * n_shore_init)
    max_inland_remove = int(0.30 * n_inland_init)

    protected = set(i for i, p in enumerate(matched_pairs)
                    if p.is_anchor)
    shore_removed = 0
    inland_removed = 0

    active = np.ones(len(ref_geo), dtype=bool)
    n_removed = 0
    
    # Initialize AD values
    try:
        tin = Delaunay(ref_pts_px, qhull_options="QJ")
        hull = ConvexHull(ref_pts_px)
        hull_verts = set(hull.vertices)
    except Exception:
        return list(matched_pairs), 0

    ads = np.full(len(ref_geo), -np.inf)
    print(f"    Evaluating initial AD for {n_internal} points...")
    for i in range(n_internal):
        if i % 5 == 0:
            print(f'      Evaluating point {i+1}/{n_internal}...', flush=True)
        if i in protected:
            ads[i] = -np.inf
            continue
        ads[i] = _single_point_ad(i, ref_pts_px, off_pts_px, tin, hull_verts, arr_ref, arr_off)

    # Iterative optimization
    while True:
        # Find the worst removable point while honoring class guardrails.
        sorted_idx = np.argsort(-ads)
        worst_idx = None
        for cand_idx in sorted_idx:
            cand_ad = ads[cand_idx]
            if cand_ad <= 0:
                break
            if cand_idx >= n_internal:
                continue
            if cand_idx in protected or not active[cand_idx]:
                continue

            cls = point_class[cand_idx]
            if cls == "shore":
                cur_shore = int(np.sum(active[:n_internal] & (point_class[:n_internal] == "shore")))
                if cur_shore <= min_shore_keep or shore_removed >= max_shore_remove:
                    continue
            elif cls == "inland":
                cur_inland = int(np.sum(active[:n_internal] & (point_class[:n_internal] == "inland")))
                if cur_inland <= min_inland_keep or inland_removed >= max_inland_remove:
                    continue
            worst_idx = int(cand_idx)
            break

        if worst_idx is None:
            break
        if ads[worst_idx] <= 0:
            break
            
        # Delete point
        active[worst_idx] = False
        ads[worst_idx] = -np.inf
        n_removed += 1
        if point_class[worst_idx] == "shore":
            shore_removed += 1
        elif point_class[worst_idx] == "inland":
            inland_removed += 1
        
        # Get neighbors to update before rebuilding TIN
        indptr, indices = tin.vertex_neighbor_vertices
        neighbors = indices[indptr[worst_idx]:indptr[worst_idx + 1]]
        
        # Rebuild mesh
        active_idx = np.where(active)[0]
        if len(active_idx[active_idx < n_internal]) < 6:
            break
            
        cur_ref_px = ref_pts_px[active_idx]
        try:
            # Note: tin indices change after rebuild, so we must map back to global indices
            tin = Delaunay(cur_ref_px, qhull_options="QJ")
            hull = ConvexHull(cur_ref_px)
            hull_verts_local = set(hull.vertices)
            hull_verts_global = set(active_idx[v] for v in hull_verts_local)
        except Exception:
            break
            
        # Re-evaluate neighbors of deleted point (and their neighbors for safety)
        # In practice, just re-evaluating the immediate neighbors is often enough.
        # Here we re-evaluate all currently active internal points that were neighbors
        # to simplify indexing logic, or just re-eval everything if the set is small.
        update_targets = [n for n in neighbors if n < n_internal and active[n]]
        
        # Since rebuilding TIN is fast but _single_point_ad is slow, 
        # we only update points whose local mesh might have changed.
        for g_idx in update_targets:
            # Map global index to local index in current active set
            l_idx = np.where(active_idx == g_idx)[0][0]
            ads[g_idx] = _single_point_ad(l_idx, cur_ref_px, off_pts_px[active_idx], tin, hull_verts_local, arr_ref, arr_off)

    surviving = [matched_pairs[i] for i in range(n_internal) if active[i]]
    if n_removed > 0:
        print(f"    FPP guardrails: removed {n_removed} total "
              f"(shore={shore_removed}, inland={inland_removed})")
    return surviving, n_removed
