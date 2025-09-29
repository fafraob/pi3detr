import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from scipy.spatial import cKDTree  # faster than KDTree

from .curve_fitter import (
    fit_cubic_bezier,
    torch_bezier_curve,
    torch_circle_fitter,
    generate_points_on_circle_torch,
    torch_arc_points,
    fit_line,
)


def snap_and_fit_curves(
    data: Data,
) -> Data:
    """
    Snap polylines to nearest point cloud points and fit geometric curves based on predicted classes.

    This function performs two main operations:
    1. Snaps each polyline vertex to its nearest neighbor in the point cloud
    2. Fits the appropriate geometric curve (line, circle, arc, or B-spline) based on the predicted class

    Class mapping:
        0: Background (no processing, kept as-is)
        1: B-spline (cubic Bezier curve fitting)
        2: Line (linear regression fitting)
        3: Circle (3D circle fitting)
        4: Arc (circular arc through 3 points)

    Args:
        data (Data): PyTorch Geometric Data object containing:
            - pos (Tensor): Point cloud coordinates [P, 3]
            - polylines (Tensor): Raw polyline predictions [M, K, 3]
            - polyline_class (Tensor): Predicted classes for each polyline [M]

    Returns:
        Data: Cloned Data object with fitted polylines replacing the original polylines.
              All other attributes remain unchanged.

    Note:
        - Robust error handling: falls back to original polyline if fitting fails
        - Validates output shapes and numerical stability (NaN/Inf checks)
        - Requires minimum points per curve type (e.g., 3 for circles, 4 for B-splines)
    """
    point_cloud = data.pos
    polylines = data.polylines
    polyline_classes = data.polyline_class
    M, K, _ = polylines.shape
    snapped_and_fitted = torch.zeros_like(polylines)

    for i, cls in enumerate(polyline_classes):
        if cls == 0:
            snapped_and_fitted[i] = polylines[i]  # Keep original for class 0
            continue

        try:
            # Snap the polyline to the nearest point in the point cloud
            distances = torch.cdist(polylines[i], point_cloud)
            nearest_idx = distances.argmin(dim=1)
            nn_points = point_cloud[nearest_idx]

            # Safety check: ensure we have valid points
            if (
                len(nn_points) == 0
                or torch.any(torch.isnan(nn_points))
                or torch.any(torch.isinf(nn_points))
            ):
                snapped_and_fitted[i] = polylines[i]
                continue

            new_curve = None

            if cls == 1:  # BSpline
                try:
                    if len(nn_points) < 4:
                        # Not enough points for cubic Bezier, fallback to original
                        new_curve = polylines[i]
                    else:
                        bezier_pts = fit_cubic_bezier(nn_points)
                        new_curve = torch_bezier_curve(
                            bezier_pts.unsqueeze(0), K
                        ).squeeze(0)

                        # Validate output shape and values
                        if (
                            new_curve.shape != (K, 3)
                            or torch.any(torch.isnan(new_curve))
                            or torch.any(torch.isinf(new_curve))
                        ):
                            new_curve = polylines[i]

                except Exception:
                    new_curve = polylines[i]

            elif cls == 2:  # Line
                try:
                    if len(nn_points) < 2:
                        new_curve = polylines[i]
                    else:
                        new_curve = fit_line(nn_points, K)

                        # Validate output shape and values
                        if (
                            new_curve.shape != (K, 3)
                            or torch.any(torch.isnan(new_curve))
                            or torch.any(torch.isinf(new_curve))
                        ):
                            new_curve = polylines[i]
                except Exception:
                    new_curve = polylines[i]

            elif cls == 3:  # Circle
                try:
                    # Check if we have enough unique points for circle fitting
                    unique_points = torch.unique(nn_points, dim=0)
                    if len(unique_points) < 3:
                        new_curve = polylines[i]
                    else:
                        center, normal, radius = torch_circle_fitter(
                            nn_points.unsqueeze(0)
                        )

                        # Validate circle parameters
                        if (
                            torch.any(torch.isnan(center))
                            or torch.any(torch.isnan(normal))
                            or torch.any(torch.isnan(radius))
                            or torch.any(torch.isinf(center))
                            or torch.any(torch.isinf(normal))
                            or torch.any(torch.isinf(radius))
                            or radius <= 0
                        ):
                            new_curve = polylines[i]
                        else:
                            new_curve = generate_points_on_circle_torch(
                                center, normal, radius, K
                            ).squeeze(0)

                            # Validate output shape and values
                            if (
                                new_curve.shape != (K, 3)
                                or torch.any(torch.isnan(new_curve))
                                or torch.any(torch.isinf(new_curve))
                            ):
                                new_curve = polylines[i]
                except Exception:
                    new_curve = polylines[i]

            elif cls == 4:  # Arc
                try:
                    if len(nn_points) < 3:
                        new_curve = polylines[i]
                    else:
                        start_pt = nn_points[0].unsqueeze(0)
                        mid_pt = nn_points[len(nn_points) // 2].unsqueeze(0)
                        end_pt = nn_points[-1].unsqueeze(0)

                        new_curve = torch_arc_points(
                            start_pt, mid_pt, end_pt, K
                        ).squeeze(0)

                        # Validate output shape and values
                        if (
                            new_curve.shape != (K, 3)
                            or torch.any(torch.isnan(new_curve))
                            or torch.any(torch.isinf(new_curve))
                        ):
                            new_curve = polylines[i]
                except Exception:
                    new_curve = polylines[i]

            else:
                # Unknown class, keep original
                new_curve = polylines[i]

            # Final safety check
            if new_curve is not None and new_curve.shape == (K, 3):
                snapped_and_fitted[i] = new_curve
            else:
                snapped_and_fitted[i] = polylines[i]

        except Exception:
            # If anything goes wrong, fallback to original polyline
            snapped_and_fitted[i] = polylines[i]

    output = data.clone()
    output.polylines = snapped_and_fitted
    return output


def filter_predictions(pred_data: Data, thresholds: list[float]) -> Data:
    """
    Filter predictions based on class-specific confidence thresholds.

    Removes polylines whose confidence scores fall below the specified threshold
    for their predicted class. This is typically used as a post-processing step
    to remove low-confidence predictions before further analysis.

    Args:
        pred_data (Data): PyTorch Geometric Data object containing:
            - pos (Tensor): Point cloud coordinates [P, 3]
            - polyline_class (Tensor): Predicted classes [N]
            - polyline_score (Tensor): Confidence scores [N]
            - polylines (Tensor): Polyline coordinates [N, K, 3]
            - query_xyz (Tensor, optional): Query coordinates [N, 3]
        thresholds (list[float]): Confidence thresholds for each class.
                                 Length must match the number of classes.
                                 thresholds[i] is the minimum confidence for class i.

    Returns:
        Data: Filtered Data object containing only polylines that meet their
              class-specific confidence thresholds. Maintains the same structure
              as input but with potentially fewer polylines.

    Example:
        # Keep only polylines with confidence > 0.5 for class 0, > 0.7 for class 1, etc.
        filtered = filter_predictions(data, [0.5, 0.7, 0.6, 0.8])
    """
    mask = (
        pred_data.polyline_score
        >= torch.tensor(thresholds, device=pred_data.pos.device)[
            pred_data.polyline_class
        ]
    )
    filtered_data = Data(
        pos=pred_data.pos,
        polyline_class=pred_data.polyline_class[mask],
        polyline_score=pred_data.polyline_score[mask],
        polylines=pred_data.polylines[mask],
        query_xyz=(
            pred_data.query_xyz[mask] if hasattr(pred_data, "query_xyz") else None
        ),
    )

    return filtered_data


def iou_filter_point_based(
    pred_data,
    iou_threshold: float = 0.6,
    background_class: int = 0,
):
    """
    Efficient per-class Non-Maximum Suppression using IoU computed on point cloud indices.

    This optimized NMS implementation:
    1. Snaps all polyline vertices to nearest point cloud neighbors
    2. Computes IoU based on overlapping point cloud indices (not 3D distances)
    3. Applies greedy NMS within each class, keeping highest-scoring polylines
    4. Uses optimized data structures (cKDTree, sorted arrays) for speed

    Algorithm details:
    - Single batched nearest neighbor query for all valid vertices
    - IoU = |intersection| / |union| of snapped point indices
    - Polylines ordered by: score (desc) → #snapped_points (desc) → index (asc)
    - Background class polylines are never removed
    - Polylines with no valid snapped points are dropped

    Args:
        pred_data (Data): PyTorch Geometric Data object with polyline predictions
        iou_threshold (float, optional): IoU threshold for suppression. Default: 0.6
        background_class (int, optional): Class ID to exclude from NMS. Default: 0

    Returns:
        Data: Filtered Data object with overlapping polylines removed per class.
              Maintains same structure with potentially fewer polylines.

    Performance:
        Significantly faster than distance-based methods due to:
        - Batched spatial queries (cKDTree)
        - Integer set operations (np.intersect1d)
        - Minimal Python loops
    """
    data = pred_data.clone()

    polylines: torch.Tensor = data.polylines  # (N, M, 3)
    classes: torch.Tensor = data.polyline_class  # (N,)
    pc: torch.Tensor = data.pos  # (P, 3)
    scores = getattr(data, "polyline_score", None)

    device = polylines.device
    N = polylines.shape[0]
    if N == 0 or pc.shape[0] == 0:
        return data

    # ---- helpers ----
    def valid_mask(pts_t: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(pts_t).all(dim=-1)
        non_zero = pts_t.abs().sum(dim=-1) > 0
        return finite & non_zero

    # ---- gather all valid vertices once (batched) ----
    # We'll collect (poly_idx, vertex_xyz) over all non-background curves.
    poly_indices_list = []
    all_vertices = []

    bg = int(background_class)
    for i in range(N):
        if int(classes[i].item()) == bg:
            continue
        vm = valid_mask(polylines[i])
        if vm.any():
            pts = polylines[i][vm].detach().cpu().numpy()
            if pts.size > 0:
                all_vertices.append(pts)
                poly_indices_list.append(np.full((pts.shape[0],), i, dtype=np.int32))

    if len(all_vertices) == 0:
        # nothing to snap; everything gets dropped
        keep_mask = torch.zeros(N, dtype=torch.bool, device=device)
        data.polylines = data.polylines[keep_mask]
        data.polyline_class = data.polyline_class[keep_mask]
        if hasattr(data, "polyline_score") and data.polyline_score is not None:
            data.polyline_score = data.polyline_score[keep_mask]
        if hasattr(data, "query_xyz") and data.query_xyz is not None:
            data.query_xyz = data.query_xyz[keep_mask]
        return data

    all_vertices = np.concatenate(all_vertices, axis=0)  # (T, 3)
    owner_poly = np.concatenate(poly_indices_list, axis=0)  # (T,)

    # ---- one cKDTree query for all vertices ----
    pc_np = pc.detach().cpu().numpy()
    tree = cKDTree(pc_np)
    # Use parallel workers if SciPy supports it (falls back silently otherwise)
    nn_idx = tree.query(all_vertices, workers=-1)[1].astype(np.int64)  # (T,)

    # ---- split back to per-curve snapped unique index arrays (sorted) ----
    snapped_arrays = [None] * N
    set_sizes = torch.zeros(N, dtype=torch.long, device=device)

    # group indices by polyline using numpy argsort
    order = np.argsort(owner_poly, kind="mergesort")
    owner_sorted = owner_poly[order]
    nn_sorted = nn_idx[order]

    # find segment starts for each unique polyline id
    unique_ids, starts = np.unique(owner_sorted, return_index=True)
    # append end sentinel
    starts = np.append(starts, owner_sorted.shape[0])

    for k in range(len(unique_ids)):
        i = int(unique_ids[k])
        seg = nn_sorted[starts[k] : starts[k + 1]]
        if seg.size == 0:
            snapped_arrays[i] = np.empty((0,), dtype=np.int64)
            continue
        uniq = np.unique(seg)  # already sorted
        snapped_arrays[i] = uniq
        set_sizes[i] = uniq.size

    # For background curves or curves with no valid vertices, ensure empty arrays
    for i in range(N):
        if snapped_arrays[i] is None:
            snapped_arrays[i] = np.empty((0,), dtype=np.int64)

    # fallback scores: prefer more snapped support
    if scores is None:
        scores = set_sizes.to(torch.float)

    keep_mask = torch.ones(N, dtype=torch.bool, device=device)

    # ---- per-class greedy NMS (IoU via fast array intersection) ----
    target_classes = torch.unique(classes[classes != background_class]).tolist()
    for cls in target_classes:
        cls_inds = torch.where(classes == cls)[0].tolist()
        if not cls_inds:
            continue

        # order by (score desc, size desc, index asc)
        cls_order = sorted(
            cls_inds,
            key=lambda idx: (
                -float(scores[idx].item()),
                -int(set_sizes[idx].item()),
                idx,
            ),
        )

        suppressed = set()
        for i_idx in cls_order:
            if i_idx in suppressed:
                continue

            A = snapped_arrays[i_idx]
            if A.size == 0:
                suppressed.add(i_idx)
                continue

            lenA = A.size
            for j_idx in cls_order:
                if j_idx <= i_idx or j_idx in suppressed:
                    continue
                B = snapped_arrays[j_idx]
                if B.size == 0:
                    suppressed.add(j_idx)
                    continue

                # fast intersection of two sorted int arrays
                inter = np.intersect1d(A, B, assume_unique=True).size
                union = lenA + B.size - inter
                if union == 0:
                    continue
                if (inter / union) > iou_threshold:
                    suppressed.add(j_idx)

        if suppressed:
            keep_mask[list(suppressed)] = False

    # ---- filter aligned fields ----
    data.polylines = data.polylines[keep_mask]
    data.polyline_class = data.polyline_class[keep_mask]
    if hasattr(data, "polyline_score") and data.polyline_score is not None:
        data.polyline_score = data.polyline_score[keep_mask]
    if hasattr(data, "query_xyz") and data.query_xyz is not None:
        data.query_xyz = data.query_xyz[keep_mask]

    return data


def iou_filter_predictions(
    data: Data,
    iou_threshold: float = 0.6,
    tol: float = 1e-2,
) -> Data:
    """
    Remove overlapping polylines within each class using point-to-point distance IoU.

    Performs class-wise Non-Maximum Suppression to eliminate redundant predictions:
    1. Filters out invalid points (NaN, Inf, near-zero)
    2. Computes pairwise point distances between polylines of the same class
    3. Calculates IoU based on points within distance tolerance
    4. Removes lower-scoring polylines when IoU exceeds threshold
    5. Protects "lonely" polylines (minimal overlap) from removal

    IoU Calculation:
        - overlap_i = number of points in polyline_i within tolerance of polyline_j
        - overlap_j = number of points in polyline_j within tolerance of polyline_i
        - intersection = min(overlap_i, overlap_j)
        - union = len(polyline_i) + len(polyline_j) - intersection
        - IoU = intersection / union

    Args:
        data (Data): PyTorch Geometric Data object containing:
            - polylines (Tensor): Polyline coordinates [N, P, 3]
            - polyline_class (Tensor): Class predictions [N]
            - polyline_score (Tensor): Confidence scores [N]
            - query_xyz (Tensor, optional): Query coordinates [N, 3]
        iou_threshold (float, optional): IoU threshold for duplicate removal. Default: 0.6
        tol (float, optional): Distance tolerance for point overlap detection. Default: 1e-2

    Returns:
        Data: Filtered Data object with overlapping polylines removed.
              Background class (0) polylines are never removed.

    Note:
        - Processes polylines in descending score order for deterministic results
        - Requires significant overlap (≥2 points, ≥10% of smaller polyline) before considering removal
        - More computationally expensive than index-based methods but handles arbitrary point clouds
    """
    polylines = data.polylines
    polyline_class = data.polyline_class
    scores = data.polyline_score

    # Precompute valid points for all polylines
    valid_pts = []
    for poly in polylines:
        mask = ~torch.isnan(poly).any(dim=1) & (
            ~torch.isinf(poly).any(dim=1) & (torch.norm(poly, dim=1) > 1e-6)
        )
        valid_pts.append(poly[mask])

    remove_set = set()

    # Process each class independently
    for cls in torch.unique(polyline_class):
        if cls == 0:  # Skip background
            continue
        # Get indices for this class
        class_mask = polyline_class == cls
        class_indices = torch.where(class_mask)[0]
        if len(class_indices) < 2:
            continue

        # Sort by score descending, then index ascending for determinism
        sorted_indices = sorted(
            class_indices.tolist(), key=lambda idx: (-scores[idx].item(), idx)
        )

        # Compare pairs in sorted order
        for i, idx_i in enumerate(sorted_indices):
            if idx_i in remove_set:
                continue
            pts_i = valid_pts[idx_i]
            if len(pts_i) == 0:
                continue
            for j in range(i + 1, len(sorted_indices)):
                idx_j = sorted_indices[j]
                if idx_j in remove_set:
                    continue
                pts_j = valid_pts[idx_j]
                if len(pts_j) == 0:
                    continue

                # Compute point-wise distances
                dists = torch.cdist(pts_i, pts_j)
                # Calculate overlaps
                overlap_i = (dists.min(dim=1).values < tol).sum().item()
                overlap_j = (dists.min(dim=0).values < tol).sum().item()
                min_points = min(len(pts_i), len(pts_j))
                # Skip if not significant overlap
                if (
                    overlap_i < 2
                    or overlap_j < 2
                    or min(overlap_i, overlap_j) < 0.1 * min_points
                ):
                    continue

                # Calculate IoU
                intersection = min(overlap_i, overlap_j)
                union = len(pts_i) + len(pts_j) - intersection
                iou = intersection / union if union > 0 else 0.0
                if iou > iou_threshold:
                    # Always remove lower-scoring polyline
                    remove_set.add(idx_j)

    # Create keep mask (protects lonely lines)
    keep_mask = torch.ones(len(polylines), dtype=torch.bool)
    for idx in remove_set:
        keep_mask[idx] = False

    # Apply filtering
    data.polylines = polylines[keep_mask]
    data.polyline_class = polyline_class[keep_mask]
    data.polyline_score = scores[keep_mask]
    if hasattr(data, "query_xyz"):
        data.query_xyz = data.query_xyz[keep_mask]

    return data
