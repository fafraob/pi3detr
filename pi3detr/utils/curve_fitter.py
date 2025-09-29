import numpy as np
import torch
import math


def torch_arc_points(start, mid, end, num_points=100):
    """
    Sample points along a circular arc defined by 3 points in 3D, batched.
    Inputs:
        start, mid, end: tensors of shape [B, 3]
        num_points: number of points sampled along the arc
    Returns:
        arc_points: tensor of shape [B, num_points, 3]
    """
    B = start.shape[0]

    # 1. Compute circle center and normal vector for each batch
    v1 = mid - start  # [B,3]
    v2 = end - start  # [B,3]

    normal = torch.cross(v1, v2, dim=1)  # [B,3]
    normal_norm = normal.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normal = normal / normal_norm  # normalize

    mid1 = (start + mid) / 2  # [B,3]
    mid2 = (start + end) / 2  # [B,3]

    # perpendicular directions in the plane
    perp1 = torch.cross(normal, v1, dim=1)  # [B,3]
    perp2 = torch.cross(normal, v2, dim=1)  # [B,3]

    # Solve line intersection for each batch:
    # Line 1: point mid1, direction perp1
    # Line 2: point mid2, direction perp2
    # Solve for t in mid1 + t * perp1 = mid2 + s * perp2

    # Construct matrix A and vector b for least squares
    A = torch.stack([perp1, -perp2], dim=2)  # [B,3,2]
    b = (mid2 - mid1).unsqueeze(2)  # [B,3,1]

    # Use torch.linalg.lstsq if available, fallback to pinv:
    try:
        t_s = torch.linalg.lstsq(A, b).solution  # [B,2,1]
    except:
        # fallback
        At = A.transpose(1, 2)  # [B,2,3]
        pinv = torch.linalg.pinv(A)  # [B,2,3]
        t_s = torch.bmm(pinv, b)  # [B,2,1]

    t = t_s[:, 0, 0]  # [B]

    center = mid1 + (perp1 * t.unsqueeze(1))  # [B,3]

    radius = (start - center).norm(dim=1, keepdim=True)  # [B,1]

    # 2. Define local basis in the arc plane
    x_axis = (start - center) / radius  # [B,3]
    y_axis = torch.cross(normal, x_axis, dim=1)  # [B,3]

    # 3. Compute angles function
    def angle_from_vector(v):
        x = (v * x_axis).sum(dim=1)  # [B]
        y = (v * y_axis).sum(dim=1)  # [B]
        angles = torch.atan2(y, x)  # [-pi, pi]
        angles = angles % (2 * math.pi)
        return angles

    theta_start = torch.zeros(B, device=start.device)  # [B], 0 since x_axis is ref
    theta_end = angle_from_vector(end - center)  # [B]
    theta_mid = angle_from_vector(mid - center)  # [B]

    # 4. Ensure arc goes the correct way (shortest arc through mid)
    # Helper function vectorized:
    def between(a, b, c):
        # returns bool tensor if b is between a and c going CCW mod 2pi
        return ((a < b) & (b < c)) | ((c < a) & ((a < b) | (b < c)))

    cond = between(theta_start, theta_mid, theta_end)

    # If not cond, swap start/end angles by adding 2pi to one side
    # We'll add 2pi to whichever angle is smaller to preserve direction
    theta_start_new = torch.where(
        cond,
        theta_start,
        torch.where(theta_start < theta_end, theta_start, theta_start + 2 * math.pi),
    )
    theta_end_new = torch.where(
        cond,
        theta_end,
        torch.where(theta_end < theta_start, theta_end + 2 * math.pi, theta_end),
    )

    # 5. Sample angles
    t_lin = (
        torch.linspace(0, 1, steps=num_points, device=start.device)
        .unsqueeze(0)
        .repeat(B, 1)
    )  # [B, num_points]

    angles = theta_start_new.unsqueeze(1) + t_lin * (
        theta_end_new - theta_start_new
    ).unsqueeze(
        1
    )  # [B, num_points]
    angles = angles % (2 * math.pi)

    # 6. Map back to 3D
    cos_a = torch.cos(angles).unsqueeze(2)  # [B, num_points, 1]
    sin_a = torch.sin(angles).unsqueeze(2)  # [B, num_points, 1]

    points = center.unsqueeze(1) + radius.unsqueeze(1) * (
        cos_a * x_axis.unsqueeze(1) + sin_a * y_axis.unsqueeze(1)
    )  # [B, num_points, 3]

    return points


def torch_circle_fitter(
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fits a circle to an arbitrary number of 3D points using least squares.

    Args:
        points: Tensor of shape (B, N, 3), where B = batch size, N = number of points per batch.

    Returns:
        center_3d: (B, 3) tensor of circle centers in 3D
        normal: (B, 3) tensor of normal vectors to the circle's plane
        radius: (B,) tensor of circle radii
    """
    B, N, _ = points.shape
    mean = points.mean(dim=1, keepdim=True)
    centered = points - mean

    # PCA via SVD
    U, S, Vh = torch.linalg.svd(centered)
    normal = Vh[
        :, -1, :
    ]  # last singular vector corresponds to the smallest variance (plane normal)

    # Project to plane
    x_axis = Vh[:, 0, :]
    y_axis = Vh[:, 1, :]
    X = torch.einsum("bij,bj->bi", centered, x_axis)  # (B, N)
    Y = torch.einsum("bij,bj->bi", centered, y_axis)  # (B, N)

    # Fit circle in 2D: (x - xc)^2 + (y - yc)^2 = r^2
    A = torch.stack([2 * X, 2 * Y, torch.ones_like(X)], dim=-1)  # (B, N, 3)
    b = (X**2 + Y**2).unsqueeze(-1)  # (B, N, 1)

    # Solve the least squares system: A @ [xc, yc, c] = b
    AtA = A.transpose(1, 2) @ A
    Atb = A.transpose(1, 2) @ b
    sol = torch.linalg.solve(AtA, Atb).squeeze(-1)  # (B, 3)

    xc, yc, c = sol[:, 0], sol[:, 1], sol[:, 2]
    radius = torch.sqrt(xc**2 + yc**2 + c)

    # Reconstruct center in 3D
    center_3d = mean.squeeze(1) + xc.unsqueeze(1) * x_axis + yc.unsqueeze(1) * y_axis

    return center_3d, normal, radius


def generate_points_on_circle(center, normal, radius, num_points=100):
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    # Find two orthogonal vectors in the plane of the circle
    if np.allclose(normal, [0, 0, 1]):
        u = np.array([1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
        u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Generate points on the circle in the plane
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_points = (
        center
        + radius * np.outer(np.cos(theta), u)
        + radius * np.outer(np.sin(theta), v)
    )

    return circle_points


def generate_points_on_circle_torch(
    center, normal, radius, num_points=100
) -> torch.Tensor:
    """
    Generate points on a circle in 3D space using PyTorch, supporting batching.

    Args:
        center: Tensor of shape (B, 3), circle centers.
        normal: Tensor of shape (B, 3), normal vectors to the circle's plane.
        radius: Tensor of shape (B,), radii of the circles.
        num_points: Number of points to generate per circle.

    Returns:
        Tensor of shape (B, num_points, 3), points on the circles.
    """
    B = center.shape[0]
    normal = normal / torch.norm(normal, dim=1, keepdim=True)  # Normalize normals

    # Find two orthogonal vectors in the plane of the circle
    u = torch.linalg.cross(
        normal,
        torch.tensor([0, 0, 1], dtype=normal.dtype, device=normal.device).expand_as(
            normal
        ),
    )
    u = torch.where(
        torch.norm(u, dim=1, keepdim=True) > 1e-6,
        u,
        torch.tensor([1, 0, 0], dtype=normal.dtype, device=normal.device).expand_as(
            normal
        ),
    )
    u = u / torch.norm(u, dim=1, keepdim=True)
    v = torch.linalg.cross(normal, u)

    # Generate points on the circle in the plane
    theta = (
        torch.linspace(0, 2 * torch.pi, num_points, device=center.device)
        .unsqueeze(0)
        .repeat(B, 1)
    )
    circle_points = (
        center.unsqueeze(1)
        + radius.unsqueeze(1).unsqueeze(2)
        * torch.cos(theta).unsqueeze(2)
        * u.unsqueeze(1)
        + radius.unsqueeze(1).unsqueeze(2)
        * torch.sin(theta).unsqueeze(2)
        * v.unsqueeze(1)
    )

    return circle_points


def torch_bezier_curve(
    control_points: torch.Tensor, num_points: int = 100
) -> torch.Tensor:
    control_points = control_points.float()
    t = (torch.linspace(0, 1, num_points).unsqueeze(-1).unsqueeze(-1)).to(
        control_points.device
    )  # shape [1, num_points, 1]
    B = (
        (1 - t) ** 3 * control_points[:, 0]
        + 3 * (1 - t) ** 2 * t * control_points[:, 1]
        + 3 * (1 - t) * t**2 * control_points[:, 2]
        + t**3 * control_points[:, 3]
    )
    # Transpose the first two dimensions to get the shape (batch_size, num_points, 3)
    B = B.transpose(0, 1)

    return B


def torch_line_points(
    start_points: torch.Tensor, end_points: torch.Tensor, num_points: int = 100
) -> torch.Tensor:
    weights = (
        torch.linspace(0, 1, num_points)
        .unsqueeze(0)
        .unsqueeze(-1)
        .to(start_points.device)
    )
    line_points = (1 - weights) * start_points.unsqueeze(
        1
    ) + weights * end_points.unsqueeze(1)
    return line_points


def fit_line(points: torch.Tensor, K: int = 100) -> torch.Tensor:
    """
    Fit a line to 3D points and sample K points along it.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "Input must be [N, 3]"

    # Step 1: Center the points
    mean = points.mean(dim=0, keepdim=True)
    centered = points - mean

    # Step 2: SVD
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    direction = Vh[0]  # First principal component

    # Step 3: Project points onto the line to get min/max
    projections = torch.matmul(centered, direction)
    t_min, t_max = projections.min(), projections.max()

    # Step 4: Sample along the line
    t_vals = torch.linspace(t_min, t_max, K).to(points.device)
    fitted_points = mean + t_vals[:, None] * direction

    return fitted_points


def fit_cubic_bezier(points_3d: torch.Tensor) -> torch.Tensor:
    """
    Fit a cubic Bézier curve to 3D points while fixing the start and end points.

    Args:
        points_3d: (N, 3) Tensor of 3D arc points.

    Returns:
        bezier_pts: Tensor of 4 control points (P0, P1, P2, P3), shape (4, 3)
    """
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)

    n = len(points_3d)

    if n < 4:
        raise ValueError("At least 4 points are required to fit a cubic Bézier curve.")

    device = points_3d.device

    # Fixed start and end points
    P0 = points_3d[0]
    P3 = points_3d[-1]

    # Normalize parameter t
    t = torch.linspace(0, 1, n, device=device)

    # Bernstein basis functions for cubic Bézier
    def bernstein(t):
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t**2
        b3 = t**3
        return torch.stack([b0, b1, b2, b3], dim=1)  # (n, 4)

    B = bernstein(t)

    # Initial guess for P1 and P2 (based on tangents)
    P1_init = P0 + (points_3d[1] - P0) * 1.5
    P2_init = P3 + (points_3d[-2] - P3) * 1.5

    # Optimization parameters - make them require gradients
    P1 = P1_init.clone().detach().requires_grad_(True)
    P2 = P2_init.clone().detach().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.LBFGS([P1, P2], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()

        # Compute Bézier curve
        curve = (
            B[:, 0].unsqueeze(1) * P0
            + B[:, 1].unsqueeze(1) * P1
            + B[:, 2].unsqueeze(1) * P2
            + B[:, 3].unsqueeze(1) * P3
        )

        # Compute loss (mean squared error)
        loss = torch.mean((curve - points_3d) ** 2)
        loss.backward()
        return loss

    # Optimize
    optimizer.step(closure)

    # Return control points
    with torch.no_grad():
        return torch.stack([P0, P1, P2, P3])
