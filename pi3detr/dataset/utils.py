import numpy as np
import json
import open3d as o3d


def read_xyz_file(file_path: str, column_idxs: list[int] = [0, 1, 2]) -> np.ndarray:
    """Reads a point cloud from a .xyz file.

    Args:
        file_path (str): Path to the .xyz file.
        column_idxs (list[int], optional): Indices of the columns to read. Defaults to [0,1,2].

    Returns:
        np.ndarray: Point cloud as a numpy array.
    """
    return np.loadtxt(file_path, usecols=column_idxs)


def read_curve_file(file_path: str) -> tuple[np.ndarray]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return np.array(data["linear"]), np.array(data["bezier"])


def read_polyline_file(file_path: str, sep: str = ",") -> np.ndarray:
    polylines = []
    with open(file_path, "r") as f:
        polyline = []
        for line in f:
            if line == "\n":
                polylines.append(polyline)
                polyline = []
            else:
                point = [float(x) for x in line.split(sep)]
                polyline.append(point)
        if polyline:
            polylines.append(polyline)
        return np.array(polylines)


def voxel_down_sample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Downsamples a point cloud using voxel grid downsampling.

    Args:
        xyz (np.ndarray): Point cloud.
        voxel_size (float): Voxel size.

    Returns:
        np.ndarray: Downsampled point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downpcd.points)


def filter_normals(
    xyz: np.ndarray, radius: float, max_nn: float, threshold: float
) -> np.ndarray:
    """Filters normals of a point cloud.

    Args:
        xyz (np.ndarray): Point cloud.
        threshold (float, optional): Threshold for filtering normals.

    Returns:
        np.ndarray: Filtered point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    new_pts = np.asarray(pcd.points)[np.abs(np.asarray(pcd.normals)[:, 2]) < threshold]
    return new_pts
