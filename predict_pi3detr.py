"""
Prediction script for PI3DETR model.

This script loads a trained PI3DETR model and performs inference on 3D point cloud data,
visualizing the detected curves and polylines.
"""

import os
import argparse
import numpy as np
import polyscope as ps
import pytorch_lightning as pl
import torch
import torch_geometric
import trimesh
from torch_geometric.data.data import Data
import fpsample

from pi3detr import (
    build_model,
    build_model_config,
    load_args,
    load_weights,
)


from pi3detr.dataset import normalize_and_scale


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict curves on point cloud data using PI3DETR"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        help="Path to input file (.ply, .obj, .pt, .xyz) or input folder.",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=32768,
        help="Number of points to sample from input point cloud (default: 32768)",
    )
    parser.add_argument(
        "--sample_mode",
        "-s",
        type=str,
        default="all",
        choices=["fps", "random", "uniform", "all"],
        help="Sampling method for point cloud subsampling",
    )
    parser.add_argument(
        "--reduction",
        "-r",
        type=int,
        default=0,
        help="Reduce points by random sampling before sample_mode sampling (default: 0, no reduction). Usable if you are using fps sampling to reduce runtime.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on (default: cuda)",
    )

    return parser.parse_args()


def file_loader(file_path: str) -> Data:
    """
    Load point cloud data from various file formats.

    Args:
        file_path: Path to the input file

    Returns:
        Data object containing point positions

    Raises:
        ValueError: If file format is not supported
    """
    if file_path.endswith(".pt"):
        pos = torch.load(file_path, weights_only=False)["pos"]
        return Data(pos=pos)

    elif file_path.endswith(".ply"):
        mesh = trimesh.load(file_path)
        if isinstance(mesh, trimesh.PointCloud):
            pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        else:
            # Sample points from mesh surface
            pos = torch.tensor(mesh.sample(100000), dtype=torch.float32)
        return Data(pos=pos)

    elif file_path.endswith(".obj"):
        mesh = trimesh.load(file_path, force="mesh")
        if isinstance(mesh, trimesh.PointCloud):
            pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        else:
            # Sample points from mesh surface
            pos = torch.tensor(mesh.sample(100000), dtype=torch.float32)
        return Data(pos=pos)

    elif file_path.endswith(".xyz"):
        points = np.loadtxt(file_path, dtype=np.float32)
        pos = torch.tensor(points[:, :3], dtype=torch.float32)
        return Data(pos=pos)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def process_data(
    data: Data, sample: int = 32768, sample_mode: str = "fps", reduction: int = 0
) -> Data:
    """
    Process and subsample point cloud data.

    Args:
        data: Input point cloud data
        sample: Number of points to sample
        sample_mode: Sampling method ("fps", "random", "uniform", "all")
        reduction: number of points to reduce before sampling

    Returns:
        Processed and normalized data
    """
    if not hasattr(data, "pos"):
        return data

    # Apply reduction if specified
    if reduction > 0 and data.pos.size(0) > reduction:
        indices = torch.randperm(data.pos.size(0))[:reduction]
        data.pos = data.pos[indices]

    # Apply sampling strategy
    if sample_mode == "random":
        indices = torch.randperm(data.pos.size(0))[:sample]
        data.pos = data.pos[indices]

    elif sample_mode == "fps":
        if data.pos.size(0) > sample:
            idx = fpsample.bucket_fps_kdline_sampling(data.pos, sample, h=6)
            data.pos = data.pos[idx]

    elif sample_mode == "uniform":
        step = max(1, data.pos.size(0) // sample)
        data.pos = data.pos[::step][:sample]

    elif sample_mode == "all":
        pass  # Keep all points

    # Add batch information for single point cloud BEFORE normalization
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.pos.size(0), dtype=torch.long)
        data.batch_size = 1

    # Normalize and scale the points
    data = normalize_and_scale(data)

    # Ensure scale and center are proper batch tensors
    if hasattr(data, "scale") and data.scale.dim() == 0:
        data.scale = data.scale.unsqueeze(0)
    if hasattr(data, "center") and data.center.dim() == 1:
        data.center = data.center.unsqueeze(0)

    return data


def visualize(data: Data) -> None:
    """
    Visualize detected curves and input point cloud using Polyscope.

    Args:
        data: Data containing polylines and point cloud
    """
    ps.init()
    ps.remove_all_structures()
    ps.remove_all_groups()
    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    input_pts = data.pos.cpu().numpy()

    # Create visualization groups
    curves = ps.create_group("Curves")
    class_names = ["None", "BSpline", "Line", "Circle", "Arc"]

    colors = [
        (0.0, 0.0, 0.0),
        (0.8, 0.2, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ]

    polylines = data.polylines.cpu().numpy()

    # Visualize detected polylines
    for i, polyline in enumerate(polylines):
        cls = data.polyline_class[i].item()
        score = data.polyline_score[i].item()
        cls_name = class_names[cls]

        # Skip low-confidence or "None" class predictions
        if cls == 0:
            continue

        # Create curve visualization
        name = f"{cls_name} {i} {score:.5f}"
        edges = np.array([[j, j + 1] for j in range(len(polyline) - 1)])
        curve = ps.register_curve_network(
            name,
            polyline,
            edges,
            radius=0.004,
            color=colors[cls],
            material="wax",
        )
        curves.add_child_structure(curve)

    # Visualize input point cloud
    ps.register_point_cloud(
        "Input Points", input_pts, radius=0.002, color=(0.5, 0.5, 0.5), material="wax"
    )

    ps.reset_camera_to_home_view()
    ps.show()


def predict(model: pl.LightningModule, data: Data, device: str) -> Data:
    """Run prediction on the input data."""
    # Run inference
    data = data.to(device)
    output = model.predict_step(
        data,
        reverse_norm=True,
        thresholds=None,
        snap_and_fit=True,
        iou_filter=False,
    )

    return output[0]


@torch.no_grad()
def main():
    """Main prediction pipeline."""
    # Set seeds for reproducibility
    pl.seed_everything(42)
    torch_geometric.seed_everything(42)

    # Parse arguments
    parsed_args = parse_args()
    args = load_args(parsed_args.config)

    if not parsed_args.path:
        raise ValueError("Please provide a file path using --file argument.")

    # Build and load model
    model_config = build_model_config(args)
    model = build_model(model_config)
    load_weights(model, parsed_args.checkpoint)
    model.eval()
    device = parsed_args.device
    model = model.to(device)

    # check if input is a file or directory
    if os.path.isdir(parsed_args.path):
        # Load all files in the directory

        for filename in sorted(os.listdir(parsed_args.path)):
            file_data = file_loader(os.path.join(parsed_args.path, filename))
            processed_data = process_data(
                file_data,
                sample=parsed_args.samples,
                sample_mode=parsed_args.sample_mode,
                reduction=parsed_args.reduction,
            )
            output_data = predict(model, processed_data, device)
            visualize(output_data)
    else:
        # Load single file
        file_data = file_loader(parsed_args.path)
        processed_data = process_data(
            file_data,
            sample=parsed_args.samples,
            sample_mode=parsed_args.sample_mode,
            reduction=parsed_args.reduction,
        )
        output_data = predict(model, processed_data, device)
        visualize(output_data)


if __name__ == "__main__":
    main()
