import torch
import argparse
import numpy as np
import polyscope as ps
from pi3detr import (
    build_dataset_config,
    build_dataset,
    load_args,
)
from pi3detr.utils.curve_fitter import (
    torch_bezier_curve,
    torch_line_points,
    generate_points_on_circle_torch,
    torch_arc_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="path to config"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="dataset to visualize",
    )
    return parser.parse_args()


def visualize_gt_data(data):
    ps.init()
    ps.remove_all_structures()
    points = data.pos.cpu().numpy()

    pc = ps.register_point_cloud("Point Cloud", points, color=(0.5, 0.5, 0.5), radius=0.002, material="wax")
    
    class_names = ["None", "BSpline", "Line", "Circle", "Arc"]
    colors = [
        (0.0, 0.0, 0.0),
        (0.8, 0.2, 0.0),  # BSpline - orange/red
        (0.0, 0.0, 1.0),  # Line - blue
        (0.0, 1.0, 0.0),  # Circle - green
        (1.0, 0.0, 0.0),  # Arc - red
    ]
    
    for idx, curve_params in enumerate(data.y_params.cpu().numpy()):
        curve_type = data.y_cls.cpu().numpy()[idx]
        curve_type_str = {0: "none", 1: "bspline", 2: "line", 3: "circle", 4: "arc"}[
            curve_type
        ]
        if curve_type == 1:  # B-spline
            curve_points = (
                torch_bezier_curve(torch.tensor(curve_params).reshape(1, 4, 3), 50)
                .squeeze(0)
                .numpy()
            )
        elif curve_type == 2:  # Line
            mid_point = curve_params[0:3]
            direction = curve_params[3:6]
            length = curve_params[6]
            curve_points = (
                torch_line_points(
                    torch.tensor(mid_point - direction * length / 2).unsqueeze(0),
                    torch.tensor(mid_point + direction * length / 2).unsqueeze(0),
                    50,
                )
                .squeeze(0)
                .numpy()
            )
        elif curve_type == 3:  # Circle
            curve_points = (
                generate_points_on_circle_torch(
                    torch.tensor(curve_params[0:3]).unsqueeze(0),
                    torch.tensor(curve_params[3:6]).unsqueeze(0),
                    torch.tensor(curve_params[6]).unsqueeze(0),
                    50,
                )
                .squeeze(0)
                .numpy()
            )
        elif curve_type == 4:  # Arc

            curve_points = (
                torch_arc_points(
                    torch.tensor(curve_params[6:9]).unsqueeze(0),
                    torch.tensor(curve_params[0:3]).unsqueeze(0),
                    torch.tensor(curve_params[3:6]).unsqueeze(0),
                    50,
                )
                .squeeze(0)
                .numpy()
            )
        else:
            continue  # Skip unknown curve types

        ps.register_curve_network(
            f"{class_names[curve_type]} {idx}",
            curve_points,
            np.array([[i, i + 1] for i in range(len(curve_points) - 1)]),
            radius=0.004,
            color=colors[curve_type],
            material="wax",
        )
        
    # up direction
    ps.set_up_dir("z_up")
    # ground plane
    ps.set_ground_plane_mode("none")


    ps.show()

def main():
    parsed_args = parse_args()
    args = load_args(parsed_args.config)
    if parsed_args.dataset == "train":
        split = args.data_root
    elif parsed_args.dataset == "val":
        split = args.data_val_root
    else:
        split = args.data_test_root
    dataset_config = build_dataset_config(args, split, augment=False)
    dataset = build_dataset(dataset_config)

    visualize_fn = visualize_gt_data
    
    for data in dataset:
        visualize_fn(data)


if __name__ == "__main__":
    main()
