from time import time
import torch
import argparse
import numpy as np
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
import polyscope as ps
import polyscope.imgui as psim
from pi3detr import (
    load_args,
    load_weights,
    build_model_config,
    build_model,
    build_dataset,
    DatasetConfig,
)
from pi3detr.evaluation.abc_metrics import ChamferIntervalMetric
import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="path to checkpoint"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="path to config",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="dataset to visualize",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=32768,
        help="Number of points to sample from input point cloud (default: 32768)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--noise",
        "-no",
        type=float,
        choices=[0.0, 1.0e3, 5.0e2, 2.0e2],
        default=0.0,
        help="Amount of noise to add to the input point cloud (default: 0.0)",
    )
    return parser.parse_args()


def subsample_data(data: Data, num_samples: int) -> Data:
    if data.pos.size(0) > num_samples:
        idx = torch.randperm(data.pos.size(0))[:num_samples]
        data.pos = data.pos[idx]
        data.batch = data.batch[idx]
        if hasattr(data, "y_seg"):
            data.y_seg = data.y_seg[idx]

    return data


def evaluate(
    dataloader: DataLoader,
    model: pl.LightningModule,
    num_samples: int,
    verbose: bool,
) -> list[dict]:
    losses = []

    chamfer_metric = ChamferIntervalMetric(interval=0.01, map_cd_thresh=0.005)

    model.eval()
    model_times = []

    if verbose:
        dataloader = tqdm.tqdm(dataloader, desc="Evaluating")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = subsample_data(batch, num_samples)
            batch = batch.to("cuda")
            model_start = time()
            output = model.predict_step(
                batch,
                reverse_norm=False,
                thresholds=None,
                snap_and_fit=True,
                iou_filter=False,
            )[0]
            model_times.append(time() - model_start)

            chamfer_metric.update(output, batch)
            metric = chamfer_metric.compute()

            if verbose:
                print(
                    f"CD: {metric['chamfer_distance']:.4f}, BHD: {metric['bidirectional_hausdorff']:.4f}, Invalid: {chamfer_metric.count - chamfer_metric.valid_count}"
                )
                print(
                    f"mAP: {metric['mAP']:.4f}, mAP Bezier: {metric['mAP_bspline']:.4f}, mAP Line: {metric['mAP_line']:.4f}, mAP Circle: {metric['mAP_circle']:.4f}, mAP Arc: {metric['mAP_arc']:.4f}"
                )
            torch.cuda.empty_cache()

    final_metrics = chamfer_metric.compute()
    print(f"\nFinal Metrics:")
    print(f"Chamfer Distance: {final_metrics['chamfer_distance']:.4f}")
    print(f"Chamfer Distance Std: {final_metrics['chamfer_distance_std']:.4f}")
    print(f"Bidirectional Hausdorff: {final_metrics['bidirectional_hausdorff']:.4f}")
    print(
        f"Bidirectional Hausdorff Std: {final_metrics['bidirectional_hausdorff_std']:.4f}"
    )
    print(f"Valid Count: {chamfer_metric.valid_count} of {chamfer_metric.count}")
    print(f"mAP: {final_metrics['mAP']:.4f}")
    print(f"mAP Bezier: {final_metrics['mAP_bspline']:.4f}")
    print(f"mAP Line: {final_metrics['mAP_line']:.4f}")
    print(f"mAP Circle: {final_metrics['mAP_circle']:.4f}")
    print(f"mAP Arc: {final_metrics['mAP_arc']:.4f}")
    print(f"Model Time: {np.mean(model_times):.4f} ± {np.std(model_times):.4f}")

    return losses


@torch.no_grad()
def main():
    pl.seed_everything(42)
    torch_geometric.seed_everything(42)
    parsed_args = parse_args()
    args = load_args(parsed_args.config)
    if parsed_args.dataset == "train":
        split = args.data_root
    elif parsed_args.dataset == "val":
        split = args.data_val_root
    else:
        split = args.data_test_root

    add_noise = parsed_args.noise > 0
    dataset_config = DatasetConfig(
        dataset=args.dataset,
        root=split,
        augment=True if add_noise else False,
        random_rotate_prob=0.0,
        random_sample_prob=0.0,
        random_sample_bounds=[1, 1],
        noise_prob=1.0 if add_noise else 0.0,
        noise_scale=parsed_args.noise,
    )
    dataset = build_dataset(dataset_config)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_params1 = build_model_config(args)

    model = build_model(model_params1)
    load_weights(model, parsed_args.checkpoint)
    model.eval()
    model.to("cuda")

    evaluate(
        dataloader,
        model,
        parsed_args.samples,
        parsed_args.verbose,
    )


# Example usage
if __name__ == "__main__":
    main()
