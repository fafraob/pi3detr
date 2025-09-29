import random
import torch
import numpy as np
from typing import Union
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from dataclasses import dataclass
from typing import Callable
import torch_geometric.transforms as T

from .point_cloud_transforms import (
    random_rotate,
    normalize_and_scale,
    add_noise,
    subsample,
)


@dataclass
class DatasetConfig:
    dataset: str
    root: str
    augment: bool = False
    random_rotate_prob: float = 1
    random_sample_prob: float = 0.5
    random_sample_bounds: tuple[float, float] = (1, 0.5)
    noise_prob: float = 0
    noise_scale: float = 0


class ABCDataset(Dataset):
    def __init__(
        self,
        config: DatasetConfig,
    ) -> None:
        self.file_names = self._read_file_names(config.root)
        self.config = config
        super().__init__(
            config.root,
            None,
            None,
            None,
        )

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def raw_file_names(self) -> Union[str, list[str], tuple]:
        return self.file_names

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple]:
        return [f"{file_name}.pt" for file_name in self.file_names]

    def process(self) -> None:
        print("Should already be processed.")

    def get(self, idx: int) -> BaseData:

        data = torch.load(
            Path(self.processed_dir) / f"{self.raw_file_names[idx]}.pt",
            weights_only=False,
        )
        data["pos"] = data["pos"].to(torch.float32)

        augment = self.config.augment
        if augment and random.random() < self.config.noise_prob:
            sigma = (
                np.max(
                    np.max(data.pos.cpu().numpy(), axis=0)
                    - np.min(data.pos.cpu().numpy(), axis=0)
                )
                / self.config.noise_scale
            )
            noise = torch.tensor(
                np.random.normal(loc=0, scale=sigma, size=data.pos.shape),
                dtype=data.pos.dtype,
                device=data.pos.device,
            )
            data.pos += noise

        if not hasattr(data, "real_scale") or not hasattr(data, "real_center"):
            data.real_center = torch.zeros(3)
            data.real_scale = torch.tensor(1.0)

        if augment and random.random() < self.config.random_sample_prob:
            data = subsample(
                data,
                *self.config.random_sample_bounds,
                max_points=None,
                extra_fields=["y_seg", "y_seg_cls"],
            )

        extra_fields = [
            "y_curve_64",
            "bspline_params",
            "line_params",
            "circle_params",
            "arc_params",
        ]
        if augment and random.random() < self.config.random_rotate_prob:
            data = random_rotate(data, 180, axis=0, extra_fields=extra_fields)
            data = random_rotate(data, 180, axis=1, extra_fields=extra_fields)
            data = random_rotate(data, 180, axis=2, extra_fields=extra_fields)

        line_direction = data.line_params[:, 1]
        circle_normal = data.circle_params[:, 1]

        data = normalize_and_scale(
            data,
            extra_fields=extra_fields,
        )
        # normal vecotrs shouldn't change
        data.line_params[:, 1] = line_direction
        data.circle_params[:, 1] = circle_normal
        # manually adjust length and radius
        data.line_length = data.line_length * data.scale
        data.circle_radius = data.circle_radius * data.scale

        data.y_params = torch.zeros(data.num_curves, 12, dtype=torch.float32)
        for i in range(data.num_curves):
            if data.y_cls[i] == 1:
                # B-spline
                # P0, P1, P2, P3
                data.y_params[i][:12] = data.bspline_params[i].reshape(-1)
            elif data.y_cls[i] == 2:
                # Line
                # midpoint, normal, length
                data.y_params[i][:3] = data.line_params[i][0].reshape(-1)
                data.y_params[i][3:6] = line_direction[i].reshape(-1)
                data.y_params[i][6] = data.line_length[i]  # already adjusted above
            elif data.y_cls[i] == 3:
                # Circle
                # center, normal, radius
                data.y_params[i][:3] = data.circle_params[i][0].reshape(-1)
                data.y_params[i][3:6] = circle_normal[i].reshape(-1)
                data.y_params[i][6] = data.circle_radius[i]  # already adjusted above
            elif data.y_cls[i] == 4:
                # Arc
                # midpoint, start, end
                data.y_params[i][:9] = data.arc_params[i].reshape(-1)
        data.filename = self.raw_file_names[idx]

        return data

    def len(self) -> int:
        return len(self.processed_file_names)

    def _read_file_names(self, root: str) -> list[Path]:
        return sorted(
            [
                fp.stem
                for fp in Path(root).joinpath("processed").glob(f"*.pt")
                if "pre_" not in fp.stem
            ]
        )
