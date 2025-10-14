import torch
from torch import nn, Tensor
from typing import Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.nn import (
    MLP,
    PointNetConv,
    fps,
    knn_interpolate,
    radius,
    global_max_pool,
)
from torch_geometric.data.data import BaseData

TensorTriple = Tuple[Tensor, Tensor, Tensor]


def radius_cpu(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: Optional[int] = None,
    loop: bool = False,
    sort_by_distance: bool = True,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    CPU replacement for torch_cluster.radius / torch_geometric.radius.

    Semantics (matching torch_geometric.radius):
      Returns (row, col) where
        row: indices into `y` (centers) in the range [0, y.size(0))
        col: indices into `x` (neighbors) in the range [0, x.size(0))

    Thus, for y = x[idx]:
      edge_index = torch.stack([col, row], dim=0)
      edge_index[0] indexes the full set (source/neighbor),
      edge_index[1] indexes the sampled centers.
    """
    # Basic checks
    if x.device.type != "cpu" or y.device.type != "cpu":
        raise ValueError("radius_cpu expects x and y to be on CPU.")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D (N, D).")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have same dimensionality D.")

    N_x = x.shape[0]
    N_y = y.shape[0]
    if N_x == 0 or N_y == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    x_np = np.asarray(x)
    y_np = np.asarray(y)

    if batch_x is None:
        batch_x = torch.zeros(N_x, dtype=torch.long)
    else:
        if batch_x.device.type != "cpu":
            batch_x = batch_x.cpu()
        batch_x = batch_x.long()

    if batch_y is None:
        batch_y = torch.zeros(N_y, dtype=torch.long)
    else:
        if batch_y.device.type != "cpu":
            batch_y = batch_y.cpu()
        batch_y = batch_y.long()

    rows = []
    cols = []

    unique_batches = torch.unique(torch.cat([batch_x, batch_y])).tolist()
    # iterate only over batches actually present in y to avoid unnecessary work
    unique_batches = sorted(set(batch_y.tolist()))

    for b in unique_batches:
        # mask and maps from local->global indices
        mask_x = (batch_x == b).numpy()
        mask_y = (batch_y == b).numpy()
        idxs_x = np.nonzero(mask_x)[0]  # global indices in x
        idxs_y = np.nonzero(mask_y)[0]  # global indices in y

        if idxs_y.size == 0 or idxs_x.size == 0:
            continue

        pts_x = x_np[mask_x]
        pts_y = y_np[mask_y]

        # build tree on source points (x) and query for each center in y
        tree = cKDTree(pts_x)
        # neighbors_list: for each center (local), a list of local indices into pts_x
        neighbors_list = tree.query_ball_point(pts_y, r)

        for local_center, neigh_locals in enumerate(neighbors_list):
            if len(neigh_locals) == 0:
                continue
            neigh_locals = np.array(neigh_locals, dtype=int)

            # remove self if requested AND x and y are the same set at same global indices
            if not loop:
                # If x and y refer to the same global indices and same coords, remove self-match
                # we detect self by checking whether global index equals center global index
                center_global = idxs_y[local_center]
                # compute global neighbor indices
                neigh_globals = idxs_x[neigh_locals]
                # boolean mask for neighbors that are not self
                not_self_mask = neigh_globals != center_global
                neigh_locals = neigh_locals[not_self_mask]

                if neigh_locals.size == 0:
                    continue

            # apply max_num_neighbors: keep closest ones by distance if requested
            if max_num_neighbors is not None and neigh_locals.size > max_num_neighbors:
                if sort_by_distance:
                    dists = np.linalg.norm(
                        pts_x[neigh_locals] - pts_y[local_center], axis=1
                    )
                    order = np.argsort(dists)[:max_num_neighbors]
                    neigh_locals = neigh_locals[order]
                else:
                    neigh_locals = np.sort(neigh_locals)[:max_num_neighbors]

            # optionally sort by distance
            if sort_by_distance and neigh_locals.size > 0:
                dists = np.linalg.norm(
                    pts_x[neigh_locals] - pts_y[local_center], axis=1
                )
                order = np.argsort(dists)
                neigh_locals = neigh_locals[order]

            # convert to global indices and append
            neigh_globals = idxs_x[neigh_locals].tolist()
            center_global = int(idxs_y[local_center])
            rows.extend(neigh_globals)  # neighbor indices into x (row)
            cols.extend(
                [center_global] * len(neigh_globals)
            )  # center indices into y (col)

    if len(rows) == 0:
        return torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long)

    row_t = torch.tensor(rows, dtype=torch.long)  # currently neighbors (x)
    col_t = torch.tensor(cols, dtype=torch.long)  # currently centers (y)

    # Swap to enforce (row=center_indices_in_y, col=neighbor_indices_in_x)
    return col_t, row_t


class SAModuleRatio(torch.nn.Module):
    def __init__(
        self, ratio: float, r: float, nn: nn.Module, max_num_neighbors: int = 64
    ):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.max_num_neighbors = max_num_neighbors

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class SAModule(torch.nn.Module):
    def __init__(
        self,
        nn: nn.Module,
        num_out_points: float = 2048,
        r: float = 0.2,
        max_num_neighbors: int = 64,
    ):
        super().__init__()
        self.num_out_points = num_out_points
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)
        self.max_num_neighbors = max_num_neighbors

    def forward(self, data: BaseData) -> list[tuple[TensorTriple]]:
        x, pos, batch = data.x, data.pos, data.batch
        num_points_per_batch = torch.bincount(batch)
        max_ratio = self.num_out_points / num_points_per_batch.min().item()
        fps_idx = fps(pos, batch, ratio=max_ratio)
        fps_batch = batch[fps_idx]
        idx = torch.cat(
            [
                fps_idx[fps_batch == i][: self.num_out_points]
                for i in range(batch.max().item() + 1)
            ]
        )
        if pos.device == torch.device("cpu"):
            row, col = radius_cpu(
                pos,
                pos[idx],
                self.r,
                batch,
                batch[idx],
                max_num_neighbors=self.max_num_neighbors,
                sort_by_distance=False,
            )
        else:  # GPU
            row, col = radius(
                pos,
                pos[idx],
                self.r,
                batch,
                batch[idx],
                max_num_neighbors=self.max_num_neighbors,
            )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return [(x, pos, batch)]


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FPModule(nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        x_skip: torch.Tensor,
        pos_skip: torch.Tensor,
        batch_skip: torch.Tensor,
    ):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetPPEncoder(nn.Module):
    def __init__(self, num_features: int = 3, out_channels: int = 512):
        super().__init__()
        self.out_channels = out_channels
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModuleRatio(
            0.5, 0.05, MLP([num_features + 3, 32, 32, 64]), 32
        )
        self.sa2_module = SAModuleRatio(0.5, 0.1, MLP([64 + 3, 64, 64, 128]), 32)
        self.sa3_module = SAModuleRatio(0.5, 0.2, MLP([128 + 3, 128, 128, 256]), 32)
        self.sa4_module = SAModuleRatio(
            0.5, 0.4, MLP([256 + 3, 256, 256, self.out_channels]), 32
        )

    def forward(self, data: BaseData) -> list[Tensor]:
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)
        return [sa0_out, sa1_out, sa2_out, sa3_out, sa4_out]


class PointNetPPDecoder(nn.Module):
    def __init__(self, num_features: int = 3, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.fp4_module = FPModule(1, MLP([512 + 256, 256, 256]))
        self.fp3_module = FPModule(3, MLP([256 + 128, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 64, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + num_features, 128, self.out_channels]))

    def forward(
        self,
        sa0_out: TensorTriple,
        sa1_out: TensorTriple,
        sa2_out: TensorTriple,
        sa3_out: TensorTriple,
        sa4_out: TensorTriple,
    ) -> TensorTriple:
        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, pos, batch = self.fp1_module(*fp2_out, *sa0_out)
        return [x, pos, batch]


class PointNetPP(nn.Module):
    def __init__(self, num_features: int, dec_out_channels: int = 256):
        super().__init__()
        self.encoder = PointNetPPEncoder(num_features)
        self.decoder = PointNetPPDecoder(num_features, dec_out_channels)
        self.out_channels = self.decoder.out_channels

    def forward(self, data: BaseData) -> TensorTriple:
        x = self.encoder(data)
        x, pos, batch = self.decoder(*x)
        return [(x, pos, batch)]
