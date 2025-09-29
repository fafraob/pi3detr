import torch
from torch import nn, Tensor
from typing import Tuple
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


class SAModule(torch.nn.Module):
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


class SAModule2(torch.nn.Module):
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
        self.sa1_module = SAModule(0.5, 0.05, MLP([num_features + 3, 32, 32, 64]), 32)
        self.sa2_module = SAModule(0.5, 0.1, MLP([64 + 3, 64, 64, 128]), 32)
        self.sa3_module = SAModule(0.5, 0.2, MLP([128 + 3, 128, 128, 256]), 32)
        self.sa4_module = SAModule(
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
