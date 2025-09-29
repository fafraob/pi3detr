import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from torch_geometric.nn import MLP, fps, knn
from torch_geometric.data.data import Data
from typing import Optional


class QueryEngine(nn.Module, ABC):
    def __init__(
        self,
        pos_embedder: Optional[nn.Module],
        feat_dim: int,
        max_points_in_param: int,
        num_queries: int,
    ):
        super().__init__()
        self.pos_embedder = pos_embedder
        self.feat_dim = feat_dim
        self.max_points_in_param = max_points_in_param
        self.num_queries = num_queries

    @abstractmethod
    def forward(self, data: Data) -> tuple[Tensor]:
        pass


class PointFPSQueryEngine(QueryEngine):
    def __init__(
        self,
        pos_embedder: nn.Module,
        feat_dim: int,
        max_points_in_param: int,
        num_queries: int,
    ):
        super().__init__(pos_embedder, feat_dim, max_points_in_param, num_queries)
        self.num_queries = num_queries
        self.query_proj = MLP(
            [self.feat_dim, self.feat_dim, self.feat_dim],
            bias=False,
            act="relu",
            norm="layer_norm",
        )

    def forward(self, data: Data) -> tuple[Tensor]:
        num_points_per_batch = torch.bincount(data.batch)
        max_ratio = self.num_queries / num_points_per_batch.min().item()
        fps_idx = fps(data.pos, data.batch, ratio=max_ratio)
        fps_batch = data.batch[fps_idx]
        query_xyz = torch.stack(
            [
                data.pos[fps_idx[fps_batch == i][: self.num_queries]]
                for i in range(data.batch.max().item() + 1)
            ]
        )
        query_pos = self.pos_embedder(query_xyz, num_channels=self.feat_dim)
        query_embed = self.query_proj(query_pos.permute(0, 2, 1))[
            :, : self.num_queries, :
        ].permute(0, 2, 1)
        return (
            query_xyz.unsqueeze(2).expand(-1, -1, self.max_points_in_param, -1),
            query_embed,
        )


class LearnedQueryEngine(QueryEngine):

    def __init__(
        self,
        pos_embedder: Optional[nn.Module],
        feat_dim: int,
        max_points_in_param: int,
        num_queries: int,
    ):
        super().__init__(None, feat_dim, max_points_in_param, num_queries)
        self.query_embed = nn.Embedding(self.num_queries, feat_dim)

    def forward(self, data: Data) -> tuple[Tensor]:
        return (
            torch.zeros(
                data.batch_size,
                self.num_queries,
                self.max_points_in_param,
                3,
                device=data.pos.device,
                requires_grad=False,
            ),
            self.query_embed.weight.unsqueeze(0)
            .expand(data.batch_size, -1, -1)
            .permute(0, 2, 1),
        )


def build_query_engine(
    query_type: str,
    pos_embedder: Optional[nn.Module],
    feat_dim: int,
    max_points_in_param: int,
    num_queries: int,
) -> QueryEngine:
    if query_type == "point_fps":
        return PointFPSQueryEngine(
            pos_embedder, feat_dim, max_points_in_param, num_queries
        )
    elif query_type == "learned":
        return LearnedQueryEngine(None, feat_dim, max_points_in_param, num_queries)
    else:
        raise ValueError(f"Unknown query type {query_type}")
