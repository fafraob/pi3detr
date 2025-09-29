import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from torch_geometric.data.data import Data


class ParametricMatcher(nn.Module):
    def __init__(self, cost_class: int = 1, cost_curve: int = 1) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_curve = cost_curve

    @torch.no_grad()
    def forward(
        self, outputs: dict[str, torch.Tensor], data: Data
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute the matching indices based on class costs and Chamfer distance.
        """
        bs, num_queries = outputs["pred_class"].shape[:2]

        # Compute the classification cost
        out_prob = (
            outputs["pred_class"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        cost_class = -out_prob[:, data.y_cls]

        pred_bspline_params = outputs["pred_bspline_params"].flatten(0, 1)
        pred_line_params = outputs["pred_line_params"].flatten(0, 1)
        pred_line_length = outputs["pred_line_length"].flatten(0, 1)
        pred_circle_params = outputs["pred_circle_params"].flatten(0, 1)
        pred_circle_radius = outputs["pred_circle_radius"].flatten(0, 1)
        pred_arc_params = outputs["pred_arc_params"].flatten(0, 1)

        # classes -> 1: bspline, 2: line, 3: circle, 4: arc
        # NOTE: scaling done assuming points are in [-1, 1] range
        bspline_costs = torch.min(
            torch.cdist(
                pred_bspline_params.flatten(-2, -1),
                data.bspline_params.flatten(-2, -1),
                p=1,
            ),
            torch.cdist(
                pred_bspline_params.flip([1]).flatten(-2, -1),
                data.bspline_params.flatten(-2, -1),
                p=1,
            ),
        )  # [batch_size * num_queries, num_curves]
        line_costs = torch.min(
            torch.cdist(
                pred_line_params.flatten(-2, -1),
                data.line_params.flatten(-2, -1),
                p=1,
            ),
            torch.cdist(
                (
                    pred_line_params
                    * torch.tensor([1.0, -1.0])
                    .view(1, 2, 1)
                    .to(pred_line_params.device)
                ).flatten(-2, -1),
                data.line_params.flatten(-2, -1),
                p=1,
            ),
        ) + torch.cdist(
            pred_line_length,
            data.line_length.unsqueeze(-1),
            p=1,
        )  # [batch_size * num_queries, num_curves]
        circle_costs = torch.min(
            torch.cdist(
                pred_circle_params.flatten(-2, -1),
                data.circle_params.flatten(-2, -1),
                p=1,
            ),
            torch.cdist(
                (
                    pred_circle_params
                    * torch.tensor([1.0, -1.0])
                    .view(1, 2, 1)
                    .to(pred_circle_params.device)
                ).flatten(-2, -1),
                data.circle_params.flatten(-2, -1),
                p=1,
            ),
        ) + torch.cdist(
            pred_circle_radius,
            data.circle_radius.unsqueeze(-1),
            p=1,
        )  # [batch_size * num_queries, num_curves]
        arc_costs = torch.min(
            torch.cdist(
                pred_arc_params.flatten(-2, -1),
                data.arc_params.flatten(-2, -1),
                p=1,
            ),
            # mid, start, end | start and end can be swapped
            torch.cdist(
                pred_arc_params[:, [0, 2, 1], :].flatten(-2, -1),
                data.arc_params.flatten(-2, -1),
            ),
        )

        cost_params = torch.stack(
            [
                torch.zeros_like(line_costs),
                bspline_costs,
                line_costs,
                circle_costs,
                arc_costs,
            ],
            dim=-1,
        )
        cost_params = cost_params[
            torch.arange(cost_params.size(0))[:, None],
            torch.arange(cost_params.size(1)),
            data.y_cls,
        ]  # [num_queries, num_curves]

        # Combine costs
        C = self.cost_class * cost_class + self.cost_curve * cost_params
        C = C.view(bs, num_queries, -1).cpu()

        # Perform Hungarian matching
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(data.num_curves.cpu().tolist(), -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
