from abc import abstractmethod
import torch
from torch import nn
from dataclasses import dataclass, field
import torch.nn.functional as F
from torch_geometric.data.data import Data
from kornia.losses import focal_loss
from .matcher import *


def chamfer_distance_batch(pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    assert len(pts1.shape) == 3 and len(pts2.shape) == 3
    if pts1.nelement() == 0 or pts2.nelement() == 0:
        return torch.tensor(0.0, device=pts1.device, requires_grad=True)
    dist_matrix = torch.cdist(
        pts1, pts2, p=2
    )  # shape: (batch_size, num_points, num_points)
    dist1 = dist_matrix.min(dim=2).values.mean(dim=1)  # min over pts2, mean over pts1
    dist2 = dist_matrix.min(dim=1).values.mean(dim=1)  # min over pts1, mean over pts2
    return (dist1 + dist2) / 2


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@torch.no_grad()
def f1_score(output, target, threshold=0.5):
    output = output > threshold
    target = target > threshold
    tp = (output & target).sum()
    tn = (~output & ~target).sum()
    fp = (output & ~target).sum()
    fn = (~output & target).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * (precision * recall) / (precision + recall + 1e-8), precision, recall


@dataclass
class LossParams:
    num_classes: int
    cost_class: int = 1
    cost_curve: int = 1
    class_loss_type: str = "cross_entropy"  # or "focal"
    class_loss_weights: list[float] = field(
        default_factory=lambda: [
            0.04834912,
            0.40329467,
            0.09588135,
            0.23071379,
            0.22176106,
        ]
    )
    # NOTE: Weights calculated based on the dataset
    # bezier, line, circle, arc, empty
    # counts = np.array([11347, 200751, 34672, 37528])
    # counts = np.append(counts, total_pred - counts.sum())
    # weights = 1 / np.sqrt(counts)
    # weights = weights / weights.sum()


class Loss(nn.Module):

    def __init__(self, params: LossParams) -> None:
        super().__init__()
        self.matcher = ParametricMatcher(params.cost_class, params.cost_curve)
        self.num_classes = params.num_classes
        self.class_loss_type = params.class_loss_type
        class_weights = torch.tensor(
            params.class_loss_weights,
        )

        self.register_buffer("class_weights", class_weights)

    def forward(
        self, outputs: dict[str, torch.Tensor], data: Data
    ) -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, data)
        losses = {}
        losses.update(self._loss_class(outputs, data, indices))
        losses.update(self._loss_polyline(outputs, data, indices))
        # In case of auxiliary losses, we repeat this process with the output
        # of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, data)
                l_dict = self._loss_class(aux_outputs, data, indices, False)
                l_dict.update(self._loss_polyline(aux_outputs, data, indices))
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                losses.update(l_dict)
        return losses

    @abstractmethod
    def _loss_polyline(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Compute the polyline loss."""
        pass

    def _loss_class(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        log: bool = True,
    ) -> torch.Tensor:
        num_targets = (
            data.num_polylines.tolist()
            if hasattr(data, "num_polylines")
            else data.num_curves.tolist()
        )
        src_logits = outputs["pred_class"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [
                target[J]
                for target, (_, J) in zip(
                    data.y_cls.split_with_sizes(num_targets), indices
                )
            ]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )  # 0: empty class
        target_classes[idx] = target_classes_o
        losses = {}

        if self.class_loss_type == "cross_entropy":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2),
                target_classes,
                weight=self.class_weights.to(src_logits.device),
                reduction="mean",
            )
        else:
            loss_class = focal_loss(
                src_logits.transpose(1, 2),
                target_classes,
                alpha=0.25,
                gamma=2.0,
                weight=self.class_weights.to(src_logits.device),
                reduction="mean",
            )
        losses["loss_class"] = loss_class
        if log:
            losses["class_error"] = (
                100
                - accuracy(
                    src_logits.reshape(-1, src_logits.size(-1)),
                    target_classes.flatten(),
                )[0]
            )
            f1, _, _ = f1_score(
                src_logits.reshape(-1, src_logits.size(-1)).softmax(-1).argmax(-1),
                target_classes.flatten(),
                threshold=0.5,
            )
            losses["class_f1_score"] = f1

        return losses

    def _get_src_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(
        self, indices: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


class ParametricLoss(Loss):
    def __init__(self, params: LossParams) -> None:
        super().__init__(params)

    def _loss_polyline(
        self,
        outputs: dict[str, torch.Tensor],
        data: Data,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_bspline_params = outputs["pred_bspline_params"][idx]
        src_bspline_points = outputs["pred_bspline_points"][idx]
        src_line_params = outputs["pred_line_params"][idx]
        src_line_length = outputs["pred_line_length"][idx]
        src_line_points = outputs["pred_line_points"][idx]
        src_circle_params = outputs["pred_circle_params"][idx]
        src_circle_radius = outputs["pred_circle_radius"][idx]
        src_circle_points = outputs["pred_circle_points"][idx]
        src_arc_params = outputs["pred_arc_params"][idx]
        src_arc_points = outputs["pred_arc_points"][idx]
        target_params = torch.cat(
            [
                target[J]
                for target, (_, J) in zip(
                    data.y_params.split_with_sizes(data.num_curves.tolist()), indices
                )
            ]
        )
        target_classes = torch.cat(
            [
                target[J]
                for target, (_, J) in zip(
                    data.y_cls.split_with_sizes(data.num_curves.tolist()), indices
                )
            ]
        )
        target_curves = torch.cat(
            [
                target[J]
                for target, (_, J) in zip(
                    data.y_curve_64.split_with_sizes(data.num_curves.tolist()), indices
                )
            ]
        )

        losses = {}

        # Filter indices for each class
        bspline_mask = target_classes == 1  # B-spline
        line_mask = target_classes == 2  # Line
        circle_mask = target_classes == 3  # Circle
        arc_mask = target_classes == 4  # Arc

        # Compute loss for B-splines
        if bspline_mask.any():
            bspline_order_l1 = torch.min(
                F.l1_loss(
                    src_bspline_params[bspline_mask].flatten(-2, -1),
                    target_params[bspline_mask],
                    reduction="none",
                ).mean(-1),
                F.l1_loss(
                    src_bspline_params[bspline_mask].flip([1]).flatten(-2, -1),
                    target_params[bspline_mask],
                    reduction="none",
                ).mean(-1),
            ).mean()
            losses["loss_bspline"] = bspline_order_l1
            bspline_chamfer = chamfer_distance_batch(
                src_bspline_points[bspline_mask], target_curves[bspline_mask]
            )
            losses["loss_bspline_chamfer"] = bspline_chamfer.mean()
        else:
            losses["loss_bspline"] = torch.tensor(0.0, device=src_bspline_params.device)
            losses["loss_bspline_chamfer"] = torch.tensor(
                0.0, device=src_bspline_points.device
            )

        # Compute loss for Lines
        if line_mask.any():
            line_position_l1 = torch.min(
                F.l1_loss(
                    src_line_params[line_mask].flatten(-2, -1),
                    target_params[line_mask, :6],
                    reduction="none",
                ).mean(-1),
                # also consider the negative direction
                F.l1_loss(
                    (
                        src_line_params[line_mask]
                        * torch.tensor([1.0, -1.0])
                        .view(1, 2, 1)
                        .to(src_line_params.device)
                    ).flatten(-2, -1),
                    target_params[line_mask, :6],
                    reduction="none",
                ).mean(-1),
            ).mean()
            line_length_loss = F.l1_loss(
                src_line_length[line_mask],
                target_params[line_mask, 6].unsqueeze(-1),
            )
            losses["loss_line_position"] = line_position_l1
            losses["loss_line_length"] = line_length_loss
            line_chamfer = chamfer_distance_batch(
                src_line_points[line_mask], target_curves[line_mask]
            )
            losses["loss_line_chamfer"] = line_chamfer.mean()
        else:
            losses["loss_line_position"] = torch.tensor(
                0.0, device=src_line_params.device
            )
            losses["loss_line_length"] = torch.tensor(
                0.0, device=src_line_length.device
            )
            losses["loss_line_chamfer"] = torch.tensor(
                0.0, device=src_line_points.device
            )

        # Compute loss for Circles
        if circle_mask.any():
            circle_position_l1 = torch.min(
                F.l1_loss(
                    src_circle_params[circle_mask].flatten(-2, -1),
                    target_params[circle_mask, :6],
                    reduction="none",
                ).mean(-1),
                # also consider the negative direction
                F.l1_loss(
                    (
                        src_circle_params[circle_mask]
                        * torch.tensor([1.0, -1.0])
                        .view(1, 2, 1)
                        .to(src_circle_params.device)
                    ).flatten(-2, -1),
                    target_params[circle_mask, :6],
                    reduction="none",
                ).mean(-1),
            ).mean()
            radius_loss = F.l1_loss(
                src_circle_radius[circle_mask],
                target_params[circle_mask, 6].unsqueeze(-1),
            )
            losses["loss_circle_position"] = circle_position_l1
            losses["loss_circle_radius"] = radius_loss
            circle_chamfer = chamfer_distance_batch(
                src_circle_points[circle_mask], target_curves[circle_mask]
            )
            losses["loss_circle_chamfer"] = circle_chamfer.mean()
        else:
            losses["loss_circle_position"] = torch.tensor(
                0.0, device=src_circle_params.device
            )
            losses["loss_circle_radius"] = torch.tensor(
                0.0, device=src_circle_radius.device
            )
            losses["loss_circle_chamfer"] = torch.tensor(
                0.0, device=src_circle_points.device
            )

        # Compute loss for Arcs
        if arc_mask.any():
            arc_order_l1 = torch.min(
                F.l1_loss(
                    src_arc_params[arc_mask].flatten(-2, -1),
                    target_params[arc_mask, :9],
                    reduction="none",
                ).mean(-1),
                F.l1_loss(
                    src_arc_params[arc_mask][:, [0, 2, 1]].flatten(-2, -1),
                    target_params[arc_mask, :9],
                    reduction="none",
                ).mean(-1),
            ).mean()
            losses["loss_arc"] = arc_order_l1
            arc_chamfer = chamfer_distance_batch(
                src_arc_points[arc_mask], target_curves[arc_mask]
            )
            losses["loss_arc_chamfer"] = arc_chamfer.mean()
        else:
            losses["loss_arc"] = torch.tensor(0.0, device=src_arc_params.device)
            losses["loss_arc_chamfer"] = torch.tensor(0.0, device=src_arc_points.device)

        losses["total_curve"] = (
            losses["loss_bspline"]
            + losses["loss_line_position"]
            + losses["loss_line_length"]
            + losses["loss_circle_position"]
            + losses["loss_circle_radius"]
            + losses["loss_line_chamfer"]
            + losses["loss_circle_chamfer"]
            + losses["loss_bspline_chamfer"]
            + losses["loss_arc"]
            + losses["loss_arc_chamfer"]
        )

        return losses
