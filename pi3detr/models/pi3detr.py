import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch_geometric.data.data import Data
from torch_geometric.nn import MLP
from torch_geometric.utils import to_dense_batch
from .model_config import ModelConfig
from .losses import LossParams, ParametricLoss
from .transformer import Transformer
from .positional_embedding import PositionEmbeddingCoordsSine
from .query_engine import build_query_engine
from pi3detr.dataset import reverse_normalize_and_scale
from ..utils.curve_fitter import (
    torch_bezier_curve,
    torch_line_points,
    generate_points_on_circle_torch,
    torch_arc_points,
)
from ..utils.postprocessing import (
    snap_and_fit_curves,
    filter_predictions,
    iou_filter_point_based,
    iou_filter_predictions,
)

from pi3detr.evaluation.abc_metrics import (
    ChamferMAP,
)
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)


class PI3DETR(pl.LightningModule):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pc_preencoder = config.get_preencoder()
        self.enc_dim = config.encoder_dim
        self.dec_dim = config.decoder_dim
        self.num_preds = config.num_preds
        self.num_curve_points = config.num_curve_points
        self.num_curve_points_val = config.num_curve_points_val
        self.num_classes = config.num_classes
        self.max_points_in_param = config.max_points_in_param
        self.preenc_to_enc_proj = MLP(
            [self.pc_preencoder.out_channels, self.enc_dim, self.enc_dim],
            act="relu",
            norm="layer_norm",
        )
        num_decoder_layers = config.num_decoder_layers
        self.transformer = Transformer(
            self.enc_dim,
            self.dec_dim,
            nhead=config.num_attn_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            enc_dim_feedforward=config.enc_dim_feedforward,
            dec_dim_feedforward=config.dec_dim_feedforward,
            enc_dropout=config.encoder_dropout,
            dec_dropout=config.decoder_dropout,
            return_intermediate_dec=True,
        )
        self.positional_embedding = PositionEmbeddingCoordsSine(
            d_pos=self.dec_dim, pos_type=self.config.pos_embed_type
        )
        self.pos_embed_proj = MLP(
            [self.dec_dim, self.dec_dim, self.dec_dim], act="relu", norm="layer_norm"
        )
        self.query_type = config.query_type
        self.query_engine = build_query_engine(
            self.query_type,
            self.positional_embedding,
            self.dec_dim,
            self.max_points_in_param,
            self.num_preds,
        )

        def make_mlp(out_dim, layers=4, base_dim=None, bias_last=True):
            base_dim = base_dim or self.dec_dim
            n_layers = layers - 1
            return MLP(
                channel_list=[base_dim] * n_layers + [out_dim],
                bias=[False] * (n_layers - 1) + [bias_last],
                dropout=self.config.mlp_dropout,
                act="relu",
                norm="layer_norm",
            )

        self.class_head = make_mlp(self.num_classes)
        self.bspline_param_head = make_mlp(4 * 3)
        self.line_param_head = make_mlp(2 * 3)
        self.line_length_head = make_mlp(1, layers=3)
        self.circle_param_head = make_mlp(2 * 3)
        self.circle_radius_head = make_mlp(1, layers=3)
        self.arc_param_head = make_mlp(3 * 3)

        self.loss = ParametricLoss(
            LossParams(
                num_classes=self.num_classes - 1,  # -1 for the EOS token
                cost_class=config.cost_weights["cost_class"],
                cost_curve=config.cost_weights["cost_curve"],
                class_loss_type=config.class_loss_type,
                class_loss_weights=config.class_loss_weights,
            )
        )
        self.auxiliary_loss = self.config.auxiliary_loss
        self.weight_dict = {
            "loss_class": config.loss_weights["loss_class"],
            "loss_bspline": config.loss_weights["loss_bspline"],
            "loss_bspline_chamfer": config.loss_weights["loss_bspline_chamfer"],
            "loss_line_position": config.loss_weights["loss_line_position"],
            "loss_line_length": config.loss_weights["loss_line_length"],
            "loss_line_chamfer": config.loss_weights["loss_line_chamfer"],
            "loss_circle_position": config.loss_weights["loss_circle_position"],
            "loss_circle_radius": config.loss_weights["loss_circle_radius"],
            "loss_circle_chamfer": config.loss_weights["loss_circle_chamfer"],
            "loss_arc": config.loss_weights["loss_arc"],
            "loss_arc_chamfer": config.loss_weights["loss_arc_chamfer"],
        }
        # TODO this is a hack
        self.aux_weight_dict = {}
        if self.auxiliary_loss:
            for i in range(num_decoder_layers - 1):
                self.aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(self.aux_weight_dict)

        self.chamfer_map = ChamferMAP(chamfer_thresh=0.05)

        # Torchmetrics for segmentation
        self.seg_iou = BinaryJaccardIndex()
        self.seg_precision = BinaryPrecision()
        self.seg_recall = BinaryRecall()

    def forward(self, data: Data) -> dict[str, Tensor]:
        x, pos, batch = self.pc_preencoder(data)[-1]
        x = self.preenc_to_enc_proj(x)
        x, mask = to_dense_batch(x, batch)
        pos_dense_batch, _ = to_dense_batch(pos, batch)
        pos_embed = self.pos_embed_proj(
            self.positional_embedding(
                pos_dense_batch, num_channels=self.dec_dim
            ).permute(0, 2, 1)
        ).permute(0, 2, 1)
        query_xyz, query_embed = self.query_engine(Data(pos=pos, batch=batch))
        x = self.transformer(
            x,  # [batch_size, num_points, enc_dim]
            # transformer expects 1s to be masked
            ~mask if not torch.all(mask) else None,  # [batch_size, num_points]
            query_embed,  # [batch_size, dec_dim, num_queries]
            pos_embed,  # [batch_size, dec_dim, num_points]
        )
        output_class = self.class_head(x)
        output_bspline_params = self.bspline_param_head(x)
        output_line_params = self.line_param_head(x)
        output_line_length = self.line_length_head(x)
        output_circle_params = self.circle_param_head(x)
        output_circle_radius = self.circle_radius_head(x)
        output_arc_params = self.arc_param_head(x)

        pred_bspline_params = (
            output_bspline_params[-1].reshape(data.batch_size, self.num_preds, 4, 3)
            + query_xyz
        )
        pred_line_params = output_line_params[-1].reshape(
            data.batch_size, self.num_preds, 2, 3
        )
        pred_line_params[:, :, 0, :] = (
            pred_line_params[:, :, 0, :] + query_xyz[:, :, 0, :]
        )

        pred_circle_params = output_circle_params[-1].reshape(
            data.batch_size, self.num_preds, 2, 3
        )
        pred_circle_params[:, :, 0, :] = (
            pred_circle_params[:, :, 0, :] + query_xyz[:, :, 0, :]
        )

        pred_arc_params = (
            output_arc_params[-1].reshape(data.batch_size, self.num_preds, 3, 3)
            + query_xyz[:, :, :3, :]
        )

        out = {
            "pred_class": output_class[-1],
            "pred_bspline_params": pred_bspline_params,
            "pred_line_params": pred_line_params,
            "pred_line_length": output_line_length[-1],
            "pred_circle_params": pred_circle_params,
            "pred_circle_radius": output_circle_radius[-1],
            "pred_arc_params": pred_arc_params,
            "query_xyz": query_xyz,
        }
        if self.auxiliary_loss and self.training:
            out["aux_outputs"] = self._set_aux_loss(
                output_bspline_params,
                output_line_params,
                output_line_length,
                output_circle_params,
                output_circle_radius,
                output_arc_params,
                query_xyz,
                output_class,
            )
        return out

    @torch.jit.unused
    def _set_aux_loss(
        self,
        output_bspline_params: torch.Tensor,
        output_line_params: torch.Tensor,
        output_line_length: torch.Tensor,
        output_circle_params: torch.Tensor,
        output_circle_radius: torch.Tensor,
        output_arc_params: torch.Tensor,
        query_xyz: torch.Tensor,
        output_class: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        out_aux = []
        for b, l, ll, c, cr, a, cl in zip(
            output_bspline_params[:-1],
            output_line_params[:-1],
            output_line_length[:-1],
            output_circle_params[:-1],
            output_circle_radius[:-1],
            output_arc_params[:-1],
            output_class[:-1],
        ):
            pred_bspline_params = b.reshape(*b.shape[:2], 4, 3) + query_xyz
            # second point is the direction vector
            pred_line_params = l.reshape(*l.shape[:2], 2, 3)
            pred_line_params_adjusted = pred_line_params.clone()
            pred_line_params_adjusted[:, :, 0, :] = (
                pred_line_params[:, :, 0, :] + query_xyz[:, :, 0, :]
            )
            pred_circle_params = c.reshape(*c.shape[:2], 2, 3)
            pred_circle_params_adjusted = pred_circle_params.clone()
            pred_circle_params_adjusted[:, :, 0, :] = (
                pred_circle_params[:, :, 0, :] + query_xyz[:, :, 0, :]
            )
            pred_arc_params = a.reshape(*a.shape[:2], 3, 3) + query_xyz[:, :, :3, :]

            layer_out = {
                "pred_bspline_params": pred_bspline_params,
                "pred_line_params": pred_line_params_adjusted,
                "pred_line_length": ll,
                "pred_circle_params": pred_circle_params_adjusted,
                "pred_circle_radius": cr,
                "pred_arc_params": pred_arc_params,
                "pred_class": cl,
            }
            layer_out.update(
                self._sample_curve_points(layer_out, self.num_curve_points)
            )
            out_aux.append(layer_out)

        return out_aux

    def _sample_curve_points(
        self, out: dict[str, Tensor], num_points: int
    ) -> dict[str, Tensor]:
        batch_size, num_preds = out["pred_bspline_params"].shape[:2]
        pred_line_params = out["pred_line_params"]
        pred_line_length = out["pred_line_length"]
        pred_line_start = (
            pred_line_params[:, :, 0, :]
            - pred_line_params[:, :, 1, :] * pred_line_length / 2.0
        )
        pred_line_end = (
            pred_line_params[:, :, 0, :]
            + pred_line_params[:, :, 1, :] * pred_line_length / 2.0
        )
        curves = {}
        curves["pred_bspline_points"] = torch_bezier_curve(
            out["pred_bspline_params"].reshape(-1, 4, 3), num_points
        ).reshape(batch_size, num_preds, -1, 3)
        curves["pred_line_points"] = torch_line_points(
            pred_line_start.reshape(-1, 3),
            pred_line_end.reshape(-1, 3),
            num_points,
        ).reshape(batch_size, num_preds, -1, 3)
        curves["pred_circle_points"] = generate_points_on_circle_torch(
            out["pred_circle_params"].reshape(-1, 2, 3)[:, 0],
            out["pred_circle_params"].reshape(-1, 2, 3)[:, 1],
            out["pred_circle_radius"].reshape(-1),
            num_points,
        ).reshape(batch_size, num_preds, -1, 3)
        curves["pred_arc_points"] = torch_arc_points(
            out["pred_arc_params"][:, :, 1, :].reshape(-1, 3),
            out["pred_arc_params"][:, :, 0, :].reshape(-1, 3),
            out["pred_arc_params"][:, :, 2, :].reshape(-1, 3),
            num_points,
        ).reshape(batch_size, num_preds, -1, 3)
        return curves

    def predict_step(
        self,
        batch: Data,
        reverse_norm: bool = True,
        thresholds: list[float] = None,
        snap_and_fit: bool = True,
        iou_filter: bool = False,
    ) -> list[Data]:
        preds = self(batch)
        preds.update(self._sample_curve_points(preds, self.num_curve_points_val))

        outputs = self.decode_predictions(batch, preds, reverse_norm)

        if thresholds:
            outputs = [filter_predictions(data, thresholds) for data in outputs]

        if snap_and_fit:
            outputs = [snap_and_fit_curves(data.clone()) for data in outputs]

        if iou_filter:
            outputs = [iou_filter_predictions(data) for data in outputs]

        return outputs

    def training_step(self, batch: Data, batch_idx: int) -> Tensor:
        outputs = self(batch)
        outputs.update(self._sample_curve_points(outputs, self.num_curve_points))
        loss_dict = self.loss(outputs, batch)
        for k, v in loss_dict.items():
            weight = self.weight_dict[k] if "loss" in k else 1
            self._default_log(k, v * weight)
        # weigh losses and sum them for backpropagation
        weighted_loss_dict = {
            k: loss_dict[k] * self.weight_dict[k] for k in self.weight_dict.keys()
        }
        total_loss = sum(weighted_loss_dict.values())
        self._default_log("loss_train", total_loss)
        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Data, batch_idx: int) -> None:
        outputs = self(batch)
        # sample the training curve points for loss computation
        outputs.update(self._sample_curve_points(outputs, self.num_curve_points))
        loss_dict = self.loss(outputs, batch)
        for k, v in loss_dict.items():
            weight = self.weight_dict[k] if "loss" in k else 1
            self._default_log(f"val_{k}", v * weight)
        without_aux = {
            k for k in self.weight_dict.keys() if k not in self.aux_weight_dict
        }
        self._default_log(
            "loss_val", sum(loss_dict[k] * self.weight_dict[k] for k in without_aux)
        )
        # Sample curve points for validation
        outputs.update(self._sample_curve_points(outputs, self.num_curve_points_val))
        self._compute_metrics(batch, outputs)

    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        metrics = self.chamfer_map.compute()
        for key, value in metrics.items():
            self._default_log(f"val_{key}", value)
        self.chamfer_map.reset()

        # Log segmentation metrics at epoch end
        self._default_log("val_seg_iou", self.seg_iou.compute())
        self._default_log(
            "val_seg_precision",
            self.seg_precision.compute(),
        )
        self._default_log("val_seg_recall", self.seg_recall.compute())
        self.seg_iou.reset()
        self.seg_precision.reset()
        self.seg_recall.reset()

    def test_step(self, batch: Data, batch_idx: int) -> None:
        outputs = self(batch)
        outputs.update(self._sample_curve_points(outputs, self.num_curve_points_val))
        self._compute_metrics(batch, outputs)

    def on_test_epoch_end(self) -> None:
        metrics = self.chamfer_map.compute()
        self.chamfer_map.reset()
        for key, value in metrics.items():
            self.log(f"test_{key}", value, prog_bar=False)

        # Log segmentation metrics at epoch end
        self.log("test_seg_iou", self.seg_iou.compute(), prog_bar=False)
        self.log("test_seg_precision", self.seg_precision.compute(), prog_bar=False)
        self.log("test_seg_recall", self.seg_recall.compute(), prog_bar=False)
        self.seg_iou.reset()
        self.seg_precision.reset()
        self.seg_recall.reset()

    def _compute_metrics(self, batch: Data, preds: dict):
        # segmentation metrics
        outputs = self.decode_predictions(batch, preds, reverse_norm=True)
        for i, output in enumerate(outputs):
            self.seg_iou.update(output.segmentation, output.y_seg)
            self.seg_precision.update(output.segmentation, output.y_seg)
            self.seg_recall.update(output.segmentation, output.y_seg)
        # chamfer metrics
        self.chamfer_map.update(preds, batch)

    def set_num_preds(self, num_preds: int) -> None:
        if num_preds == self.num_preds:
            return
        self.num_preds = num_preds
        old_state = (
            self.query_engine.state_dict()
            if isinstance(self.query_engine, nn.Module)
            else None
        )
        new_engine = build_query_engine(
            self.query_type,
            self.positional_embedding,
            self.dec_dim,
            self.max_points_in_param,
            self.num_preds,
        )
        if old_state is not None:
            new_state = new_engine.state_dict()
            for k, v in old_state.items():
                assert k in new_state, f"Missing parameter in new query engine: {k}"
                nv = new_state[k]
                assert (
                    v.shape == nv.shape
                ), f"Shape mismatch for {k}: {v.shape} != {nv.shape}"
                nv.copy_(v.to(nv.device))
            new_engine.load_state_dict(new_state, strict=True)
        self.query_engine = new_engine.to(self.device)

    @torch.no_grad()
    def decode_predictions(
        self, batch: Data, preds: Data, reverse_norm: bool = True
    ) -> list[Data]:
        outputs = []

        # Vectorized class prediction and score
        preds_class = preds["pred_class"].softmax(-1)
        polyline_class = preds_class.argmax(-1)  # (batch_size, num_preds)
        polyline_score = preds_class.max(-1).values  # (batch_size, num_preds)

        # Prepare all possible polylines: (batch_size, num_preds, num_polypoints, 3)
        bspline_points = preds["pred_bspline_points"]
        line_points = preds["pred_line_points"]
        circle_points = preds["pred_circle_points"]
        arc_points = preds["pred_arc_points"]
        zeros_points = torch.zeros_like(bspline_points)  # EOS/empty

        # Stack all types: (batch_size, num_preds, 4, num_polypoints, 3)
        all_polylines = torch.stack(
            [zeros_points, bspline_points, line_points, circle_points, arc_points],
            dim=2,
        )

        # Gather correct polyline for each prediction
        # polyline_class: (batch_size, num_preds)
        # Need to expand to match all_polylines shape for gather
        idx = (
            polyline_class.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(2)  # shape: (batch_size, num_preds, 1, num_polypoints, 3)
            .expand(-1, -1, 1, self.num_curve_points_val, 3)
        )
        polylines = torch.gather(all_polylines, 2, idx).squeeze(
            2
        )  # (batch_size, num_preds, num_polypoints, 3)

        batch_size = batch.batch_size
        device = batch.pos.device
        segmentations = []
        for i in range(batch_size):
            # If all predicted classes are zero (EOS), segmentation should be all zeros
            if torch.all(polyline_class[i] == 0):
                pc_pts = batch.pos[batch.batch == i]
                segmentation = torch.zeros(
                    pc_pts.shape[0], dtype=torch.long, device=device
                )
                segmentations.append(segmentation)
                continue

            poly_pts = polylines[i, polyline_class[i] != 0].reshape(-1, 3)
            pc_pts = batch.pos[batch.batch == i]  # (num_points_in_cloud, 3)
            dists = torch.cdist(poly_pts, pc_pts)
            closest_idx = dists.argmin(dim=1)
            segmentation = torch.zeros(pc_pts.shape[0], dtype=torch.long, device=device)
            segmentation[closest_idx.unique()] = 1
            segmentations.append(segmentation)

        for i in range(batch.batch_size):
            output = Data(
                pos=batch.pos[batch.batch == i].clone(),  # point cloud
                bspline_points=bspline_points[i],  # prediction of B-spline head
                line_points=line_points[i],  # prediction of line heads
                circle_points=circle_points[i],  # prediction of circle heads
                arc_points=arc_points[i],  # prediction of arc head
                polyline_class=polyline_class[i],  # class of each polyline
                polyline_score=polyline_score[i],  # score of polyline class
                polylines=polylines[i],  # polyline that matches polyline_class
                segmentation=segmentations[
                    i
                ],  # curve segmentation for whole point cloud
                query_xyz=preds["query_xyz"][i],  # query points for the transformer
            )
            if hasattr(batch, "y_seg"):
                output.y_seg = batch.y_seg[batch.batch == i]

            if reverse_norm:
                output.center = batch.center[i]
                output.scale = batch.scale[i]

                output = reverse_normalize_and_scale(
                    output,
                    extra_fields=[
                        "polylines",
                        "bspline_points",
                        "line_points",
                        "circle_points",
                        "arc_points",
                        "query_xyz",
                    ],
                )
            outputs.append(output)

        return outputs

    def _default_log(self, name: str, value: Tensor) -> None:
        batch_size = (
            self.config.batch_size if self.training else self.config.batch_size_val
        )
        self.log(
            name,
            value,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        param_dict = None
        config = self.config
        if config.freeze_backbone:
            for param in self.pc_preencoder.parameters():
                param.requires_grad = False
            param_dict = self.parameters()
        elif config.lr != config.preencoder_lr:
            param_dict = [
                {
                    "params": [
                        p for n, p in self.named_parameters() if "pc_encoder" not in n
                    ]
                },
                {
                    "params": [
                        p for n, p in self.named_parameters() if "pc_encoder" in n
                    ],
                    "lr": self.config.preencoder_lr,
                },
            ]
        else:
            param_dict = self.parameters()
        # ----- OPTIMIZER -----
        optimizer = torch.optim.AdamW(param_dict, lr=self.config.lr)
        # ----- WARMUP SCHEDULER -----
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.lr_warmup_start_factor,  # start near zero
            end_factor=1.0,  # ramp up to base LR
            total_iters=config.lr_warmup_epochs,
        )
        # ----- STEP SCHEDULER -----
        # Drop LR by factor after (step_epoch - warmup_epochs) epochs
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step - config.lr_warmup_epochs,
            gamma=0.1,  # drop LR to 10%
            last_epoch=config.epochs,
        )
        # ----- COMBINE -----
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[config.lr_warmup_epochs],
        )

        return [optimizer], {"scheduler": scheduler, "interval": "epoch"}
