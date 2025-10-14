from dataclasses import dataclass, field
from typing import Optional
from torch_geometric.nn import (
    MLP,
)
from .pointnetpp import SAModule


@dataclass
class ModelConfig:
    model: str
    num_features: int
    epochs: int = 1700
    lr: float = 1e-4
    lr_warmup_epochs: int = 15
    lr_warmup_start_factor: float = 1e-6
    lr_step: int = 1230
    batch_size: int = 8
    batch_size_val: int = 8
    loss_weights: Optional[dict[str, float]] = None
    num_curve_points: Optional[int] = 64
    num_curve_points_val: Optional[int] = 256
    preencoder_type: Optional[str] = "samodule"
    preencoder_lr: Optional[float] = 1e-4
    freeze_backbone: bool = False
    encoder_dim: Optional[int] = 768
    decoder_dim: Optional[int] = 768
    num_encoder_layers: Optional[int] = 3
    num_decoder_layers: Optional[int] = 9
    encoder_dropout: float = 0.1
    decoder_dropout: float = 0.1
    num_attn_heads: Optional[int] = 8
    enc_dim_feedforward: Optional[int] = 2048
    dec_dim_feedforward: Optional[int] = 2048
    mlp_dropout: float = 0.0
    num_preds: Optional[int] = 128
    num_classes: Optional[int] = 5
    cost_weights: Optional[dict[str, float]] = None
    auxiliary_loss: bool = True
    max_points_in_param: Optional[int] = 4
    num_transformer_points: Optional[int] = 2048
    query_type: str = "point_fps"
    pos_embed_type: str = "sine"
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

    def get_preencoder(self):
        preencoder_type = self.preencoder_type
        preencoder = None
        if preencoder_type == "samodule":
            preencoder = SAModule(
                MLP([self.num_features + 3, 64, 128, self.encoder_dim]),
                num_out_points=self.num_transformer_points,
            )
            preencoder.out_channels = self.encoder_dim
        else:
            raise ValueError(f"Unknown preencoder type: {self.preencoder_type}.")
        return preencoder
