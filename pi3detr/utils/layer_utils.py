from typing import Optional
import torch
from torch import nn


def load_weights(
    model: nn.Module, ckpt_path: str, dont_load: Optional[list[str]] = []
) -> None:
    ckpt = torch.load(ckpt_path, weights_only=False)
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if not any([dl in k for dl in dont_load]):
            state_dict[k] = v
        else:
            print(f"Didn't load {k}")
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")


def no_grad(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
