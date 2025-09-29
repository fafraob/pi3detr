import argparse
from typing import Union, Optional
import torch.nn as nn
from torch_geometric.data import Dataset
import inspect
from .models import ModelConfig
from .utils import load_args, load_weights
from .models import PI3DETR
from .dataset import DatasetConfig, ABCDataset


def build_model_config(args: Union[argparse.Namespace, str]) -> ModelConfig:
    if isinstance(args, str):
        args = load_args(args)

    # Get required parameters from ModelConfig constructor
    model_config_signature = inspect.signature(ModelConfig.__init__)
    required_params = [
        param for param in model_config_signature.parameters.keys() if param != "self"
    ]

    for param in required_params:
        if not hasattr(args, param):
            print(f"ERROR: Parameter '{param}' has to be specified in the arguments")
            raise ValueError(f"Missing required parameter: {param}")

    # Create model config with all parameters from args
    model_config_args = {param: getattr(args, param) for param in required_params}
    model_config = ModelConfig(**model_config_args)

    print(model_config)
    return model_config


def build_dataset_config(
    args: Union[argparse.Namespace, str], data_root: str, augment: bool
) -> DatasetConfig:
    if isinstance(args, str):
        args = load_args(args)

    # Get required parameters from DatasetConfig constructor (excluding root and augment)
    dataset_config_signature = inspect.signature(DatasetConfig.__init__)
    required_params = [
        param
        for param in dataset_config_signature.parameters.keys()
        if param not in ["self", "root", "augment"]
    ]

    for param in required_params:
        if not hasattr(args, param):
            print(f"ERROR: Parameter '{param}' has to be specified in the arguments")
            raise ValueError(f"Missing required parameter: {param}")

    # Create dataset config with parameters from args plus root and augment
    dataset_config_args = {param: getattr(args, param) for param in required_params}
    dataset_config = DatasetConfig(
        root=data_root, augment=augment, **dataset_config_args
    )

    print(dataset_config)
    return dataset_config


def build_dataset(config: DatasetConfig) -> Dataset:
    if config.dataset == "abc_dataset":
        return ABCDataset(config)
    else:
        raise ValueError(f"Unknown dataset {config.dataset}")


def build_model(config: ModelConfig) -> nn.Module:
    print(f"Model: {config.model}")
    if config.model == "pi3detr":
        return PI3DETR(config)
    else:
        raise ValueError(f"Unknown model {config.model}")
