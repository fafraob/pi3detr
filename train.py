import os
import argparse
import torch
from copy import copy
from datetime import datetime
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pi3detr import (
    load_args,
    load_weights,
    build_model_config,
    build_model,
    build_dataset_config,
    build_dataset,
    ModelConfig,
)


torch.set_float32_matmul_precision("medium")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="path to config"
    )
    ## add argument for specifying gpus as list
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        help="List of GPU ids to use for training (default: [0])",
    )
    return parser.parse_args()


def create_log_dir_name(model_config: ModelConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = timestamp.replace("_", "")
    name += f"_{model_config.preencoder_type}_"
    name += f"e{model_config.num_encoder_layers}_d{model_config.num_decoder_layers}_"
    name += f"ed{model_config.encoder_dim}_dd{model_config.decoder_dim}"

    return name


def setup_trainer(
    logger: TensorBoardLogger,
    epochs: int,
    monitor: str,
    mode: str = "min",
    grad_clip_val: float = 0.0,
    gpus: list[int] = [0],
    accumulate_grad_batches: int = 1,
    val_check_interval: int = 1,
) -> pl.Trainer:
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_last=True,
        save_top_k=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        gradient_clip_val=grad_clip_val,
        gradient_clip_algorithm="norm",
        max_epochs=epochs,
        accelerator="gpu",
        callbacks=[checkpoint_callback, lr_monitor],
        devices=gpus,
        accumulate_grad_batches=accumulate_grad_batches,
        logger=logger,
        check_val_every_n_epoch=val_check_interval,
    )
    return trainer


def main() -> None:
    pl.seed_everything(42)
    parsed_args = parse_args()
    args = load_args(parsed_args.config, parsed_args)
    train_dataset_config = build_dataset_config(args, args.data_root, args.augment)
    train_dataset = build_dataset(train_dataset_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if args.data_val_root:
        val_dataset_config = copy(train_dataset_config)
        val_dataset_config.root = args.data_val_root
        val_dataset_config.augment = False
        val_dataset = build_dataset(val_dataset_config)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            shuffle=False,
            num_workers=args.num_workers,
        )
    model_config = build_model_config(args)
    model = build_model(model_config)
    if args.weights:
        load_weights(model, args.weights)
    logger = TensorBoardLogger(
        os.getcwd(), name="lightning_logs", version=create_log_dir_name(model_config)
    )
    logger.log_hyperparams(vars(model_config))
    logger.log_hyperparams(vars(train_dataset_config))
    trainer = setup_trainer(
        logger,
        args.epochs,
        monitor=args.to_monitor,
        mode=args.monitor_mode,
        grad_clip_val=args.grad_clip_val,
        gpus=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_interval,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
