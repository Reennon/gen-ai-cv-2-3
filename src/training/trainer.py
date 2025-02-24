from typing import Optional

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import torch


default_project_name = 'CIFAR10-Training'


def train_model(model_class, params, train_loader, val_loader, wandb_project_name: Optional[str] = default_project_name):
    """
    Trains a model using PyTorch Lightning.

    Args:
        model_class: The model class to be instantiated and trained.
        params: A dictionary of parameters for training and experiment setup.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        wandb_project_name: Optional[str] Name of the project in wandb.
    :returns
        model: Init model after training.
    """
    # Initialize the model
    hparams = params.hyperparameters
    model = model_class(hparams)

    # Initialize Wandb logger
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=params["run_parameters"].get('experiment_name'),
        log_model=True
    )

    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=hparams['epochs'],
        logger=wandb_logger,
        accelerator=params.training.accelerator,
        devices=params.training.devices,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss',
                save_top_k=3,
                mode='min'
            )
        ]
    )

    # Start training
    trainer.fit(model, train_loader, val_loader)

    return model
