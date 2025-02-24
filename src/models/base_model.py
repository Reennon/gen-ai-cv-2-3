import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseModel, self).__init__()
        self.save_hyperparameters(hparams)
        # Initialize a list to store validation outputs
        self.validation_outputs = []

    def forward(self, x):
        # Define the forward pass
        pass

    def on_train_epoch_start(self):
        """
        Hook to log the learning rates of all optimizers at the start of each training epoch.
        """
        # Get the optimizers
        optimizers = self.optimizers()

        # Log the learning rates for each optimizer
        for idx, opt in enumerate(optimizers):
            current_lr = opt.param_groups[0]['lr']
            self.log(f'learning_rate_optimizer_{idx}', current_lr, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        # Implement the training logic
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.mse_loss(y_hat, y)  # Example loss calculation
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Implement the validation logic
        x, y = batch
        y_hat = self.forward(x)
        val_loss = nn.functional.mse_loss(y_hat, y)  # Example validation loss calculation
        self.log('val_loss', val_loss)

        return val_loss

    def configure_optimizers(self):
        optimizer_config = self.hparams['optimizer']
        optimizer_type = optimizer_config['type']
        optimizer_params = {k: v for k, v in optimizer_config.items() if k != 'type'}
        optimizer = getattr(optim, optimizer_type)(self.parameters(), **optimizer_params | {"lr": self.hparams['lr']})

        scheduler_config = self.hparams.get('scheduler', None)
        if scheduler_config and scheduler_config['type'] is not None:
            scheduler_type = scheduler_config['type']
            scheduler_params = scheduler_config['params']
            scheduler = getattr(optim.lr_scheduler, scheduler_type)(optimizer, **scheduler_params)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  # or 'step'
                    'monitor': 'val_loss'  # for schedulers like ReduceLROnPlateau
                }
            }
        else:
            return optimizer

    def on_validation_epoch_end(self):
        """
        Hook to log reconstructed images at the end of the validation epoch.
        Logs the original and reconstructed images as a grid to Weights & Biases.
        """
        # Ensure the logger is an instance of WandbLogger
        if isinstance(self.logger, WandbLogger):
            # Select a few samples from the validation outputs
            outputs = self.validation_outputs[:8]  # Take the first 8 batches
            original_images, reconstructed_images = zip(*outputs)

            # Combine images into a single tensor
            original_images = torch.cat(original_images, dim=0)
            reconstructed_images = torch.cat(reconstructed_images, dim=0)

            # Create grids of original and reconstructed images
            original_grid = torchvision.utils.make_grid(original_images.cpu(), nrow=4, normalize=True)
            reconstructed_grid = torchvision.utils.make_grid(reconstructed_images.cpu(), nrow=4, normalize=True)

            # Log the images to W&B
            self.logger.experiment.log({
                "Original Images": [wandb.Image(original_grid, caption='Original Images')],
                "Reconstructed Images": [wandb.Image(reconstructed_grid, caption='Reconstructed Images')]
            })

        # Clear the stored validation outputs
        self.validation_outputs.clear()

