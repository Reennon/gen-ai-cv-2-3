import torch
import torch.nn as nn
import torchvision
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch import optim
import pytorch_lightning as pl

from src.models.base_model import BaseModel
from src.models.gan import Generator, Discriminator


class AffineCoupling(nn.Module):
    def __init__(self, in_features):
        super(AffineCoupling, self).__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2),
            nn.Tanh()  # Scale transformation
        )
        self.translate_net = nn.Sequential(
            nn.Linear(in_features // 2, 128),
            nn.ReLU(),
            nn.Linear(128, in_features // 2)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)  # Split into two halves
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=-1), s.sum(dim=-1)

    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=-1)
        s = self.scale_net(y1)
        t = self.translate_net(y1)
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([x1, x2], dim=-1)


class NormalizingFlow(nn.Module):
    def __init__(self, in_features, num_layers):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([AffineCoupling(in_features) for _ in range(num_layers)])

    def forward(self, x):
        log_det_jacobians = 0
        for layer in self.layers:
            x, log_det_jacobian = layer(x)
            log_det_jacobians += log_det_jacobian
        return x, log_det_jacobians

    def inverse(self, z):
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


class GANWithFlow(BaseModel):
    def __init__(self, hparams):
        super(GANWithFlow, self).__init__(hparams)

        # Components
        self.generator = Generator(self.hparams['latent_dim'])
        self.discriminator = Discriminator()
        self.norm_flow = NormalizingFlow(self.hparams['latent_dim'], self.hparams['flow']['num_layers'])

        # Learning rate
        self.lr = self.hparams['lr']
        self.automatic_optimization = False  # Manual optimization for multiple optimizers

        # Validation storage for logging images
        self.validation_outputs = []

    def forward(self, z):
        z, _ = self.norm_flow(z)  # Transform latent vector using normalizing flow
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Sample noise
        z = torch.randn(batch_size, self.hparams['latent_dim'], device=device)

        # Get optimizers
        g_opt, d_opt = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()
        real_preds = self.discriminator(real_imgs)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_imgs = self(z).detach()
        fake_preds = self.discriminator(fake_imgs)
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        # Train Generator
        self.toggle_optimizer(g_opt)
        g_opt.zero_grad()
        fake_imgs = self(z)
        fake_preds = self.discriminator(fake_imgs)
        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
        self.manual_backward(g_loss)
        g_opt.step()
        self.untoggle_optimizer(g_opt)

        # Log metrics
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Sample noise
        z = torch.randn(batch_size, self.hparams['latent_dim'], device=device)

        # Generate fake images
        fake_imgs = self(z)

        # Discriminator predictions
        real_preds = self.discriminator(real_imgs)
        fake_preds = self.discriminator(fake_imgs)

        # Compute validation loss (Discriminator loss as example)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        val_loss = (real_loss + fake_loss) / 2

        # Log validation loss
        self.log('val_loss', val_loss, prog_bar=True)

        # Store outputs for image logging
        self.validation_outputs.append((real_imgs, fake_imgs))

        return val_loss

    def on_validation_epoch_end(self):
        """
        Hook to log generated images at the end of the validation epoch.
        """
        if isinstance(self.logger, WandbLogger):
            # Select a few samples from the validation outputs
            outputs = self.validation_outputs[:8]  # Take the first 8 batches
            real_images, fake_images = zip(*outputs)

            # Combine images into a single tensor
            real_images = torch.cat(real_images, dim=0)
            fake_images = torch.cat(fake_images, dim=0)

            # Create grids of real and generated images
            real_grid = torchvision.utils.make_grid(real_images.cpu(), nrow=4, normalize=True)
            fake_grid = torchvision.utils.make_grid(fake_images.cpu(), nrow=4, normalize=True)

            # Log the images to W&B
            self.logger.experiment.log({
                "Real Images": [wandb.Image(real_grid, caption='Real Images')],
                "Generated Images": [wandb.Image(fake_grid, caption='Generated Images')]
            })

        # Clear the stored validation outputs
        self.validation_outputs.clear()

    def configure_optimizers(self):
        g_opt = optim.Adam(list(self.generator.parameters()) + list(self.norm_flow.parameters()),
                           lr=self.lr, betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return [g_opt, d_opt]

