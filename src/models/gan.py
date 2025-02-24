import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb

from src.models.base_model import BaseModel
from torch.autograd import Variable
from pytorch_lightning.loggers import WandbLogger

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 512),
            nn.ReLU(True),
            nn.Unflatten(1, (512, 8, 8))  # Reshape to (512, 8, 8)
        )

        self.upsample_blocks = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # (128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (128, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # (3, 32, 32)
            nn.Tanh()  # Scale to [-1, 1]
        )

    def forward(self, z):
        x = self.initial(z)
        img = self.upsample_blocks(x)
        return img




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (64, 16, 16)
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (128, 8, 8)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (256, 4, 4)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 1),  # Binary classification
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.conv_blocks(img)
        flat_features = self.flatten(features)
        validity = self.fc(flat_features)
        return validity




class GAN(BaseModel):
    def __init__(self, hparams):
        super(GAN, self).__init__(hparams)
        self.generator = Generator(hparams["latent_dim"])
        self.discriminator = Discriminator()
        self.automatic_optimization = False  # Disable automatic optimization

    def forward(self, z):
        return self.generator(z)

    def compute_lipschitz_penalty(self, real_imgs, fake_imgs, discriminator):
        """
        Compute Lipschitz Penalty for the discriminator.
        """
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Randomly interpolate between real and fake images
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs
        interpolates.requires_grad_(True)

        # Get discriminator outputs for interpolated images
        interpolates_preds = discriminator(interpolates)

        # Compute gradients with respect to interpolated images
        gradients = torch.autograd.grad(
            outputs=interpolates_preds,
            inputs=interpolates,
            grad_outputs=torch.ones_like(interpolates_preds),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient norm
        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)

        # Compute Lipschitz penalty
        lipschitz_penalty = ((gradient_norm - 1) ** 2).mean()
        return lipschitz_penalty

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Sample latent noise vector
        z = self.sample_latent(batch_size, device)
        fake_imgs = self.generator(z).detach()

        # Get optimizers
        g_opt, d_opt = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()

        # Real and fake predictions
        real_preds = self.discriminator(real_imgs)
        fake_preds = self.discriminator(fake_imgs)

        # Discriminator loss
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2

        # Compute Lipschitz penalty
        lipschitz_penalty = self.compute_lipschitz_penalty(real_imgs, fake_imgs, self.discriminator)

        # Add Lipschitz penalty to discriminator loss
        lambda_lp = self.hparams.get("lambda_lp", 10)  # Regularization coefficient
        d_loss += lambda_lp * lipschitz_penalty

        # Backpropagation
        self.manual_backward(d_loss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        # Train Generator
        self.toggle_optimizer(g_opt)
        g_opt.zero_grad()
        fake_imgs = self.generator(z)
        fake_preds = self.discriminator(fake_imgs)
        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))
        self.manual_backward(g_loss)
        g_opt.step()
        self.untoggle_optimizer(g_opt)

        # Logging
        self.log('d_loss', d_loss, prog_bar=True)
        self.log('g_loss', g_loss, prog_bar=True)
        self.log('lipschitz_penalty', lipschitz_penalty, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        batch_size = real_imgs.size(0)
        device = real_imgs.device

        # Generate fake images
        z = self.sample_latent(batch_size, device)
        generated_imgs = self.generator(z)

        # Calculate losses (if needed)
        real_preds = self.discriminator(real_imgs)
        fake_preds = self.discriminator(generated_imgs)
        real_loss = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
        d_loss = (real_loss + fake_loss) / 2
        g_loss = nn.BCELoss()(fake_preds, torch.ones_like(fake_preds))

        # Append real and generated images to outputs
        if not hasattr(self, 'validation_outputs'):
            self.validation_outputs = []
        self.validation_outputs.append((real_imgs[:10], generated_imgs[:10]))

        val_loss = d_loss + g_loss
        self.log('val_loss', val_loss, prog_bar=True)

        # Log losses
        self.log('val_d_loss', d_loss, prog_bar=True)
        self.log('val_g_loss', g_loss, prog_bar=True)

        return {'val_loss': val_loss, 'val_d_loss': d_loss, 'val_g_loss': g_loss}

    def sample_latent(self, batch_size, device):
        return torch.randn(batch_size, self.hparams["latent_dim"], device=device)

    def on_validation_epoch_end(self):
        """
        Hook to log generated and real images at the end of the validation epoch.
        Logs real images and generated images as grids to Weights & Biases.
        """
        # Ensure the logger is an instance of WandbLogger
        if isinstance(self.logger, WandbLogger):
            # Retrieve a few samples from the validation outputs
            outputs = self.validation_outputs[:8]  # Take the first 8 outputs
            real_images, generated_images = zip(*outputs)

            # Combine images into single tensors
            real_images = torch.cat(real_images, dim=0)
            generated_images = torch.cat(generated_images, dim=0)

            # Create grids for visualization
            real_grid = torchvision.utils.make_grid(real_images.cpu(), nrow=4, normalize=True)
            generated_grid = torchvision.utils.make_grid(generated_images.cpu(), nrow=4, normalize=True)

            # Log the images to W&B
            self.logger.experiment.log({
                "Real Images": [wandb.Image(real_grid, caption='Real Images')],
                "Generated Images": [wandb.Image(generated_grid, caption='Generated Images')]
            })

        # Clear the stored validation outputs
        self.validation_outputs.clear()

    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(), lr=self.hparams["lr"], betas=(0.5, 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(), lr=self.hparams["lr"], betas=(0.5, 0.999))
        return [g_opt, d_opt]
