# File: src/models/latent_diffusion_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.mnist_vae import MnistVAE  # Ensure this file contains your MNIST VAE implementation


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class LatentDiffusionModel(BaseModel):
    def __init__(self, hparams):
        """
        hparams: A dict or OmegaConf dict with keys:
            - timesteps: number of diffusion steps (default: 1000)
            - lr: learning rate (default: 1e-3)
            - latent_dim: dimension of the VAE latent space (e.g., 64)
            - diff_hidden_dim: hidden dimension for the latent diffusion network (e.g., 128)
            - time_embed_dim: dimension for time embedding (e.g., 128)
            - (Other keys like optimizer, scheduler, etc.)
        """
        super(LatentDiffusionModel, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.timesteps = hparams.get("timesteps", 1000)
        self.lr = hparams.get("lr", 1e-3)
        # Instantiate the MNIST VAE using the full hparams (which should include latent_dim)
        self.vae = MnistVAE(hparams)
        latent_dim = hparams["latent_dim"]
        diff_hidden_dim = hparams.get("diff_hidden_dim", 128)
        time_embed_dim = hparams.get("time_embed_dim", 128)

        # Define the diffusion network:
        # It takes as input a concatenation of the latent code (of size latent_dim)
        # and a normalized timestep (1 value), and outputs a vector of size latent_dim.
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + 1, diff_hidden_dim),
            nn.ReLU(),
            nn.Linear(diff_hidden_dim, latent_dim)
        )

        # Precompute diffusion schedule parameters
        betas = linear_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # For storing validation outputs (if BaseModel supports visualization)
        self.validation_outputs = []

    def forward(self, x, t):
        """
        x: Input image batch (e.g., [B, 1, 28, 28])
        t: Timestep indices (tensor of shape [B])
        Returns:
            Predicted noise in latent space (tensor of shape [B, latent_dim])
        """
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # Shape: [B, latent_dim]
        # Normalize t to [0, 1]
        t_norm = t.view(-1, 1).float() / self.timesteps
        # Concatenate latent code with normalized timestep
        zt = torch.cat([z, t_norm], dim=1)
        noise_pred = self.diffusion_net(zt)
        return noise_pred

    def training_step(self, batch, batch_idx):
        x, _ = batch  # x: [B, 1, 28, 28]
        B = x.size(0)
        device = x.device

        # Sample random timesteps for each image in the batch
        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # [B, latent_dim]
        # Sample random noise to add in latent space
        noise = torch.randn_like(z)
        # Get diffusion schedule parameters for sampled timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)
        # Create noisy latent code
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.view(-1, 1).float() / self.timesteps
        # Predict noise in latent space
        noise_pred = self.diffusion_net(torch.cat([noisy_z, t_norm], dim=1))
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        device = x.device

        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.view(-1, 1).float() / self.timesteps
        noise_pred = self.diffusion_net(torch.cat([noisy_z, t_norm], dim=1))
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)
        self.validation_outputs.append((x, self.vae.decode(z)))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
