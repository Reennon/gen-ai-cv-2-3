import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class LatentDiffusionModel(pl.LightningModule):
    def __init__(self, pretrained_vae, latent_dim=20, total_timesteps=1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_timesteps = total_timesteps
        self.pretrained_vae = pretrained_vae

        # A basic MLP to predict noise
        self.noise_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, latent_input, time_idx):
        return self.noise_predictor(latent_input)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        # Encode images using the provided VAE (parameters not updated here)
        with torch.no_grad():
            _, mu, _ = self.pretrained_vae(images)

        # Sample a random timestep for each image in the batch
        time_tensor = torch.randint(0, self.total_timesteps, (images.size(0),), device=self.device).long()
        random_noise = torch.randn_like(mu)
        noisy_latent = mu + random_noise

        # Predict the noise and compute loss
        predicted_noise = self(noisy_latent, time_tensor)
        loss = F.mse_loss(predicted_noise, random_noise)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
