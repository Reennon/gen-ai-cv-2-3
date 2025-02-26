import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from src.models.base_model import BaseModel


class MnistVAE(pl.LightningModule):
    def __init__(self, latent_dimension=20):
        super().__init__()
        self.latent_dimension = latent_dimension

        # Encoder network
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.mu_layer = nn.Linear(64 * 7 * 7, latent_dimension)
        self.logvar_layer = nn.Linear(64 * 7 * 7, latent_dimension)

        # Decoder network
        self.latent_to_feat = nn.Linear(latent_dimension, 64 * 7 * 7)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        # Encoding
        encoded = self.enc(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.sample_latent(mu, logvar)

        # Decoding
        features = self.latent_to_feat(z).view(-1, 64, 7, 7)
        x_recon = self.dec(features)
        return x_recon, mu, logvar

    def training_step(self, batch, batch_idx):
        inputs, _ = batch
        reconstructions, mu, logvar = self(inputs)
        recon_loss = F.mse_loss(reconstructions, inputs, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


