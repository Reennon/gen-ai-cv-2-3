# File: src/models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel

class MnistVAE(BaseModel):
    def __init__(self, hparams):
        super(MnistVAE, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        # Updated encoder for MNIST (1 channel, 28x28)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
        )
        # The flattened dimension is now 128 * 4 * 4 = 2048.
        self.fc_mu = nn.Linear(128 * 4 * 4, self.hparams['latent_dim'])
        self.fc_logvar = nn.Linear(128 * 4 * 4, self.hparams['latent_dim'])

        # Define decoder accordingly.
        self.decoder_fc = nn.Linear(self.hparams['latent_dim'], 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # Output: (1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        loss = recon_loss + kld_loss
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        val_loss = recon_loss + kld_loss
        self.log('val_loss', val_loss)
        self.validation_outputs.append((x, x_hat))
        return val_loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
