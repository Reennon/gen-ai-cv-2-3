import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

from src.models.base_model import BaseModel


class VAE(BaseModel):
    def __init__(self, hparams):
        super(VAE, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        # Define encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, self.hparams['latent_dim'])  # Mean
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.hparams['latent_dim'])  # Log variance

        # Define decoder
        self.decoder_fc = nn.Linear(self.hparams['latent_dim'], 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),  # Reshape to match encoder output
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def encode(self, x):
        """Encode input into mean and log variance."""
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Apply reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent space back to image space."""
        x = self.decoder_fc(z)
        return self.decoder(x)

    def forward(self, x):
        """Pass through the full VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)
        recon_loss = mse_loss(x_hat, x, reduction='sum') / x.size(0)  # Reconstruction loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  # KL divergence
        loss = recon_loss + kld_loss
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, logvar = self.forward(x)
        recon_loss = mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        val_loss = recon_loss + kld_loss
        self.log('val_loss', val_loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kld_loss', kld_loss)
        self.validation_outputs.append((x, x_hat))
        return val_loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()  # Call the BaseModel's method if overridden

