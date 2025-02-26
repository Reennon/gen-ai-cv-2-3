import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel

class MnistVAE(BaseModel):
    def __init__(self, hparams):
        super(MnistVAE, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        # 1) Encoder for MNIST (1 channel, 28x28)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (B,32,14,14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (B,64,7,7)
            nn.ReLU()
        )

        # 2) Convs to get mu and logvar
        latent_dim = self.hparams['latent_dim']
        self.conv_mu = nn.Conv2d(64, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(64, latent_dim, kernel_size=1)

        # 3) Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),  # (B,64,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),          # (B,32,28,28)
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),                                        # (B,1,28,28)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)  # (B,64,7,7)
        mu = self.conv_mu(x)         # (B,latent_dim,7,7)
        logvar = self.conv_logvar(x) # (B,latent_dim,7,7)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        print(f"[VAE] Latent Space Output Shape: {z.shape}")
        return z

    def decode(self, z):
        return self.decoder(z)       # (B,1,28,28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        print(f"[VAE] Latent Space Output Shape: {z.shape}")

        x_hat = self.decode(z)
        return x_hat, mu, logvar

