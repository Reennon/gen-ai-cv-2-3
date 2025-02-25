# File: src/models/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel

class MnistVAE(BaseModel):
    def __init__(self, hparams):
        super(MnistVAE, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        # Encoder for MNIST (1 channel, 28x28) produces a flattened size of 128*4*4 = 2048.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # Output: (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, 4, 4)
            nn.ReLU(),
            nn.Flatten(),  # 128*4*4 = 2048
        )
        self.fc_mu = nn.Linear(128 * 4 * 4, self.hparams['latent_dim'])
        self.fc_logvar = nn.Linear(128 * 4 * 4, self.hparams['latent_dim'])

        # Decoder that mirrors the encoder.
        self.decoder_fc = nn.Linear(self.hparams['latent_dim'], 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # (1, 28, 28)
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        # Debug: print(x.shape) should be [B, 2048]
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

    # (training_step, validation_step, etc.)
