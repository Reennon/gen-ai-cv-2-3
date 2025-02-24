import torch.nn as nn

from src.models.base_model import BaseModel


class Autoencoder(BaseModel):
    def __init__(self, hparams):
        super(Autoencoder, self).__init__(hparams)
        self.save_hyperparameters(hparams)
        # Define the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: (256, 4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, self.hparams['latent_dim'])  # Latent representation
        )

        # Define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams['latent_dim'], 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # Reshape to match the encoder's output
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: (128, 8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: (3, 32, 32)
            nn.Sigmoid()  # Scale output to [0, 1]
        )

    def forward(self, x):
        # Pass through encoder
        z = self.encoder(x)
        # Pass through decoder
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = nn.functional.mse_loss(x_hat, x)
        self.log('val_loss', val_loss)
        # Store outputs for later use
        self.validation_outputs.append((x, x_hat))
        return val_loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()  # Call the BaseModel's method if overridden

