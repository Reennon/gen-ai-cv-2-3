import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.mnist_vae import MnistVAE
from src.models.unet import TimeEmbeddingUNet  # Your UNet with time embedding

class LatentDiffusionModel(BaseModel):
    def __init__(self, hparams):
        super(LatentDiffusionModel, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.timesteps = hparams.get("timesteps", 1000)
        self.lr = hparams.get("lr", 1e-3)
        latent_dim = hparams["latent_dim"]
        time_embed_dim = hparams.get("time_embed_dim", 64)  # Must match UNet's expected time embedding dimension

        # Instantiate the MNIST VAE
        self.vae = MnistVAE(hparams)

        # Define the diffusion model using the U-Net with time embedding.
        # The UNet is responsible for processing both the latent code and time internally.
        self.diffusion_model = TimeEmbeddingUNet(
            in_channels=latent_dim,
            out_channels=latent_dim,
            time_embedding_dim=time_embed_dim
        )

        # Precompute diffusion schedule parameters.
        betas = torch.linspace(0.0001, 0.02, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def forward(self, x, t):
        """
        x: Input image batch [B, 1, 28, 28]
        t: Timestep indices [B] (integers in [0, timesteps))
        Returns:
            Predicted noise in the latent space [B, latent_dim]
        """
        # Encode image to latent space.
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # Shape: [B, latent_dim]

        # Normalize time (values in [0,1])
        t_norm = t.view(-1, 1).float() / self.timesteps

        # Let the UNet internally handle time embedding.
        noise_pred = self.diffusion_model(z, t_norm)
        return noise_pred

    def training_step(self, batch, batch_idx):
        x, _ = batch  # x: [B, 1, 28, 28]
        B = x.size(0)
        device = x.device

        # Sample random timesteps.
        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # [B, latent_dim]
        noise = torch.randn_like(z)

        # Retrieve diffusion schedule parameters for each sampled timestep.
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)

        # Create noisy latent code.
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise

        # Normalize time.
        t_norm = t.view(-1, 1).float() / self.timesteps

        # Predict noise using the diffusion model.
        noise_pred = self.diffusion_model(noisy_z, t_norm)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # x: [B, 1, 28, 28]
        B = x.size(0)
        device = x.device

        # Sample random timesteps.
        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise

        t_norm = t.view(-1, 1).float() / self.timesteps

        noise_pred = self.diffusion_model(noisy_z, t_norm)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        """
        Hook to log the learning rates of all optimizers at the start of each training epoch.
        """
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for idx, opt in enumerate(optimizers):
            current_lr = opt.param_groups[0]['lr']
            self.log(f'learning_rate_optimizer_{idx}', current_lr, on_step=False, on_epoch=True)
