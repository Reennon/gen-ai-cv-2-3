import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.mnist_vae import MnistVAE
from src.models.unet import TimeEmbeddingUNet  # Your UNet with time embedding

def sinusoidal_time_embedding(timesteps, embedding_dim):
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad if embedding_dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class LatentDiffusionModel(BaseModel):
    def __init__(self, hparams):
        super(LatentDiffusionModel, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.timesteps = hparams.get("timesteps", 1000)
        self.lr = hparams.get("lr", 1e-3)
        latent_dim = hparams["latent_dim"]  # expected to be 64
        time_embed_dim = hparams.get("time_embed_dim", 64)

        # Instantiate the MNIST VAE (should output a latent feature map with shape [B, latent_dim, H, W])
        self.vae = MnistVAE(hparams)

        # IMPORTANT: Make sure the UNet receives the correct number of channels.
        # We set init_features equal to latent_dim so that the time embedding,
        # after being processed, has the same number of channels as the latent code.
        self.diffusion_model = TimeEmbeddingUNet(
            in_channels=latent_dim,
            out_channels=latent_dim,
            time_embedding_dim=time_embed_dim,
            init_features=latent_dim  # set init_features to latent_dim (e.g. 64)
        )

        # Precompute diffusion schedule parameters
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
            Predicted noise in the latent space [B, latent_dim, H, W]
        """
        # Encode image to latent space (ensure your MnistVAE returns a spatial latent map)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)

        t_emb = sinusoidal_time_embedding(t, self.hparams['time_embed_dim'])  # [B, time_embed_dim]
        print(f"[LDM] Input to UNet Shape: {z.shape}, Time Embedding Shape: {t_emb.shape}")
        noise_pred = self.diffusion_model(z, t_emb)

        return noise_pred

    def training_step(self, batch, batch_idx):
        x, _ = batch  # x: [B, 1, 28, 28]
        B = x.size(0)
        device = x.device

        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # [B, latent_dim, H, W]
        noise = torch.randn_like(z)

        # Retrieve diffusion schedule parameters for each timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # Create noisy latent code
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise

        # Generate time embeddings
        t_emb = sinusoidal_time_embedding(t, self.hparams['time_embed_dim'])  # [B, time_embed_dim]
        noise_pred = self.diffusion_model(noisy_z, t_emb)

        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch  # [B, 1, 28, 28]
        B = x.size(0)
        device = x.device

        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_emb = sinusoidal_time_embedding(t, self.hparams['time_embed_dim'])
        noise_pred = self.diffusion_model(noisy_z, t_emb)

        if noise_pred.size() != noise.size():
            noise_pred = F.interpolate(noise_pred, size=noise.size()[2:], mode='bilinear', align_corners=False)

        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        """
        Log learning rates at the beginning of each training epoch.
        """
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for idx, opt in enumerate(optimizers):
            current_lr = opt.param_groups[0]['lr']
            self.log(f'learning_rate_optimizer_{idx}', current_lr, on_step=False, on_epoch=True)
