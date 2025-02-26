import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

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

        # IMPORTANT: Ensure the UNet’s in_channels and out_channels match the latent space channels.
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
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        # Precompute previous cumulative alphas (for DDIM updates)
        alphas_cumprod_prev = torch.cat([alphas_cumprod.new_ones(1), alphas_cumprod[:-1]], dim=0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

    def forward(self, x, t):
        """
        x: Input image batch [B, 1, 28, 28]
        t: Timestep indices [B] (integers in [0, timesteps))
        Returns:
            Predicted noise in the latent space [B, latent_dim, H, W]
        """
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        t_emb = sinusoidal_time_embedding(t, self.hparams['time_embed_dim'])  # [B, time_embed_dim]
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

        # Create noisy latent code
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise

        # Generate time embeddings
        t_emb = sinusoidal_time_embedding(t, self.hparams['time_embed_dim'])
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
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)

        # For logging: decode the clean latent z for reconstruction visualization.
        with torch.no_grad():
            reconstructed = self.vae.decode(z)
        # Append (original, reconstructed) tuple so that BaseModel logs them.
        self.validation_outputs.append((x, reconstructed))
        return {"val_loss": loss, "reconstructed": reconstructed, "original": x}

    @torch.no_grad()
    def sample_ddpm(self, num_samples=16, device='cuda'):
        """
        Latent-space DDPM sampling:
          1. Sample z_T ~ N(0, I) in latent space.
          2. Iteratively denoise using the diffusion model.
          3. Decode final latent z_0 to image.
        """
        latent_dim = self.hparams["latent_dim"]
        # Assume latent spatial dimensions are provided (e.g., 7x7 for MNIST VAE)
        z_h = self.hparams.get("latent_height", 7)
        z_w = self.hparams.get("latent_width", 7)
        z_t = torch.randn(num_samples, latent_dim, z_h, z_w, device=device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = sinusoidal_time_embedding(t_tensor, self.hparams['time_embed_dim'])
            eps = self.diffusion_model(z_t, t_emb)
            # Get alpha values
            alpha_t = (self.sqrt_alphas_cumprod[t_tensor] ** 2).view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            z0_pred = (z_t - sqrt_one_minus_alpha_t * eps) / sqrt_alpha_t
            if t > 0:
                alpha_prev = self.alphas_cumprod_prev[t_tensor].view(-1, 1, 1, 1)
                noise = torch.randn_like(z_t)
                z_t = torch.sqrt(alpha_prev) * z0_pred + torch.sqrt(1 - alpha_prev) * noise
            else:
                z_t = z0_pred
        x_0 = self.vae.decode(z_t).clamp(0, 1)
        return x_0

    @torch.no_grad()
    def sample_ddim(self, num_steps=50, eta=0.0, num_samples=16, device='cuda'):
        """
        Latent-space DDIM sampling:
          1. Sample z_T ~ N(0, I) in latent space.
          2. Iteratively denoise with the DDIM update rule.
          3. Decode final latent z_0 to image.

        Args:
            num_steps: Number of DDIM steps (fewer than training timesteps).
            eta: Controls stochasticity (eta=0.0 yields deterministic sampling).
            num_samples: Number of samples to generate.
            device: Device on which to sample.
        """
        latent_dim = self.hparams["latent_dim"]
        z_h = self.hparams.get("latent_height", 7)
        z_w = self.hparams.get("latent_width", 7)
        z = torch.randn(num_samples, latent_dim, z_h, z_w, device=device)
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device, dtype=torch.long)
        for i in range(num_steps):
            t = times[i]
            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)
            t_emb = sinusoidal_time_embedding(t_tensor, self.hparams['time_embed_dim'])
            eps = self.diffusion_model(z, t_emb)
            alpha_cumprod_t = self.alphas_cumprod[t_tensor].view(-1, 1, 1, 1)
            sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alpha_cumprod_t)
            z0_pred = (z - sqrt_one_minus_alpha_cumprod_t * eps) / sqrt_alpha_cumprod_t
            if t > 0:
                alpha_prev = self.alphas_cumprod_prev[t_tensor].view(-1, 1, 1, 1)
                # DDIM update: combine the deterministic component with optional noise if eta>0
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_prev))
                noise = torch.randn_like(z) if eta > 0 else 0.0
                z = torch.sqrt(alpha_prev) * z0_pred + torch.sqrt(1 - alpha_prev - sigma ** 2) * eps + sigma * noise
            else:
                z = z0_pred
        x_0 = self.vae.decode(z).clamp(0, 1)
        return x_0

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_start(self):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        for idx, opt in enumerate(optimizers):
            current_lr = opt.param_groups[0]['lr']
            self.log(f'learning_rate_optimizer_{idx}', current_lr, on_step=False, on_epoch=True)
