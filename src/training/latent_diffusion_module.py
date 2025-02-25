import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.mnist_vae import MnistVAE


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class LatentDiffusionModel(BaseModel):
    def __init__(self, hparams):
        """
        hparams: a dict (or OmegaConf dict) with keys:
            - timesteps: number of diffusion steps (default: 1000)
            - lr: learning rate (default: 1e-3)
            - latent_dim: dimension of the VAE latent space (e.g., 64)
            - diff_hidden_dim: hidden dimension for the latent diffusion network (e.g., 128)
            - time_embed_dim: dimension for time embedding (e.g., 128)
            - (Other keys like optimizer, scheduler, etc.)
        """
        super(LatentDiffusionModel, self).__init__(hparams)
        self.save_hyperparameters(hparams)

        self.timesteps = hparams.get("timesteps", 1000)
        self.lr = hparams.get("lr", 1e-3)
        latent_dim = hparams["latent_dim"]
        diff_hidden_dim = hparams.get("diff_hidden_dim", 128)
        time_embed_dim = hparams.get("time_embed_dim", 128)

        # Instantiate the MNIST VAE (configured for 1-channel, 28x28 images)
        self.vae = MnistVAE(hparams)

        # Define a time embedding network.
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Diffusion network: input is the concatenation of the latent code and time embedding.
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, diff_hidden_dim),
            nn.ReLU(),
            nn.Linear(diff_hidden_dim, latent_dim)
        )

        # Precompute diffusion schedule parameters.
        betas = linear_beta_schedule(self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

        # For storing validation outputs (if needed)
        self.validation_outputs = []

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

        # Normalize t to [0, 1] and compute its embedding.
        t_norm = t.view(-1, 1).float() / self.timesteps
        t_emb = self.time_embed(t_norm)  # Shape: [B, time_embed_dim]

        # Concatenate latent code with time embedding.
        zt = torch.cat([z, t_emb], dim=1)  # Shape: [B, latent_dim + time_embed_dim]
        noise_pred = self.diffusion_net(zt)
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
        t_norm = t.view(-1, 1).float() / self.timesteps

        # Concatenate noisy latent code with time embedding.
        noise_pred = self.diffusion_net(torch.cat([noisy_z, self.time_embed(t_norm)], dim=1))
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        device = x.device

        t = torch.randint(0, self.timesteps, (B,), device=device)
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.view(-1, 1).float() / self.timesteps
        noise_pred = self.diffusion_net(torch.cat([noisy_z, self.time_embed(t_norm)], dim=1))
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)
        self.validation_outputs.append((x, self.vae.decode(z)))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
