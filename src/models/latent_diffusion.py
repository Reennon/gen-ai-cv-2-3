import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.mnist_vae import MnistVAE
from omegaconf import OmegaConf  # if using OmegaConf for hyperparameters


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class LatentDiffusionNetwork(nn.Module):
    """
    A simple MLP to predict noise in the latent space.
    It takes as input a noisy latent code and a normalized timestep.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, time_embed_dim: int = 64):
        super(LatentDiffusionNetwork, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Noisy latent code of shape [B, latent_dim]
            t: Normalized timestep tensor of shape [B] (values in [0,1])
        Returns:
            Predicted noise of shape [B, latent_dim]
        """
        B = z.size(0)
        t_embed = self.time_embed(t.view(B, 1))  # [B, hidden_dim]
        h = self.fc1(z) + t_embed
        h = F.relu(h)
        noise_pred = self.fc2(h)
        return noise_pred


class LatentDiffusion(BaseModel):
    def __init__(self, hparams):
        """
        hparams: A dictionary or OmegaConf dict containing keys such as:
            - timesteps: number of diffusion steps (e.g., 1000)
            - lr: learning rate
            - latent_dim: dimensionality of the VAE latent space
            - diff_hidden_dim: hidden dimension for the latent diffusion network (optional, default: 128)
            - time_embed_dim: dimension for time embedding (optional, default: 64)
            - VAE-related parameters (e.g., latent_dim, etc.)
            - optimizer and scheduler settings
        """
        super(LatentDiffusion, self).__init__(hparams)

        self.timesteps = hparams.get('timesteps', 1000)
        self.latent_dim = hparams['latent_dim']

        # Instantiate the VAE. (Ensure the VAE is configured for MNIST; e.g., 1-channel input.)
        self.vae = MnistVAE(hparams)
        # Optionally, you can freeze the VAE if it's pretrained:
        # for param in self.vae.parameters():
        #     param.requires_grad = False

        # Diffusion network for latent space.
        diff_hidden_dim = hparams.get('diff_hidden_dim', 128)
        time_embed_dim = hparams.get('time_embed_dim', 64)
        self.diffusion_net = LatentDiffusionNetwork(self.latent_dim, diff_hidden_dim, time_embed_dim)

        self.lr = hparams.get('lr', 1e-3)

        # Precompute diffusion schedule parameters (for latent space).
        betas = linear_beta_schedule(self.timesteps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x and a normalized timestep t,
        encodes x into latent space and predicts the noise in the latent code.
        """
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        return self.diffusion_net(z, t)

    def training_step(self, batch, batch_idx):
        # x: [B, 1, H, W] (MNIST images)
        x, _ = batch
        B = x.size(0)
        device = x.device

        # Encode x into latent space.
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)  # z shape: [B, latent_dim]

        # Sample random noise to add in latent space.
        noise = torch.randn_like(z)

        # Sample random timesteps for each latent code.
        t = torch.randint(0, self.timesteps, (B,), device=device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)

        # Create the noisy latent code.
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.float() / self.timesteps  # normalize timestep

        # Predict noise using the diffusion network.
        noise_pred = self.diffusion_net(noisy_z, t_norm)
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        device = x.device

        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        noise = torch.randn_like(z)
        t = torch.randint(0, self.timesteps, (B,), device=device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1)
        noisy_z = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.float() / self.timesteps
        noise_pred = self.diffusion_net(noisy_z, t_norm)
        loss = F.mse_loss(noise_pred, noise)
        self.log("val_loss", loss)
        self.validation_outputs.append((x, self.vae.decode(z)))
        return loss

    @torch.no_grad()
    def sample_ddim(self, num_steps: int = 50, eta: float = 0.0, sample_size: int = 16) -> torch.Tensor:
        """
        Generate samples using deterministic DDIM sampling in the latent space,
        then decode the latent codes using the VAE decoder.

        Args:
            num_steps: Number of sampling steps (fewer than training timesteps).
            eta: Controls stochasticity (0.0 yields deterministic sampling).
            sample_size: Number of samples to generate.

        Returns:
            Generated images tensor of shape [sample_size, channels, H, W]
        """
        device = self.betas.device
        # Start from pure noise in latent space.
        z = torch.randn(sample_size, self.latent_dim, device=device)
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device, dtype=torch.long)

        for i in range(num_steps):
            t = times[i]
            t_tensor = torch.full((sample_size,), t, device=device, dtype=torch.float32)
            t_norm = t_tensor / self.timesteps
            noise_pred = self.diffusion_net(z, t_norm)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            z0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            if t > 0:
                alpha_prev = self.alphas_cumprod[t - 1]
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                noise = torch.randn_like(z) if eta > 0 else 0.0
                z = torch.sqrt(alpha_prev) * z0_pred + torch.sqrt(
                    1 - alpha_prev - sigma ** 2) * noise_pred + sigma * noise
            else:
                z = z0_pred

        # Decode the latent codes to generate images.
        x_gen = self.vae.decode(z)
        return x_gen

    @torch.no_grad()
    def sample_ddpm(self, sample_size: int = 16) -> torch.Tensor:
        """
        Generate samples using the full DDPM reverse process in latent space,
        then decode them using the VAE decoder.

        Args:
            sample_size: Number of samples to generate.

        Returns:
            Generated images tensor of shape [sample_size, channels, H, W]
        """
        device = self.betas.device
        z = torch.randn(sample_size, self.latent_dim, device=device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((sample_size,), t, device=device, dtype=torch.float32)
            t_norm = t_tensor / self.timesteps
            noise_pred = self.diffusion_net(z, t_norm)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            z0_pred = (z - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            if t > 0:
                alpha_prev = self.alphas_cumprod[t - 1]
                noise = torch.randn_like(z)
                z = torch.sqrt(alpha_prev) * z0_pred + torch.sqrt(1 - alpha_prev) * noise
            else:
                z = z0_pred
        x_gen = self.vae.decode(z)
        return x_gen
