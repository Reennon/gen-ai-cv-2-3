# File: src/training/diffusion_module.py
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from models.diffusion import SimpleDiffusionModel


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionLitModule(pl.LightningModule):
    def __init__(self, hparams: dict):
        """
        hparams: Dictionary containing hyperparameters, for example:
            {
                'epochs': 10,
                'timesteps': 1000,
                'lr': 1e-3,
                'img_channels': 1,
                'hidden_dim': 64,
                'time_embed_dim': 128
            }
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        # Initialize the diffusion model.
        self.model = SimpleDiffusionModel(
            img_channels=self.hparams.get('img_channels', 1),
            hidden_dim=self.hparams.get('hidden_dim', 64),
            time_embed_dim=self.hparams.get('time_embed_dim', 128)
        )
        self.timesteps = self.hparams.get('timesteps', 1000)
        self.lr = self.hparams.get('lr', 1e-3)

        # Precompute diffusion schedule parameters.
        self.register_buffer('betas', linear_beta_schedule(self.timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        # Batch is expected to be a tuple (images, labels); we only use images.
        x, _ = batch  # x shape: [B, 1, 28, 28]
        batch_size = x.size(0)
        # Sample a random timestep for each image.
        t = torch.randint(0, self.timesteps, (batch_size,), device=x.device)
        # Retrieve precomputed coefficients.
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        # Sample random Gaussian noise.
        noise = torch.randn_like(x)
        # Create the noisy images.
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        # Normalize t (between 0 and 1) for network input.
        t_norm = t.float() / self.timesteps
        predicted_noise = self.model(noisy_x, t_norm)
        loss = F.mse_loss(predicted_noise, noise)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample_ddim(self, num_steps: int = 50, eta: float = 0.0, shape: tuple = (1, 1, 28, 28)) -> torch.Tensor:
        """
        Deterministic sampling using DDIM.
        :param num_steps: Number of sampling steps (fewer than training timesteps).
        :param eta: Controls stochasticity (0.0 yields deterministic sampling).
        :param shape: Shape of the generated image batch.
        :return: Generated images tensor.
        """
        device = self.betas.device
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.timesteps - 1, 0, num_steps, device=device, dtype=torch.long)
        for i in range(num_steps):
            t = times[i]
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.float32)
            t_norm = t_tensor / self.timesteps
            predicted_noise = self.model(x, t_norm)
            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x0_pred = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
            if t > 0:
                alpha_prev = self.alphas_cumprod[t - 1]
                sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
                noise = torch.randn_like(x) if eta > 0 else 0.0
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(
                    1 - alpha_prev - sigma ** 2) * predicted_noise + sigma * noise
            else:
                x = x0_pred
        return x
