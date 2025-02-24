# File: src/models/diffusion_model.py
import torch
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.diffusion import SimpleUnconditionalDiffusion


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


class DiffusionModel(BaseModel):
    def __init__(self, hparams):
        """
        hparams should contain keys such as:
            - 'timesteps': number of diffusion steps (e.g., 1000)
            - 'lr': learning rate
            - 'img_channels': number of image channels (e.g., 1 for MNIST)
            - 'hidden_dim': CNN hidden dimension
            - 'time_embed_dim': dimension for time embedding
            - optimizer and scheduler settings in 'optimizer' and 'scheduler'
        """
        super(DiffusionModel, self).__init__(hparams)
        self.timesteps = hparams.get('timesteps', 1000)

        # Initialize the unconditional diffusion network.
        self.model = SimpleUnconditionalDiffusion(
            img_channels=hparams.get('img_channels', 1),
            hidden_dim=hparams.get('hidden_dim', 64),
            time_embed_dim=hparams.get('time_embed_dim', 128)
        )
        self.lr = hparams.get('lr', 1e-3)

        # Precompute the diffusion schedule.
        self.register_buffer('betas', linear_beta_schedule(self.timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def on_train_epoch_start(self):
        """
        Hook to log the learning rates of all optimizers at the start of each training epoch.
        """
        optimizers = self.optimizers()
        # Ensure optimizers is iterable.
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        # Log the learning rates for each optimizer.
        for idx, opt in enumerate(optimizers):
            current_lr = opt.param_groups[0]['lr']
            self.log(f'learning_rate_optimizer_{idx}', current_lr, on_step=False, on_epoch=True)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that delegates to the unconditional diffusion network.
        """
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        # For diffusion training, labels are not used.
        x, _ = batch  # x shape: [B, img_channels, H, W]
        B = x.size(0)
        device = x.device

        # Sample a random timestep t (integer in [0, timesteps-1]).
        t = torch.randint(0, self.timesteps, (B,), device=device)

        # Retrieve corresponding diffusion schedule parameters.
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)

        # Sample random noise.
        noise = torch.randn_like(x)

        # Create noisy images.
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise

        # Normalize t to [0,1] for the network.
        t_norm = t.float() / self.timesteps

        # Predict noise using the diffusion model.
        predicted_noise = self.model(noisy_x, t_norm)
        loss = F.mse_loss(predicted_noise, noise)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        B = x.size(0)
        device = x.device
        t = torch.randint(0, self.timesteps, (B,), device=device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(B, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1, 1)
        noise = torch.randn_like(x)
        noisy_x = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        t_norm = t.float() / self.timesteps
        predicted_noise = self.model(noisy_x, t_norm)
        loss = F.mse_loss(predicted_noise, noise)
        self.log("val_loss", loss)

        # For visualization, store original and noisy images.
        self.validation_outputs.append((x, noisy_x))
        return loss
