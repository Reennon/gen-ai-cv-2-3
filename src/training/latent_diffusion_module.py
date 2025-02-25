import torch
import torch.nn.functional as F
from src.models.base_model import BaseModel
from src.models.diffusion import SimpleUnconditionalDiffusion
from src.models.latent_diffusion import LatentDiffusionNetwork, linear_beta_schedule
from src.training.diffusion_module import DiffusionModel


class LatentDiffusionModel(DiffusionModel):
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
        self.model = LatentDiffusionNetwork(
            latent_dim=hparams.get('latent_dim', 128),
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
