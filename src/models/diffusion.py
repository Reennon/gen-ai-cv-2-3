# File: src/models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDiffusionModel(nn.Module):
    """
    A simple diffusion model that uses a time-conditioned CNN.
    It takes a noisy image and a normalized timestep (between 0 and 1)
    and predicts the noise that was added.
    """
    def __init__(self, img_channels: int = 1, hidden_dim: int = 64, time_embed_dim: int = 128):
        super(SimpleDiffusionModel, self).__init__()
        # Time embedding: transforms a scalar timestep into an embedding vector.
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        # Project the time embedding to match the number of feature maps.
        self.fc = nn.Linear(time_embed_dim, hidden_dim)
        # A simple CNN for noise prediction.
        self.conv1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        :param x: Input image tensor of shape [B, img_channels, H, W].
        :param t: Normalized timestep tensor of shape [B] (values in [0,1]).
        :return: Predicted noise of shape [B, img_channels, H, W].
        """
        # Expect t to be normalized between 0 and 1 and of shape [B]
        t = t.unsqueeze(-1)  # Shape: [B, 1]
        t_emb = self.time_embed(t)  # Shape: [B, time_embed_dim]
        # Map embedding to feature map shape and add to convolution output.
        t_emb = self.fc(t_emb).unsqueeze(-1).unsqueeze(-1)  # Shape: [B, hidden_dim, 1, 1]
        h = F.relu(self.conv1(x) + t_emb)
        h = F.relu(self.conv2(h))
        out = self.conv3(h)
        return out
