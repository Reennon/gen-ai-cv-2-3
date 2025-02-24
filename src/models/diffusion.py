# File: src/models/simple_uncond_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUnconditionalDiffusion(nn.Module):
    """
    A simplified, unconditional diffusion model for MNIST.

    This model uses a time embedding to condition on the timestep
    and a simple CNN with BatchNorm and a residual connection to predict
    the noise added to the image.
    """

    def __init__(self, img_channels: int = 1, hidden_dim: int = 64, time_embed_dim: int = 128):
        super(SimpleUnconditionalDiffusion, self).__init__()
        # Time embedding: transforms a scalar timestep into an embedding vector.
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, hidden_dim)
        )

        # Convolutional block.
        self.conv1 = nn.Conv2d(img_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)

        # Residual branch (1x1 conv if dimensions differ).
        self.res_conv = nn.Conv2d(img_channels, hidden_dim, kernel_size=1) if img_channels != hidden_dim else None

        # Final convolution to project features back to image space.
        self.out_conv = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input noisy image [B, img_channels, H, W].
            t: Normalized timestep tensor [B] (values in [0,1]).
        Returns:
            Predicted noise tensor [B, img_channels, H, W].
        """
        B = x.size(0)
        # Create time embedding.
        t_embed = self.time_embed(t.view(B, 1))  # [B, hidden_dim]
        t_embed = t_embed.view(B, -1, 1, 1)  # [B, hidden_dim, 1, 1]

        # First convolutional block.
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + t_embed)  # inject time conditioning

        # Second conv block with residual connection.
        residual = self.res_conv(x) if self.res_conv is not None else x
        out2 = self.conv2(out)
        out2 = self.bn2(out2)
        out2 = F.relu(out2 + residual)

        # Final projection.
        out = self.out_conv(out2)
        return out
