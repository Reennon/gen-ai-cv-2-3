# File: src/models/diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A simple residual block with two convolutional layers, BatchNorm, and ReLU.
    If the input and output channels differ, a 1x1 convolution adjusts the dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        out += residual
        out = self.relu(out)
        return out


class SelfAttention2d(nn.Module):
    """
    A self-attention layer for 2D feature maps.
    Computes attention over spatial dimensions and applies a residual connection.
    """

    def __init__(self, in_channels: int):
        super(SelfAttention2d, self).__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, C//8]
        proj_key = self.key(x).view(B, -1, H * W)  # [B, C//8, H*W]
        energy = torch.bmm(proj_query, proj_key)  # [B, H*W, H*W]
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)  # [B, C, H*W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # [B, C, H*W]
        out = out.view(B, C, H, W)
        out = self.gamma * out + x
        return out


class SimpleDiffusionModel(nn.Module):
    """
    A diffusion model with attention, BatchNorm, and residual layers.

    This network takes a noisy image and a normalized timestep (between 0 and 1)
    and predicts the noise that was added. The architecture consists of:
      - A time embedding network to condition on the timestep.
      - Two residual blocks (with BatchNorm and ReLU).
      - A self-attention layer to capture long-range spatial dependencies.
      - A final convolution to map features back to the image space.
    """

    def __init__(self, img_channels: int = 1, hidden_dim: int = 64, time_embed_dim: int = 128):
        super(SimpleDiffusionModel, self).__init__()
        # Time embedding: transforms a scalar timestep into an embedding vector.
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        # Project the time embedding to a bias term for the residual features.
        self.fc = nn.Linear(time_embed_dim, hidden_dim)

        # Residual blocks with BatchNorm.
        self.resblock1 = ResidualBlock(img_channels, hidden_dim)
        self.resblock2 = ResidualBlock(hidden_dim, hidden_dim)

        # Self-attention layer to model spatial dependencies.
        self.attn = SelfAttention2d(hidden_dim)

        # Final convolution layer to predict noise.
        self.out_conv = nn.Conv2d(hidden_dim, img_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        :param x: Input image tensor of shape [B, img_channels, H, W].
        :param t: Normalized timestep tensor of shape [B] (values in [0,1]).
        :return: Predicted noise tensor of shape [B, img_channels, H, W].
        """
        # Compute time embedding and project it to match feature dimensions.
        t = t.unsqueeze(-1)  # Shape: [B, 1]
        t_emb = self.time_embed(t)  # Shape: [B, time_embed_dim]
        time_bias = self.fc(t_emb)  # Shape: [B, hidden_dim]

        # Pass the input through the first residual block.
        out = self.resblock1(x)  # Shape: [B, hidden_dim, H, W]
        # Inject time conditioning: broadcast and add the time bias.
        out = out + time_bias.unsqueeze(-1).unsqueeze(-1)
        # Further process through the second residual block.
        out = self.resblock2(out)
        # Apply self-attention to capture global spatial dependencies.
        out = self.attn(out)
        # Map the features back to the image space.
        out = self.out_conv(out)
        return out
