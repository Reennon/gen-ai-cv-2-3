import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbeddingUNet(nn.Module):
    """
    A UNet that injects a time embedding into each stage of the encoder
    and the bottleneck. The "init_features" determines how many channels
    we have at the first encoder block. Subsequent blocks double that
    number at each level.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim, init_features=32):
        super(TimeEmbeddingUNet, self).__init__()

        features = init_features
        # -----------------
        #   ENCODER
        # -----------------
        # Block 1: in_channels -> features (32)
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2: features (32) -> features*2 (64)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 3: features*2 (64) -> features*4 (128)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 4: features*4 (128) -> features*8 (256)
        self.encoder4 = self._block(features * 4, features * 8)

        # -----------------
        #  BOTTLENECK
        # -----------------
        # Bottleneck: features*8 (256) -> features*16 (512)
        self.bottleneck = self._block(features * 8, features * 16)

        # -----------------
        #   DECODER
        # -----------------
        # Each upconv halves the channel dimension vs. the bottleneck or previous stage
        # and we then concatenate the skip connection from the encoder.

        # Upconv4: 512 -> 256, then decoder4 merges with encoder4 skip => 512 -> 256
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False)
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)

        # Upconv3: 256 -> 128, then merges with encoder3 => 256 -> 128
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False)
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)

        # Upconv2: 128 -> 64, then merges with encoder2 => 128 -> 64
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1, bias=False)
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)

        # Upconv1: 64 -> 32, then merges with encoder1 => 64 -> 32
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False)
        )
        self.decoder1 = self._block(features * 2, features)

        # Final 1x1 convolution to get desired out_channels
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # -----------------
        # TIME EMBEDDINGS
        # -----------------
        # We project the time embedding to the same number of channels
        # as each encoder level output, plus the bottleneck.
        self.time_proj1 = nn.Linear(time_embedding_dim, features)  # => 32
        self.time_proj2 = nn.Linear(time_embedding_dim, features * 2)  # => 64
        self.time_proj3 = nn.Linear(time_embedding_dim, features * 4)  # => 128
        self.time_proj4 = nn.Linear(time_embedding_dim, features * 8)  # => 256
        self.time_proj_bottleneck = nn.Linear(time_embedding_dim, features * 16)  # => 512

    def forward(self, x, time_emb):
        """
        Args:
            x         : (B, in_channels, H, W) - latent input (e.g. from VAE)
            time_emb  : (B, time_embedding_dim) - embedding for the current diffusion timestep

        Returns:
            out : (B, out_channels, H, W)
        """
        B = x.size(0)

        # Project the time embedding to match each encoder stage + bottleneck
        t1 = self.time_proj1(time_emb).view(B, -1, 1, 1)  # => [B, 32, 1, 1]
        t2 = self.time_proj2(time_emb).view(B, -1, 1, 1)  # => [B, 64, 1, 1]
        t3 = self.time_proj3(time_emb).view(B, -1, 1, 1)  # => [B, 128, 1, 1]
        t4 = self.time_proj4(time_emb).view(B, -1, 1, 1)  # => [B, 256, 1, 1]
        tb = self.time_proj_bottleneck(time_emb).view(B, -1, 1, 1)  # => [B, 512, 1, 1]

        # -----------------
        #   ENCODER
        # -----------------
        enc1 = self.encoder1(x)  # [B, 32, H, W]
        enc1 = enc1 + t1  # add time embedding to first block

        enc2 = self.pool1(enc1)  # [B, 32, H/2, W/2]
        enc2 = self.encoder2(enc2)  # [B, 64, H/2, W/2]
        enc2 = enc2 + t2

        enc3 = self.pool2(enc2)  # [B, 64, H/4, W/4]
        enc3 = self.encoder3(enc3)  # [B, 128, H/4, W/4]
        enc3 = enc3 + t3

        enc4 = self.pool3(enc3)  # [B, 128, H/8, W/8]
        enc4 = self.encoder4(enc4)  # [B, 256, H/8, W/8]
        enc4 = enc4 + t4

        # -----------------
        #  BOTTLENECK
        # -----------------
        bottleneck = self.bottleneck(enc4)  # [B, 512, H/8, W/8]
        bottleneck = bottleneck + tb

        # -----------------
        #   DECODER
        # -----------------
        # Decode 4
        dec4 = self.upconv4(bottleneck)  # => [B, 256, H/4, W/4]
        dec4 = torch.cat((dec4, enc4), dim=1)  # => [B, 512, H/4, W/4]
        dec4 = self.decoder4(dec4)  # => [B, 256, H/4, W/4]

        # Decode 3
        dec3 = self.upconv3(dec4)  # => [B, 128, H/2, W/2]
        dec3 = torch.cat((dec3, enc3), dim=1)  # => [B, 256, H/2, W/2]
        dec3 = self.decoder3(dec3)  # => [B, 128, H/2, W/2]

        # Decode 2
        dec2 = self.upconv2(dec3)  # => [B, 64, H, W]
        dec2 = torch.cat((dec2, enc2), dim=1)  # => [B, 128, H, W]
        dec2 = self.decoder2(dec2)  # => [B, 64, H, W]

        # Decode 1
        dec1 = self.upconv1(dec2)  # => [B, 32, 2H, 2W] if the input was half that size, etc.
        dec1 = torch.cat((dec1, enc1), dim=1)  # => [B, 64, ...]
        dec1 = self.decoder1(dec1)  # => [B, 32, ...]

        # Final 1x1 conv
        out = self.conv(dec1)  # => [B, out_channels, ...]

        return out

    @staticmethod
    def _block(in_channels, features):
        """
        A simple two-conv block with BatchNorm + ReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
