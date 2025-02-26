import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddingUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, init_features=32):
        super().__init__()
        f = init_features

        # -------------------------
        #       Encoder
        # -------------------------
        self.encoder1 = self._block(in_channels, f)      # => [B, f,   H,   W]
        self.pool1    = nn.MaxPool2d(2)                  # => [B, f,   H/2, W/2]

        self.encoder2 = self._block(f, f * 2)            # => [B, 2f,  H/2, W/2]
        self.pool2    = nn.MaxPool2d(2)                  # => [B, 2f,  H/4, W/4]

        self.encoder3 = self._block(f * 2, f * 4)        # => [B, 4f,  H/4, W/4]
        self.pool3    = nn.MaxPool2d(2)                  # => [B, 4f,  H/8, W/8]

        self.encoder4 = self._block(f * 4, f * 8)        # => [B, 8f,  H/8, W/8]
        self.pool4    = nn.MaxPool2d(2)                  # => [B, 8f,  H/16, W/16]

        # -------------------------
        #     Bottleneck
        # -------------------------
        self.bottleneck = self._block(f * 8, f * 16)     # => [B, 16f, H/16, W/16]

        # -------------------------
        #       Decoder
        # -------------------------
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 16, f * 8, kernel_size=3, padding=1, bias=False),
        )
        self.decoder4 = self._block((f * 8) * 2, f * 8)  # cat => [B, 16f, H/8, W/8] -> [B, 8f, H/8, W/8]

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 8, f * 4, kernel_size=3, padding=1, bias=False),
        )
        self.decoder3 = self._block((f * 4) * 2, f * 4)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 4, f * 2, kernel_size=3, padding=1, bias=False),
        )
        self.decoder2 = self._block((f * 2) * 2, f * 2)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(f * 2, f, kernel_size=3, padding=1, bias=False),
        )
        self.decoder1 = self._block(f * 2, f)

        self.conv = nn.Conv2d(f, out_channels, kernel_size=1)

        # Time embeddings for each level
        self.time_proj1 = nn.Linear(time_embedding_dim, f)
        self.time_proj2 = nn.Linear(time_embedding_dim, f * 2)
        self.time_proj3 = nn.Linear(time_embedding_dim, f * 4)
        self.time_proj4 = nn.Linear(time_embedding_dim, f * 8)
        self.time_proj_bottleneck = nn.Linear(time_embedding_dim, f * 16)

    @staticmethod
    def _block(in_ch, out_ch):
        """A double-conv block with BatchNorm + ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )

    def forward(self, x, t_emb):
        B = x.size(0)

        # 1) Project time embeddings to each stage
        t1 = self.time_proj1(t_emb).view(B, -1, 1, 1)
        t2 = self.time_proj2(t_emb).view(B, -1, 1, 1)
        t3 = self.time_proj3(t_emb).view(B, -1, 1, 1)
        t4 = self.time_proj4(t_emb).view(B, -1, 1, 1)
        tb = self.time_proj_bottleneck(t_emb).view(B, -1, 1, 1)

        # -------------------------
        #   ENCODER
        # -------------------------
        e1 = self.encoder1(x)       # => [B, f,   H,   W]
        e1 = e1 + t1
        p1 = self.pool1(e1)         # => [B, f,   H/2, W/2]

        e2 = self.encoder2(p1)      # => [B, 2f,  H/2, W/2]
        e2 = e2 + t2
        p2 = self.pool2(e2)         # => [B, 2f,  H/4, W/4]

        e3 = self.encoder3(p2)      # => [B, 4f,  H/4, W/4]
        e3 = e3 + t3
        p3 = self.pool3(e3)         # => [B, 4f,  H/8, W/8]

        e4 = self.encoder4(p3)      # => [B, 8f,  H/8, W/8]
        e4 = e4 + t4
        p4 = self.pool4(e4)         # => [B, 8f,  H/16, W/16]

        # -------------------------
        #  BOTTLENECK
        # -------------------------
        bottleneck = self.bottleneck(p4)  # => [B, 16f, H/16, W/16]
        bottleneck = bottleneck + tb

        # -------------------------
        #   DECODER
        # -------------------------
        d4 = self.upconv4(bottleneck)          # => [B, 8f,  H/8, W/8]
        d4 = torch.cat([d4, e4], dim=1)        # => [B, 16f, H/8, W/8]
        d4 = self.decoder4(d4)                 # => [B, 8f,  H/8, W/8]

        d3 = self.upconv3(d4)                  # => [B, 4f,  H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)        # => [B, 8f,  H/4, W/4]
        d3 = self.decoder3(d3)                 # => [B, 4f,  H/4, W/4]

        d2 = self.upconv2(d3)                  # => [B, 2f,  H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)        # => [B, 4f,  H/2, W/2]
        d2 = self.decoder2(d2)                 # => [B, 2f,  H/2, W/2]

        d1 = self.upconv1(d2)                  # => [B, f,   H,   W]
        d1 = torch.cat([d1, e1], dim=1)        # => [B, 2f,  H,   W]
        d1 = self.decoder1(d1)                 # => [B, f,   H,   W]

        out = self.conv(d1)                    # => [B, out_channels, H, W]
        return out
