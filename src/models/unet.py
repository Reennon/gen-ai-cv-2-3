import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddingUNet(nn.Module):
    """
    A UNet that injects a time embedding at each encoder stage and in the bottleneck.
    We use 4 down-sampling steps (enc1->enc4) plus a bottleneck.
    Each 'encX' is followed by a pooling layer, except for enc4 which has its own pool4,
    leading into the bottleneck. The decode path mirrors this structure.
    """

    def __init__(self, in_channels, out_channels, time_embedding_dim, init_features=32):
        super(TimeEmbeddingUNet, self).__init__()

        features = init_features

        # -------------------------
        #       ENCODER
        # -------------------------
        # Level 1
        self.encoder1 = self._block(in_channels, features)  # out: (B, 32, H,   W)
        self.pool1    = nn.MaxPool2d(kernel_size=2, stride=2)  # out: (B, 32, H/2, W/2)

        # Level 2
        self.encoder2 = self._block(features, features * 2)  # out: (B, 64, H/2,   W/2)
        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)  # out: (B, 64, H/4,   W/4)

        # Level 3
        self.encoder3 = self._block(features * 2, features * 4)  # out: (B, 128, H/4, W/4)
        self.pool3    = nn.MaxPool2d(kernel_size=2, stride=2)     # out: (B, 128, H/8, W/8)

        # Level 4
        self.encoder4 = self._block(features * 4, features * 8)   # out: (B, 256, H/8, W/8)
        self.pool4    = nn.MaxPool2d(kernel_size=2, stride=2)     # out: (B, 256, H/16, W/16)

        # -------------------------
        #     BOTTLENECK
        # -------------------------
        # Level 5
        self.bottleneck = self._block(features * 8, features * 16) # out: (B, 512, H/16, W/16)

        # -------------------------
        #       DECODER
        # -------------------------
        # upconv4: 512 -> 256, then cat with encoder4 => 512 -> 256
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False)
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)  # cat => 512 -> 256

        # upconv3: 256 -> 128, cat with encoder3 => 256 -> 128
        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False)
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)  # cat => 256 -> 128

        # upconv2: 128 -> 64, cat with encoder2 => 128 -> 64
        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1, bias=False)
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)  # cat => 128 -> 64

        # upconv1: 64 -> 32, cat with encoder1 => 64 -> 32
        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False)
        )
        self.decoder1 = self._block(features * 2, features)          # cat => 64 -> 32

        # final 1×1 conv
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # -------------------------
        #  TIME EMBEDDING PROJECTIONS
        # -------------------------
        # Match each encoder block + bottleneck channel dimension
        self.time_proj1 = nn.Linear(time_embedding_dim, features)         # 32
        self.time_proj2 = nn.Linear(time_embedding_dim, features * 2)     # 64
        self.time_proj3 = nn.Linear(time_embedding_dim, features * 4)     # 128
        self.time_proj4 = nn.Linear(time_embedding_dim, features * 8)     # 256
        self.time_proj_bottleneck = nn.Linear(time_embedding_dim, features * 16)  # 512

    def forward(self, x, time_emb):
        """
        x       : [B, in_channels, H, W]
        time_emb: [B, time_embedding_dim]
        """
        B = x.size(0)

        # project time embeddings to match each stage
        t1 = self.time_proj1(time_emb).view(B, -1, 1, 1)         # [B, 32, 1, 1]
        t2 = self.time_proj2(time_emb).view(B, -1, 1, 1)         # [B, 64, 1, 1]
        t3 = self.time_proj3(time_emb).view(B, -1, 1, 1)         # [B, 128, 1, 1]
        t4 = self.time_proj4(time_emb).view(B, -1, 1, 1)         # [B, 256, 1, 1]
        tb = self.time_proj_bottleneck(time_emb).view(B, -1, 1, 1)  # [B, 512, 1, 1]

        # -------------------------
        #   ENCODER
        # -------------------------
        # Level 1
        enc1 = self.encoder1(x)              # [B, 32,  H,   W]
        enc1 = enc1 + t1
        p1   = self.pool1(enc1)             # [B, 32,  H/2, W/2]

        # Level 2
        enc2 = self.encoder2(p1)            # [B, 64,  H/2, W/2]
        enc2 = enc2 + t2
        p2   = self.pool2(enc2)             # [B, 64,  H/4, W/4]

        # Level 3
        enc3 = self.encoder3(p2)            # [B, 128, H/4, W/4]
        enc3 = enc3 + t3
        p3   = self.pool3(enc3)             # [B, 128, H/8, W/8]

        # Level 4
        enc4 = self.encoder4(p3)            # [B, 256, H/8, W/8]
        enc4 = enc4 + t4
        p4   = self.pool4(enc4)             # [B, 256, H/16, W/16]

        # -------------------------
        #   BOTTLENECK
        # -------------------------
        bottleneck = self.bottleneck(p4)    # [B, 512, H/16, W/16]
        bottleneck = bottleneck + tb

        # -------------------------
        #   DECODER
        # -------------------------
        # decode4
        dec4 = self.upconv4(bottleneck)                # [B, 256, H/8,  W/8]
        dec4 = torch.cat((dec4, enc4), dim=1)          # [B, 512, H/8,  W/8]
        dec4 = self.decoder4(dec4)                     # [B, 256, H/8,  W/8]

        # decode3
        dec3 = self.upconv3(dec4)                      # [B, 128, H/4, W/4]
        dec3 = torch.cat((dec3, enc3), dim=1)          # [B, 256, H/4, W/4]
        dec3 = self.decoder3(dec3)                     # [B, 128, H/4, W/4]

        # decode2
        dec2 = self.upconv2(dec3)                      # [B, 64,  H/2, W/2]
        dec2 = torch.cat((dec2, enc2), dim=1)          # [B, 128, H/2, W/2]
        dec2 = self.decoder2(dec2)                     # [B, 64,  H/2, W/2]

        # decode1
        dec1 = self.upconv1(dec2)                      # [B, 32, H,   W]
        dec1 = torch.cat((dec1, enc1), dim=1)          # [B, 64, H,   W]
        dec1 = self.decoder1(dec1)                     # [B, 32, H,   W]

        # final output
        out = self.conv(dec1)                          # [B, out_channels, H, W]
        return out

    @staticmethod
    def _block(in_channels, features):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )
