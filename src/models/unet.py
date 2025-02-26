import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddingUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, init_features=32):
        super(TimeEmbeddingUNet, self).__init__()

        features = init_features
        # Encoder layers
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64 -> 32
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32 -> 16
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16 -> 8
        self.encoder4 = self._block(features * 4, features * 8)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)

        # Decoder layers with consistent upsampling
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False)
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False)
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1, bias=False)
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)

        self.upconv1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False)
        )
        self.decoder1 = self._block(features * 2, features)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

        # Time Embeddings
        self.time_proj1 = nn.Linear(time_embedding_dim, 64)
        self.time_proj2 = nn.Linear(time_embedding_dim, 128)
        self.time_proj3 = nn.Linear(time_embedding_dim, 256)
        self.time_proj4 = nn.Linear(time_embedding_dim, 512)
        self.time_proj_bottleneck = nn.Linear(time_embedding_dim, 1024)

    def forward(self, x, time_emb):
        B = x.size(0)

        # Project the time embedding to match each block's channel dimension:
        t1 = self.time_proj1(time_emb).view(B, 64, 1, 1)
        t2 = self.time_proj2(time_emb).view(B, 128, 1, 1)
        t3 = self.time_proj3(time_emb).view(B, 256, 1, 1)
        t4 = self.time_proj4(time_emb).view(B, 512, 1, 1)
        tb = self.time_proj_bottleneck(time_emb).view(B, 1024, 1, 1)

        # Encoder path
        enc1 = self.encoder1(x + t1)
        enc2 = self.encoder2(self.pool1(enc1)) + t2
        enc3 = self.encoder3(self.pool2(enc2)) + t3
        enc4 = self.encoder4(self.pool3(enc3)) + t4

        # Bottleneck
        bottleneck = self.bottleneck(enc4) + tb

        # Decoder path
        dec4 = self.upconv4(bottleneck)
        if enc4.size(2) != dec4.size(2) or enc4.size(3) != dec4.size(3):
            enc4 = F.interpolate(enc4, size=dec4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if enc3.size(2) != dec3.size(2) or enc3.size(3) != dec3.size(3):
            enc3 = F.interpolate(enc3, size=dec3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if enc2.size(2) != dec2.size(2) or enc2.size(3) != dec2.size(3):
            enc2 = F.interpolate(enc2, size=dec2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if enc1.size(2) != dec1.size(2) or enc1.size(3) != dec1.size(3):
            enc1 = F.interpolate(enc1, size=dec1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        print(f"[UNet] Input Shape: {x.shape}")
        print(f"[UNet] After Encoder 1: {enc1.shape}")
        print(f"[UNet] After Encoder 2: {enc2.shape}")
        print(f"[UNet] After Encoder 3: {enc3.shape}")
        print(f"[UNet] After Encoder 4: {enc4.shape}")
        print(f"[UNet] Bottleneck: {bottleneck.shape}")

        print(f"[UNet] After UpConv4: {dec4.shape}")
        print(f"[UNet] Before Cat with Enc4: {enc4.shape}, {dec4.shape}")
        print(f"[UNet] After Decoder 4: {dec4.shape}")

        print(f"[UNet] After UpConv3: {dec3.shape}")
        print(f"[UNet] Before Cat with Enc3: {enc3.shape}, {dec3.shape}")
        print(f"[UNet] After Decoder 3: {dec3.shape}")

        print(f"[UNet] After UpConv2: {dec2.shape}")
        print(f"[UNet] Before Cat with Enc2: {enc2.shape}, {dec2.shape}")
        print(f"[UNet] After Decoder 2: {dec2.shape}")

        print(f"[UNet] After UpConv1: {dec1.shape}")
        print(f"[UNet] Before Cat with Enc1: {enc1.shape}, {dec1.shape}")
        print(f"[UNet] After Decoder 1: {dec1.shape}")

        print(f"[UNet] Final Output: {self.conv(dec1).shape}")

        return self.conv(dec1)

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
