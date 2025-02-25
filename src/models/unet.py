import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbeddingUNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, init_features=32):
        super(TimeEmbeddingUNet, self).__init__()
        # Removed self.time_mlp and SinusoidalPositionEmbeddings usage

        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self._block(features * 8, features * 16)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x, time_emb):
        # Expecting time_emb of shape [B, time_embedding_dim].
        # Reshape to [B, time_embedding_dim, 1, 1] for broadcasting.
        t = time_emb.view(time_emb.size(0), time_emb.size(1), 1, 1)

        # Now add the time embedding to feature maps.
        enc1 = self.encoder1(x + t)
        enc2 = self.encoder2(self.pool1(enc1) + t)
        enc3 = self.encoder3(self.pool2(enc2) + t)
        enc4 = self.encoder4(self.pool3(enc3) + t)
        bottleneck = self.bottleneck(self.pool4(enc4) + t)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4 + t)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3 + t)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2 + t)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1 + t)
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
