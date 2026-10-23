
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Three depthwise-separable convolution layers with residual links."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, 3,
                padding=1, groups=in_channels, bias=False
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, 3,
                padding=1, groups=out_channels, bias=False
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, 3,
                padding=1, groups=out_channels, bias=False
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        residual = self.residual(x)

        x = self.conv1(x)
        middle = x

        x = F.relu(self.conv2(x) + middle)
        x = self.conv3(x)

        return F.relu(x + residual)


class ECCNet(nn.Module):
    """
    ECC-Net-style encoder-decoder.

    Image:         [B, 1, 32, 64]
    Target angle:  [B, 2]
    Vector field:  [B, 2, 32, 64]
    Brightness:    [B, 1, 32, 64]
    """

    def __init__(self):
        super().__init__()

        # Four encoder stages.
        # Image + two tiled target-angle channels = 3 input channels.
        self.enc1 = DepthwiseSeparableConv(5, 32)
        self.enc2 = DepthwiseSeparableConv(32, 64)
        self.enc3 = DepthwiseSeparableConv(64, 128)
        self.enc4 = DepthwiseSeparableConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck after the fourth downsampling.
        self.bottleneck = DepthwiseSeparableConv(256, 256)

        # Four decoder up-convolution stages with skip connections.
        self.up4 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec4 = DepthwiseSeparableConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DepthwiseSeparableConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DepthwiseSeparableConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DepthwiseSeparableConv(64, 32)

        # Two vector components and one brightness channel.
        self.output_conv = nn.Conv2d(
            32, 3, kernel_size=3, padding=1
        )

    def forward(self, image, target_angle):
        batch_size, _, height, width = image.shape

        if height != 32 or width != 64:
            raise ValueError(
                f"Expected image shape [B, 1, 32, 64], "
                f"got {tuple(image.shape)}"
            )

        if target_angle.shape != (batch_size, 2):
            raise ValueError(
                f"Expected target_angle shape {(batch_size, 2)}, "
                f"got {tuple(target_angle.shape)}"
            )

        if image.ndim != 4 or image.shape[1:] != (3, 32, 64):
            raise ValueError(
                f"Expected image shape [B, 3, 32, 64], "
                f"got {tuple(image.shape)}"
            )

        # Tile the horizontal and vertical target-angle components.
        angle_map = target_angle[:, :, None, None].expand(
            batch_size, 2, height, width
        )

        x = torch.cat([image, angle_map], dim=1)

        # Encoder: 32x64 -> 16x32 -> 8x16 -> 4x8 -> 2x4.
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))

        # Decoder: 2x4 -> 4x8 -> 8x16 -> 16x32 -> 32x64.
        d4 = self.up4(bottleneck)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        output = self.output_conv(d1)

        vector_field = output[:, :2, :, :]
        brightness_map = torch.sigmoid(output[:, 2:3, :, :])

        return vector_field, brightness_map
