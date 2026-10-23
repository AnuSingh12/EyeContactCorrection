
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    """Depthwise-separable convolution block with residual connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        x = self.depthwise(x)
        x = self.relu(self.bn1(x))
        x = self.pointwise(x)
        x = self.bn2(x)

        return self.relu(x + residual)



class ECCNet(nn.Module):
    """
    ECC-Net-inspired model.
    Input image: [B, 3, 64, 32]
    Target angle: [B, 2]
    Outputs: vector field [B, 2, 64, 32],
             brightness map [B, 1, 64, 32]
    """

    def __init__(self):
        super().__init__()

        self.enc1 = SeparableConvBlock(5, 32)
        self.down1 = SeparableConvBlock(32, 32, stride=2)

        self.enc2 = SeparableConvBlock(32, 64)
        self.down2 = SeparableConvBlock(64, 64, stride=2)

        self.enc3 = SeparableConvBlock(64, 128)
        self.down3 = SeparableConvBlock(128, 128, stride=2)

        self.enc4 = SeparableConvBlock(128, 256)
        self.down4 = SeparableConvBlock(256, 256, stride=2)

        
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = SeparableConvBlock(128 + 256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = SeparableConvBlock(64 + 128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = SeparableConvBlock(32 + 64, 32)

        self.up0 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec0 = SeparableConvBlock(32 + 32, 32)

        self.output_head = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, eye_patch, target_angle):
        if eye_patch.ndim != 4 or eye_patch.shape[1:] != (3, 64, 32):
            raise ValueError("eye_patch must have shape [B, 3, 64, 32]")

        if target_angle.shape != (eye_patch.shape[0], 2):
            raise ValueError("target_angle must have shape [B, 2]")

        angle_map = target_angle[:, :, None, None].expand(
            -1, -1, eye_patch.shape[2], eye_patch.shape[3]
        )
        x = torch.cat([eye_patch, angle_map], dim=1)

        # Encoder
        s1 = self.enc1(x)                 
        s2 = self.enc2(self.down1(s1))     
        s3 = self.enc3(self.down2(s2))     
        s4 = self.enc4(self.down3(s3))     
        x = self.down4(s4)               

        # Decoder + skip connections
        x = self.up3(x)
        x = self.dec3(torch.cat([x, s4], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, s3], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, s2], dim=1))

        x = self.up0(x)
        x = self.dec0(torch.cat([x, s1], dim=1))

        outputs = self.output_head(x)
        vector_field = outputs[:, :2]
        brightness_map = torch.sigmoid(outputs[:, 2:3])

        return vector_field, brightness_map


if __name__ == "__main__":
    model = ECCNet()
    eye = torch.randn(2, 3, 64, 32)
    angle = torch.zeros(2, 2)

    vector_field, brightness_map = model(eye, angle)

    print("Vector field:", vector_field.shape)
    print("Brightness map:", brightness_map.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
