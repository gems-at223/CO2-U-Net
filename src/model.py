import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention layer for U-Net"""

    def __init__(self, in_channels: int) -> torch.Tensor:
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.down1 = self.contract_block(in_channels, 64)
        self.attention1 = SelfAttention(64)
        self.down2 = self.contract_block(64, 128)
        self.attention2 = SelfAttention(128)
        self.down3 = self.contract_block(128, 256)
        self.attention3 = SelfAttention(256)
        self.down4 = self.contract_block(256, 512)
        self.attention4 = SelfAttention(512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = self.expand_block(1536, 512)  # 1024 from bottleneck + 512 from down4
        self.up2 = self.expand_block(768, 256)  # 512 from up1 + 256 from down3
        self.up3 = self.expand_block(384, 128)  # 256 from up2 + 128 from down2
        self.up4 = self.expand_block(192, 64)  # 128 from up3 + 64 from down1

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.down1(x)
        x1 = self.attention1(x1)
        x2 = self.down2(x1)
        x2 = self.attention2(x2)
        x3 = self.down3(x2)
        x3 = self.attention3(x3)
        x4 = self.down4(x3)
        x4 = self.attention4(x4)

        bottleneck = self.bottleneck(x4)

        x = self.concat_and_expand(self.up1, bottleneck, x4)
        x = self.concat_and_expand(self.up2, x, x3)
        x = self.concat_and_expand(self.up3, x, x2)
        x = self.concat_and_expand(self.up4, x, x1)

        out = self.final(x)
        out = out.permute(0, 2, 3, 1)
        return out

    def contract_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        return block

    def expand_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )
        return block

    def concat_and_expand(
        self, block: nn.Sequential, up_input: torch.Tensor, skip_input: torch.Tensor
    ) -> torch.Tensor:
        upsampled = F.interpolate(up_input, size=skip_input.shape[2:], mode="nearest")
        concatenated = torch.cat([upsampled, skip_input], dim=1)
        return block(concatenated)
