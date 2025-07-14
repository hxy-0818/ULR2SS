import torch
from torch import nn
from typing import List

disc_config = [
    (3, 64, 1),
    (3, 64, 2),
    (3, 128, 1),
    (3, 128, 2),
    (3, 256, 1),
    (3, 256, 2),
    (3, 512, 1),
    (3, 512, 2),
]

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_channels: int=3,
                 out_channels: int=64,
                 kernel_size: int=3,
                 stride: int=1,
                 padding: int=1,
                 act_fn: nn.Module=nn.Identity(),
                 **kwargs
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, **kwargs)
        self.act = act_fn

    def forward(self, x):
        return self.act(self.conv(x))

class DenseBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 channels: int=32, 
                 residual_coef: float=0.2
                 ):
        super().__init__()
        self.residual_coef = residual_coef
        self.blocks = nn.ModuleList()

        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    act_fn=nn.LeakyReLU(0.2, inplace=True) if i<4 else nn.Identity()
                )
            )

    def forward(self, x):
        h = x
        for block in self.blocks:
            out = block(h)
            h = torch.cat([h, out], dim=1)
        return self.residual_coef * out + x

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_coef=0.2):
        super().__init__()
        self.residual_coef = residual_coef
        self.rrdb = nn.Sequential(*[DenseBlock(in_channels) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_coef + x


class Generator(nn.Module):
    def __init__(self, 
                 in_channels: int=3, 
                 num_channels: int=64,
                 num_blocks: int=23
                 ):
        super().__init__()
        self.pre_block = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[RRDB(num_channels) for _ in range(num_blocks)])
        self.conv = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.upsample = self.make_upsample_blocks(num_channels, 2)
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, kernel_size=3, padding=1)
        )
    
    def make_upsample_blocks(self, channels: int=64, scale_factor: int=2):
        upsample_blocks = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return upsample_blocks
    
    def forward(self, x):
        pre = self.pre_block(x)
        out = self.res_blocks(pre)
        out = self.conv(out) + pre
        out = self.upsample(out)
        out = self.final(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels: int=3, disc_config: List=None):
        super().__init__()
        assert disc_config is not None, "must provide disc_config for discrimiator"
        
        self.blocks = self.make_layers(in_channels, disc_config)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
        )
    
    def make_layers(self, in_channels: int=3, disc_config: List=None):
        blocks = []
        for (kernel_size, out_channels, stride) in disc_config:
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    act_fn=nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.head(self.blocks(x))

if __name__ == "__main__":
    gen = Generator().to('cuda:0')
    # disc = Discriminator(disc_config=disc_config)
    x = torch.randn(160, 3, 96, 96).to('cuda:0')
    print(gen(x).shape)