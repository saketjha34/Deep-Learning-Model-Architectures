import torch.nn as nn
import torch


class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,discriminator=False,use_act=True,use_bn=True,**kwargs):
        super(ConvLayer,self).__init__()
        self.use_act =use_act
        self.use_bn = use_bn
        self.conv2d = nn.Conv2d(in_channels,out_channels,bias=not use_bn,**kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if self.use_bn else nn.Identity()
        self.act = nn.LeakyReLU(0.2,inplace=True) if discriminator else nn.PReLU(out_channels)

    def forward(self,x):
        return self.act(self.bn(self.conv2d(x))) if self.use_act else self.bn(self.conv2d(x))


class UpSampleBlock(nn.Module):
    def __init__(self,in_channels,scale_factor):
        super(UpSampleBlock,self).__init__()
        self.conv2d = nn.Conv2d(in_channels,in_channels*scale_factor**2,kernel_size=3,padding=1,stride=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(in_channels)

    def forward(self,x):
        return self.act(self.pixel_shuffle(self.conv2d(x)))
    

class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            ConvLayer(in_channels,in_channels,kernel_size=3,stride=1,padding=1),
            ConvLayer(in_channels,in_channels,kernel_size=3,stride=1,padding=1,use_act=False)
        )

    def forward(self,x):
        out =  self.block(x)
        return out + x


class Generator(nn.Module):
    def __init__(self,in_channels=3,num_residual_blocks=16,scale_factor=2):
        super(Generator,self).__init__()
        self.initial_block =  ConvLayer(in_channels,out_channels=64,kernel_size=9,stride=1,padding=4,use_bn = False)
        self.residual_blocks = nn.Sequential(*[ResidualBlock(in_channels=64) for _ in range(num_residual_blocks)])
        self.bottleneck = ConvLayer(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1,use_act=False)
        self.upsample_blocks = nn.Sequential(
            UpSampleBlock(in_channels=64,scale_factor=scale_factor),
            UpSampleBlock(in_channels=64,scale_factor=scale_factor),
        )
        self.final_block = ConvLayer(in_channels=64,out_channels=3,kernel_size=9,stride=1,padding=4,use_act=False,use_bn=False)

    def forward(self,x):
        initial = self.initial_block(x)
        x = self.residual_blocks(initial)
        x = self.bottleneck(x) + initial
        out = self.upsample_blocks(x)
        return torch.tanh(self.final_block(out))