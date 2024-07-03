import torch
import torch.nn as nn

class DownConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,**kwargs):
        super(DownConvBlock,self).__init__()

        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,padding_mode="reflect",**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.Layer(x)
    
class UpConvBlock(nn.Module):
    def __init__(self,in_channels , out_channels , **kwargs):
        super(UpConvBlock,self).__init__()

        self.Layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.Layer(x)      
    
class ResidualBlock(nn.Module):
    def __init__(self,in_channels):
        super(ResidualBlock,self).__init__()

        self.ResLayer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,padding_mode="reflect",kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,padding_mode="reflect",kernel_size=3,padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.Identity(),
        )

    def forward(self,x):
        return x + self.ResLayer(x)
    
class Generator(nn.Module):
    def __init__(self,img_channels=3):
        super(Generator,self).__init__()

        self.DownBlock1 = DownConvBlock(in_channels=img_channels,out_channels=64,kernel_size=7,stride=1,padding=3)
        self.DownBlock2 = DownConvBlock(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1)
        self.DownBlock3 = DownConvBlock(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1)

        self.ResBlock1 = ResidualBlock(in_channels=256)
        self.ResBlock2 = ResidualBlock(in_channels=256)
        self.ResBlock3 = ResidualBlock(in_channels=256)
        self.ResBlock4 = ResidualBlock(in_channels=256)
        self.ResBlock5 = ResidualBlock(in_channels=256)
        self.ResBlock6 = ResidualBlock(in_channels=256)
        self.ResBlock7 = ResidualBlock(in_channels=256)
        self.ResBlock8 = ResidualBlock(in_channels=256)
        self.ResBlock9 = ResidualBlock(in_channels=256)

        self.UpBlock1 = UpConvBlock(in_channels=256,out_channels=128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.UpBlock2 = UpConvBlock(in_channels=128,out_channels=64,kernel_size=3,stride=2,padding=1,output_padding=1)
        
        self.BottleNeck = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=img_channels,kernel_size=7,stride=1,padding=3,padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.DownBlock1(x)
        x = self.DownBlock2(x)
        x = self.DownBlock3(x)
        x = self.ResBlock1(x)
        x = self.ResBlock2(x)
        x = self.ResBlock3(x)
        x = self.ResBlock4(x)
        x = self.ResBlock5(x)
        x = self.ResBlock6(x)
        x = self.ResBlock7(x)
        x = self.ResBlock8(x)
        x = self.ResBlock9(x)
        x = self.UpBlock1(x)
        x = self.UpBlock2(x)
        x = self.BottleNeck(x)
        return x