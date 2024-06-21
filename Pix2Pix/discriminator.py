import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvBlock,self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=stride,padding=1,padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True),
        )
    def forward(self, x ):
        return self.Layer(x)
    
class Discriminator(nn.Module):
    def __init__(self , in_channels = 3):
        super(Discriminator,self).__init__()

        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2 , out_channels=64 , kernel_size=4 , stride=2 , padding=1 , padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.ConvLayer2 = ConvBlock(in_channels=64 , out_channels=128)
        self.ConvLayer3 = ConvBlock(in_channels=128 , out_channels=256)
        self.ConvLayer4 = ConvBlock(in_channels=256 , out_channels=512 , stride=1)
        self.ConvLayer5 = nn.Sequential(
            nn.Conv2d(in_channels=512 , out_channels=1 , kernel_size=4 , stride=1 , padding=1 , padding_mode="reflect")
        )

    def forward(self, x, y):
        x = torch.cat([x ,y], dim=1)
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.ConvLayer5(x)
        return x

