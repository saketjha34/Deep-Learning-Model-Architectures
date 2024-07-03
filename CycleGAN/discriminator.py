import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels,stride = 2,use_instancenorm = True):
        super(ConvBlock,self).__init__()   
        self.use_instancenorm = use_instancenorm
        self.Conv2d = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=stride,padding=1,bias=True,padding_mode="reflect",)
        self.InstanceNorm = nn.InstanceNorm2d(out_channels)
        self.LeakyRelu = nn.LeakyReLU(0.2 , inplace=True)

    def forward(self,x):
        x = self.Conv2d(x)
        if self.use_instancenorm == True:
            x = self.InstanceNorm(x)
        x = self.LeakyRelu(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self,in_channels = 3):
        super(Discriminator,self).__init__()

        self.Layer1 = ConvBlock(in_channels=in_channels,out_channels=64,stride=2,use_instancenorm=False)
        self.Layer2 = ConvBlock(in_channels=64,out_channels=128,stride=2,use_instancenorm=True)
        self.Layer3 = ConvBlock(in_channels=128,out_channels=256,stride=2,use_instancenorm=True)
        self.Layer4 = ConvBlock(in_channels=256,out_channels=512,stride=1,use_instancenorm=True)
        self.Layer5 = nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=1,padding=1,padding_mode="reflect",)
        self.Sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)
        x = self.Sigmoid(x)
        return x