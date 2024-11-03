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


class Discriminator(nn.Module):
    def __init__(self,in_channels=3,):

        super(Discriminator,self).__init__()
        self.initial_block = ConvLayer(in_channels,out_channels=64,discriminator=True,kernel_size=3,stride=1,padding=1,use_bn=False)
        
        self.blocks = nn.Sequential(
            ConvLayer(in_channels=64,out_channels=64,discriminator=True,kernel_size=3,stride=1,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=64,out_channels=128,discriminator=True,kernel_size=3,stride=2,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=128,out_channels=128,discriminator=True,kernel_size=3,stride=1,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=128,out_channels=256,discriminator=True,kernel_size=3,stride=2,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=256,out_channels=256,discriminator=True,kernel_size=3,stride=1,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=256,out_channels=512,discriminator=True,kernel_size=3,stride=2,padding=1,use_act=True,use_bn=True),
            ConvLayer(in_channels=512,out_channels=512,discriminator=True,kernel_size=3,stride=1,padding=1,use_act=True,use_bn=True),
        )

        self.final_block = nn.Sequential(
                nn.AdaptiveAvgPool2d((6,6)),
                nn.Flatten(),
                nn.Linear(512*6*6,1024),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Linear(1024,1)
            )
        
    def forward(self,x):
        x = self.initial_block(x)
        x = self.blocks(x)
        return self.final_block(x)

