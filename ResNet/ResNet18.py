import torch 
import torch.nn as nn

def residualblock(in_channels , out_channels , stride , padding ):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride),
        nn.LeakyReLU(0.1),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride),
        nn.LeakyReLU(0.1),
        nn.BatchNorm2d(out_channels)
        )
    return block

def basicblock(in_channels , out_channels , kernel,  stride , padding ):
    block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel , padding=stride , stride =padding),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2 , stride=2),
            nn.BatchNorm2d(out_channels),
        )
    return block


class ResNet18(nn.Module):
    def __init__(self , num_classes ):
        super(ResNet18 , self).__init__()

        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=(7 , 7) , stride=2 , padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(64),
        )      
                 
        self.ResBlock1 = residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)
        self.ResBlock2 = residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)
        self.ConvBlock2 = basicblock(in_channels=64 , out_channels=128 , kernel=3 , padding=1 , stride=1)

        self.ResBlock3 = residualblock(in_channels=128 , out_channels=128 , stride=1 , padding=1)
        self.ResBlock4 = residualblock(in_channels=128 , out_channels=128 , stride=1 , padding=1)
        self.ConvBlock3 = basicblock(in_channels=128 , out_channels=256 , kernel=3 , padding=1 , stride=1)

        self.ResBlock5 = residualblock(in_channels=256 , out_channels=256 , stride=1 , padding=1)
        self.ResBlock6 = residualblock(in_channels=256 , out_channels=256 , stride=1 , padding=1)
        self.ConvBlock4 = basicblock(in_channels=256 , out_channels=512 , kernel=3 , padding=1 , stride=1)

        self.avgpool = nn.AvgPool2d(kernel_size=7 , stride=7)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=512 , out_features=num_classes),
        )

    def forward(self, x):
        out = self.ConvBlock1(x)
        out = self.ResBlock1(out) + out
        out = self.ResBlock2(out) + out
        out = self.ConvBlock2(out)
        out = self.ResBlock3(out) + out
        out = self.ResBlock4(out) + out
        out = self.ConvBlock3(out)
        out = self.ResBlock5(out) + out
        out = self.ResBlock6(out) + out
        out = self.ConvBlock4(out)
        out = self.avgpool(out)
        out = self.FC(out)
        return out


