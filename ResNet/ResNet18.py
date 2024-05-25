import torch 
import torch.nn as nn


class ResNet18(nn.Module):
    def __init__(self , num_classes ):
        super(ResNet18 , self).__init__()

        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=64 , kernel_size=(7 , 7) , stride=2 , padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2,2),
        )      
                 
        self.ResBlock1 = self._create_residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)
        self.ResBlock2 = self._create_residualblock(in_channels=64 , out_channels=64 , stride=1 , padding=1)
        
        self.ConvBlock2 = self._create_basicblock(in_channels=64 , out_channels=128 , kernel=3 , padding=1 , stride=1)
        self.ResBlock3 = self._create_residualblock(in_channels=128 , out_channels=128 , stride=1 , padding=1)

        self.ConvBlock3 = self._create_basicblock(in_channels=128 , out_channels=256 , kernel=3 , padding=1 , stride=1)
        self.ResBlock4 = self._create_residualblock(in_channels=256 , out_channels=256 , stride=1 , padding=1)

        self.ConvBlock4 = self._create_basicblock(in_channels=256 , out_channels=512 , kernel=3 , padding=1 , stride=1)
        self.ResBlock5 = self._create_residualblock(in_channels=512 , out_channels=512 , stride=1 , padding=1)  

        self.avgpool = nn.AvgPool2d(kernel_size=7 , stride=7)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=512 , out_features=self.num_classes),
        )

    def forward(self, x):
        out = self.ConvBlock1(x)
        out = self.ResBlock1(out) + out
        out = self.ResBlock2(out) + out
        out = self.ConvBlock2(out)
        out = self.ResBlock3(out) + out
        out = self.ConvBlock3(out)
        out = self.ConvBlock4(out)
        out = self.ResBlock5(out) + out
        out = self.avgpool(out)
        out = self.FC(out)
        return out
    
    def _create_residualblock(self,in_channels , out_channels , stride , padding ):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=(3,3) , padding=padding , stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            )
        

    def _create_basicblock(self,in_channels , out_channels , kernel,  stride , padding ):
        return nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel , padding=stride , stride =padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(kernel_size=2 , stride=2),
            )


