import torch
import torch.nn as nn

VGG = {
    'vgg11' : [64 , 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13' : [64 , 64, 'M' , 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16' : [64 , 64, 'M' , 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19' : [64 , 64, 'M' , 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}

class VGGNet(nn.Module):
    def __init__(self , in_channels : int= 3 , num_classes : int= 10 ):
        super(VGGNet , self).__init__()
        self.in_channels = in_channels

        self.ConvLayers = self._create_layers(VGG['vgg11'])

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512*7*7 , out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096 , 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096 , num_classes)

        )

    def forward(self, x ):
        x = self.ConvLayers(x)
        x = self.FC(x)
        return x

    def _create_layers(self , architecture  ):
        layers = []
        in_channels = self.in_channels
        for x in architecture:

            if type(x) == int:
                out_channels = x
                layers.append([
                    nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=3 , stride=1 , padding=1),
                    nn.BatchNorm2d(x),
                    nn.LeakyReLU(0.1),
                    ])
                in_channels = x

            elif x == 'M':
                layers.append([
                    nn.MaxPool2d(kernel_size=2 , stride=2)
                ])

        return nn.Sequential(*layers)


x = torch.randn([32,3,224,224])
model= VGGNet()
print(model(x).shape)
