import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_maxpool=False):
        super().__init__()
        self.use_maxpool = use_maxpool
        
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) if self.use_maxpool == True else nn.Identity()
        )
        
    def forward(self,x):
        return self.convblock(x)
    
class YoloV1(nn.Module):
    def __init__(self, in_channels=3, S=7 , num_boxes=2 ,num_classes=20):
        super().__init__()
        
        self.ConvLayers1 = nn.Sequential(
            ConvLayer(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, use_maxpool=True),
            ConvLayer(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1, use_maxpool=True),
            ConvLayer(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, use_maxpool=True),
        )
        
        self.ConveLayer2 = nn.Sequential(
            ConvLayer(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, use_maxpool=True),
        )
            
        self.ConveLayer3 = nn.Sequential(
            ConvLayer(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            ConvLayer(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
            ConvLayer(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            ConvLayer(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(496, S* S* (num_classes + num_boxes*5))
        )
            
    def forward(self, x):
        x = self.ConvLayers1(x)
        x = self.ConveLayer2(x)
        x = self.ConveLayer3(x)
        x = self.FC(x)
        return x

if __name__ == "__main__":
    x = torch.randn([8,3,448,448])
    print(x.shape)
    model = YoloV1()
    print(model(x).shape)