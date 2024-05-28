import torch
import torch.nn as nn

#Alexnet input image size - 3 x 227 x 227
class AlexNet227(nn.Module):
    def __init__(self , num_classes = 1000) :
        super().__init__()
        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=96 , kernel_size=(11,11) , padding=0, stride=4 , bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3) , stride = 2),
            nn.BatchNorm2d(96)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=(5,5) , padding=2 , stride=1, bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3) , stride=2),
            nn.BatchNorm2d(256)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=(3,3) , stride=1, padding=1),
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384 , kernel_size=(3,3) , padding=1 , stride = 1),
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=(3,3) , padding=1 ,stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3) ,stride=2)
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256*6*6 , out_features=4096),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096 , out_features=4096),
            nn.LeakyReLU(0.3 , inplace=True),
            nn.Linear(in_features=4096 , out_features=self.num_classes)
        )

    def forward(self,x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.FC(x)
        return x    
    
#Alexnet input image size - 3 x 224 x 224
class AlexNet224(nn.Module):
    def __init__(self,num_classes = 1000):
        super(AlexNet224, self).__init__()
        self.num_classes = num_classes

        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=96 , kernel_size=(11,11) , padding=1 , stride=4, bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(2,2)  ,stride=2),
            nn.BatchNorm2d(96)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=(7,7) , padding=3 , stride=1, bias = False),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3)  ,stride=2),
            nn.BatchNorm2d(256)
        )

        self.ConvBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=(5,5) , padding=2 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.BatchNorm2d(384)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels= 384 , kernel_size=(3,3) , padding=1 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.ConvBlock5 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=(3,3) , padding=1 , stride=1),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.MaxPool2d(kernel_size=(3,3)  ,stride=2),
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256*6*6 , out_features=4096),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096 , out_features=4096),
            nn.LeakyReLU(0.3 , inplace=True),
            nn.Linear(in_features=4096 , out_features=self.num_classes)
        )

    def forward(self,x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.FC(x)
        return x
