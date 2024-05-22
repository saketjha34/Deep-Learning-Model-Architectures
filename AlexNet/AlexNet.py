import torch
import torch.nn as nn

#Alexnet input image size - 3 x 227 x 227
class Alexnet(nn.Module):
    def __init__(self , num_classes) :
        super().__init__()
        self.num_classes = num_classes

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3 , out_channels=96 , kernel_size=(11,11) , padding=0, stride=4),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3,3) , stride = 2),
            nn.BatchNorm2d(96)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96 , out_channels=256 , kernel_size=(5,5) , padding=2 , stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3,3) , stride=2),
            nn.BatchNorm2d(256)
        )

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256 , out_channels=384 , kernel_size=(3,3) , stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384 , kernel_size=(3,3) , padding=1 , stride = 1),
            nn.LeakyReLU()
        )

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384 , out_channels=256 , kernel_size=(3,3) , padding=1 ,stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(3,3) ,stride=2)
        )

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256*6*6 , out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096 , out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096 , out_features=self.num_classes)
        )

    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.FC(x)
        return x
    


model = Alexnet(num_classes=10)
x = torch.randn([1,3,227,227])
print(model(x).shape)

