import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self , in_channels , num_filters1x1 ,num_filters3x3_reduce , num_filters3x3 ,
                 num_filters5x5_reduce,num_filters5x5 , pooling
                  ):
        super(InceptionBlock,self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=num_filters1x1 , kernel_size=1 ,padding=0 , stride=1 ,bias=False),
            nn.BatchNorm2d(num_filters1x1),
            nn.LeakyReLU(0.2 ,inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=num_filters3x3_reduce, kernel_size=1 ,padding=0 , stride=1,bias=False),
            nn.BatchNorm2d(num_filters3x3_reduce),
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.Conv2d(in_channels=num_filters3x3_reduce , out_channels=num_filters3x3 , kernel_size=3 , padding=1 , stride=1,bias=False),
            nn.BatchNorm2d(num_filters3x3),
            nn.LeakyReLU(0.2 ,inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=num_filters5x5_reduce, kernel_size=1 ,padding=0 , stride=1 ,bias=False),
            nn.BatchNorm2d(num_filters5x5_reduce),
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.Conv2d(in_channels=num_filters5x5_reduce , out_channels=num_filters5x5 , kernel_size=5 , padding=2 , stride=1,bias=False),
            nn.BatchNorm2d(num_filters5x5),
            nn.LeakyReLU(0.2 ,inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3 ,stride=1 , padding=1),
            nn.Conv2d(in_channels=in_channels , out_channels=pooling , kernel_size=1,bias=False),
            nn.BatchNorm2d(pooling),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x), ] , dim=1)

class InceptionNetV1(nn.Module):
    def __init__(self , in_channels =3 ,num_classes = 10):
        super(InceptionNetV1,self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=64 , kernel_size=7 , stride=2 , padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2,padding=1)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=64 , out_channels=192 , kernel_size=3 , stride=1 , padding=1,bias=False),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2 ,inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2,padding=1)
        )

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3 ,stride=2 , padding=1)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3 ,stride=2 , padding=1)

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.AvgPool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.FC= nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(1024 , num_classes)
        )

    def forward(self, x):
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.AvgPool(x)
        x = self.FC(x)
        return x

x = torch.randn([10,3,224,224])
model = InceptionNetV1()
print(model(x).shape)