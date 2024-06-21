import torch
import torch.nn as nn

class MobileNetV1(nn.Module):
    def __init__(self, in_channels =3 , num_classes = 10):
        super(MobileNetV1,self).__init__()

        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=32 , kernel_size=3 , stride=2 , padding=1 , bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2 , inplace=True)
        )

        self.ConvLayer2 = nn.Sequential(
            self._create_depthwise_block(in_channels=32 , out_channels=64 ,stride=1),
            self._create_depthwise_block(in_channels=64 , out_channels=128 ,stride=2),
        )

        self.ConvLayer3 = nn.Sequential(
            self._create_depthwise_block(in_channels=128 , out_channels=128 ,stride=1),
            self._create_depthwise_block(in_channels=128 , out_channels=128 ,stride=2),
        )

        self.ConvLayer4 = nn.Sequential(
            self._create_depthwise_block(in_channels=128 , out_channels=256 ,stride=1),
            self._create_depthwise_block(in_channels=256 , out_channels=256 ,stride=1),
        )

        self.ConvLayer5 = nn.Sequential(
            self._create_depthwise_block(in_channels=256 , out_channels=512 ,stride=2),
            self._create_depthwise_block(in_channels=512 , out_channels=512 ,stride=1),
            self._create_depthwise_block(in_channels=512 , out_channels=512 ,stride=1),
            self._create_depthwise_block(in_channels=512 , out_channels=512 ,stride=1),
            self._create_depthwise_block(in_channels=512 , out_channels=512 ,stride=1),
            self._create_depthwise_block(in_channels=512 , out_channels=512 ,stride=1),
        )
        self.ConvLayer6 = nn.Sequential(
            self._create_depthwise_block(in_channels=512 , out_channels=1024 ,stride=2),
            self._create_depthwise_block(in_channels=1024 , out_channels=1024 ,stride=1),
        )

        self.AvgPool = nn.AvgPool2d(7 , 7)

        self.FC = nn.Sequential(
            nn.Flatten(),
            nn.LeakyReLU(0.2 , inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024 , num_classes)
        )


    def forward(self , x ) :
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.ConvLayer5(x)
        x = self.ConvLayer6(x)
        x = self.AvgPool(x)
        x = self.FC(x)
        return x

    def _create_depthwise_block(self, in_channels ,out_channels , stride):
        return nn.Sequential(
            
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=(3,3),padding=1,stride = stride, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2 , inplace=True),

            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),padding=0,stride =1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True),
        )

# x = torch.randn([1,3,224,224])
# model = MobileNetV1()
# print(model(x).shape)