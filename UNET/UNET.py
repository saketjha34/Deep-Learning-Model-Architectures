import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvLayer, self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.Layer(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNET, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.DownLayer1 = ConvLayer(in_channels=in_channels, out_channels=64)
        self.DownLayer2 = ConvLayer(in_channels=64, out_channels=128)
        self.DownLayer3 = ConvLayer(in_channels=128, out_channels=256)
        self.DownLayer4 = ConvLayer(in_channels=256, out_channels=512)
        self.BottleNeck = ConvLayer(in_channels=512, out_channels=1024)
        self.UpSample1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.UpLayer4 = ConvLayer(in_channels=1024, out_channels=512)
        self.UpSample2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.UpLayer3 = ConvLayer(in_channels=512, out_channels=256)
        self.UpSample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.UpLayer2 = ConvLayer(in_channels=256, out_channels=128)
        self.UpSample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.UpLayer1 = ConvLayer(in_channels=128, out_channels=64)
        self.FinalConv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.DownLayer1(x)
        out = self.MaxPool(out1)
        out2 = self.DownLayer2(out)
        out = self.MaxPool(out2)
        out3 = self.DownLayer3(out)
        out = self.MaxPool(out3)
        out4 = self.DownLayer4(out)
        out = self.MaxPool(out4)
        out = self.BottleNeck(out)
        out = self.UpSample1(out)
        skip1 = torch.cat((out , out4), dim=1)
        out = self.UpLayer4(skip1)
        out = self.UpSample2(out)
        skip2 = torch.cat((out , out3), dim=1)
        out = self.UpLayer3(skip2)
        out = self.UpSample3(out)
        skip3 = torch.cat((out , out2), dim=1)
        out = self.UpLayer2(skip3)
        out = self.UpSample4(out)
        skip4 = torch.cat((out , out1), dim=1)
        out = self.UpLayer1(skip4)
        out = self.FinalConv(out)
        return out