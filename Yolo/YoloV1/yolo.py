import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    """
    A convolutional block consisting of Conv2D, BatchNorm, LeakyReLU, 
    and an optional MaxPooling layer.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layer.
        padding (int): Padding for the convolutional layer.
        use_maxpool (bool): Whether to include a MaxPooling layer (default: False).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_maxpool :bool=False):
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
    """
    YOLOv1 neural network implementation with convolutional layers, 
    fully connected layers, and loss computation.

    Parameters:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        S (int): Grid size of the output (default: 7).
        num_boxes (int): Number of bounding boxes per grid cell (default: 2).
        num_classes (int): Number of classes for classification (default: 20).
    """
    def __init__(self, in_channels: int=3, S :int=7 , num_boxes :int=2 ,num_classes :int=20):
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
        """
        Forward pass of the YOLOv1 model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, S * S * (num_classes + num_boxes * 5)).
        """
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