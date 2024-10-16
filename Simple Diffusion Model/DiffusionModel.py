import torch
import torch.nn as nn
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Define the forward diffusion process with proper dimension broadcasting
def forward_diffusion(x0, noise, t, T):
    """
    Applies the forward diffusion process.
    
    Args:
    - x0: Original image tensor (batch_size, channels, height, width)
    - noise: Gaussian noise tensor with the same shape as x0
    - t: Current time step in the diffusion process
    - T: Total number of time steps
    
    Returns:
    - xt: Noised image tensor at time step t
    """
    batch_size, channels, height, width = x0.shape
    
    # t should be shaped for broadcasting: (batch_size, 1, 1, 1)
    t = t.view(batch_size, 1, 1, 1).float()  # Expand t for broadcasting
    T = float(T)  # Ensure T is a float for division

    # Apply diffusion process (broadcasting over all dimensions)
    alpha = 1 - (t / T)  # Shape: (batch_size, 1, 1, 1), broadcastable
    xt = alpha * x0 + (1 - alpha) * noise
    
    return xt



#U-Net-like model for reverse process (denoising)
class DownConvLayer(nn.Module):
    def __init__(self , in_channels , out_channels):
        super(DownConvLayer, self).__init__()
        self.Layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels , kernel_size=4 , stride = 2 , padding = 1 , padding_mode="reflect" , bias = False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2 , inplace=True)
        )
    def forward(self , x):
        return self.Layer(x)

class UpConvLayer(nn.Module):
    def __init__(self , in_channels , out_channels , use_dropout = False):
        super(UpConvLayer, self).__init__()
        self.use_dropout = use_dropout
        self.Layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels , kernel_size=4 , stride = 2 , padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.Dropout = nn.Dropout(p = 0.5)
    def forward(self , x):
        x = self.Layer(x)
        return self.Dropout(x) if self.use_dropout else x

# input -> [batch_size,3,512,512]
class Pix2PixUnet(nn.Module):
    def __init__(self, in_channels = 3):
        super(Pix2PixUnet,self).__init__()

        self.DownConvLayer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=2,padding=1,padding_mode="reflect"),# [batch_size,64,256,256]
            nn.LeakyReLU(0.2 , inplace=True),
        )

        self.DownConvLayer2 = DownConvLayer(in_channels=64 ,out_channels=128) # [batch_size,128,128,128]
        self.DownConvLayer3 = DownConvLayer(in_channels=128 ,out_channels=256) # [batch_size,256,64,64]
        self.DownConvLayer4 = DownConvLayer(in_channels=256 ,out_channels=512) # [batch_size,512,32,32]
        self.DownConvLayer5 = DownConvLayer(in_channels=512 ,out_channels=512) # [batch_size,512,16,16]
        self.DownConvLayer6 = DownConvLayer(in_channels=512 ,out_channels=512) # [batch_size,512,8,8]
        self.DownConvLayer7 = DownConvLayer(in_channels=512 ,out_channels=512) # [batch_size,512,4,4]

        self.BottleNeck = nn.Sequential(
            nn.Conv2d(in_channels=512 , out_channels= 512,kernel_size= 4, stride=2, padding=1), # [batch_size,512,2,2]
            nn.ReLU(inplace=True)
        )

        self.UpConvLayer1 = UpConvLayer(in_channels=512 , out_channels=512 ,use_dropout=True) # [batch_size,512,4,4]
        self.UpConvLayer2 = UpConvLayer(in_channels=1024 , out_channels=512 ,use_dropout=True)
        self.UpConvLayer3 = UpConvLayer(in_channels=1024, out_channels=512 ,use_dropout=True)
        self.UpConvLayer4 = UpConvLayer(in_channels=1024 , out_channels=512 ,use_dropout=False)
        self.UpConvLayer5 = UpConvLayer(in_channels=1024 , out_channels=256 ,use_dropout=False)
        self.UpConvLayer6 = UpConvLayer(in_channels=512 , out_channels=128 ,use_dropout=False)
        self.UpConvLayer7 = UpConvLayer(in_channels=256 , out_channels=64 ,use_dropout=False)

        self.FinalConvLayer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self,x):
        down1 = self.DownConvLayer1(x)
        down2 = self.DownConvLayer2(down1)
        down3 = self.DownConvLayer3(down2)
        down4 = self.DownConvLayer4(down3)
        down5 = self.DownConvLayer5(down4)
        down6 = self.DownConvLayer6(down5)
        down7 = self.DownConvLayer7(down6)
        bottleneck = self.BottleNeck(down7)
        up1 = self.UpConvLayer1(bottleneck)
        up2 = self.UpConvLayer2(torch.cat([up1 ,down7],dim=1))
        up3 = self.UpConvLayer3(torch.cat([up2 ,down6],dim=1))
        up4 = self.UpConvLayer4(torch.cat([up3 ,down5],dim=1))
        up5 = self.UpConvLayer5(torch.cat([up4 ,down4],dim=1))
        up6 = self.UpConvLayer6(torch.cat([up5 ,down3],dim=1))
        up7 = self.UpConvLayer7(torch.cat([up6 ,down2],dim=1))
        return self.FinalConvLayer(torch.cat([up7 ,down1],dim=1))
    

if __name__ == "__main__":
    diffusion_model = Pix2PixUnet(in_channels = 3).to(device)