import torch
import torch.nn as nn

class DownConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConvLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.layer(x)

class UpConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super(UpConvLayer, self).__init__()
        self.use_dropout = use_dropout
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.layer(x)
        return self.dropout(x) if self.use_dropout else x

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VAE, self).__init__()

        # U-Net style Encoder
        self.down1 = DownConvLayer(in_channels=in_channels, out_channels=64)
        self.down2 = DownConvLayer(in_channels=64, out_channels=128)
        self.down3 = DownConvLayer(in_channels=128, out_channels=256)
        self.down4 = DownConvLayer(in_channels=256, out_channels=512)

        # Latent space parameters
        self.mean_layer = nn.Linear(512 * 16 * 16, latent_dim)  # Mean of the latent space
        self.log_var_layer = nn.Linear(512 * 16 * 16, latent_dim)  # Log variance of the latent space

        # U-Net style Decoder
        self.fc_decoder = nn.Linear(latent_dim, 512 * 16 * 16)  # Latent vector to decoder input
        self.up1 = UpConvLayer(in_channels=512, out_channels=256, use_dropout=True)
        self.up2 = UpConvLayer(in_channels=512, out_channels=128, use_dropout=True)
        self.up3 = UpConvLayer(in_channels=256, out_channels=64, use_dropout=True)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid to map the output between 0 and 1
        )

    def forward(self, x):
        mu, log_var, down_outputs = self.encode(x)
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        reconstructed_x = self.decode(z, down_outputs)
        return reconstructed_x, mu, log_var

    def encode(self, x):
        """Encode the input into mean and log variance of latent space."""
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        x_flat = down4.view(down4.size(0), -1)  # Flatten the tensor for the linear layers
        mu = self.mean_layer(x_flat)
        log_var = self.log_var_layer(x_flat)
        
        return mu, log_var, [down1, down2, down3, down4]  # Return downsampling outputs

    def decode(self, z, down_outputs):
        """Decode the latent vector back into the image."""
        z = self.fc_decoder(z)
        z = z.view(z.size(0), 512, 16, 16)  # Reshape to match the decoder's input dimensions

        up1 = self.up1(z)
        up2 = self.up2(torch.cat([up1, down_outputs[2]], dim=1))  # down3
        up3 = self.up3(torch.cat([up2, down_outputs[1]], dim=1))  # down2
        reconstructed_x = self.final_layer(torch.cat([up3, down_outputs[0]], dim=1))  # down1
        return reconstructed_x