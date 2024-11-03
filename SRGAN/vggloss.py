import torch
import torch.nn as nn
from torchvision.models import vgg19

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class VGGLoss(nn.Module):
    def __init__(self,):
        self.vgg =  vgg19(pretrained=True).features[:36].to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self,input,target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features,vgg_target_features)
    
    