import torch
import torch.nn as nn
from torchvision import  models
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SpinalVGGNet(nn.Module):
    def __init__(self, half_in_size, layer_width, num_classes):
        super(SpinalVGGNet, self).__init__()
        self.half_in_size = half_in_size
        
        self.SpinalLayer1 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        
        self.SpinalLayer2 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        
        self.SpinalLayer3 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        
        self.SpinalLayer4 = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(half_in_size+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), 
            nn.ReLU(inplace=True),)
        
        self.FC = nn.Sequential(
            nn.Dropout(p = 0.5), 
            nn.Linear(layer_width*4, num_classes),)
        
    def forward(self, x):
        
        x1 = self.SpinalLayer1(x[:, 0:self.half_in_size])
        x2 = self.SpinalLayer2(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x1], dim=1))
        x3 = self.SpinalLayer3(torch.cat([ x[:,0:self.half_in_size], x2], dim=1))
        x4 = self.SpinalLayer4(torch.cat([ x[:,self.half_in_size:2*self.half_in_size], x3], dim=1))
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = self.FC(x)
        return x

def test():
    model_features = models.vgg19_bn(pretrained=True).to(device)
    num_features = model_features.classifier[0].in_features
    half_in_size = round(num_features/2)
    layer_width = 512
    num_classes = 1000
    model_features.classifier = SpinalVGGNet(half_in_size, layer_width, num_classes).to(device)
    model = model_features.to(device)
    x = torch.randn([10,3,224,224]).to(device)
    print(model(x).shape)

if __name__ == "__main__":
    test()
