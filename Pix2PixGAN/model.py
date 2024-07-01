from generator import Generator
from discriminator import Discriminator
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test():
    x = torch.randn([32,3,256,256]).to(device)
    y = torch.randn([32,3,256,256]).to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    print(f'Generator Output Shape:{generator(x).shape}')
    print(f'Discriminator Output Shape:{discriminator(x,y).shape}')

if __name__ == "__main__":
    test()