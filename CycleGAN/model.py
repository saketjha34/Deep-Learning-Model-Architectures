from generator import Generator
from discriminator import Discriminator
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test()->None:
    x = torch.randn([1,3,256,256]).to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    print(f'Generator Output Shape:{generator(x).shape}')
    print(f"Discriminator Output Shape:{discriminator(x).shape}")

if __name__ == "__main__":
    test()