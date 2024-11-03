import torch
from generator import Generator
from discriminator import Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def test()->None:
    x = torch.randn((8,3,24,24)).to(device)
    gen = Generator(in_channels=3,num_residual_blocks=16,scale_factor=2).to(device)
    dis = Discriminator(in_channels=3).to(device)
    print(gen(x).shape)
    print(dis(x).shape)

if __name__ == "__main__":
    test()