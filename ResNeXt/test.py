import torch 
from ResNeXt import ResNeXt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test()->None:
    model = ResNeXt(layers=[3, 4, 6, 3], groups=32, width_per_group=4).to(device)
    x = torch.randn([32,3,224,224] , device =device)
    out = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {out.shape}")
    
if __name__ == "__main__":
    test()