

import torch.nn as nn
import torch
import tqdm
from DiffusionModel import diffusion_model

def train_model(model, train_loader, optimizer, criterion, T, device):
    model.train()
    
    total_loss = 0
    for batch_idx, (input_image, target_image) in enumerate(tqdm(train_loader)):
        input_image, target_image = input_image.to(device), target_image.to(device)
        
        optimizer.zero_grad()
        
        # Apply forward diffusion (add noise)
        noise = torch.randn_like(input_image)
        t = torch.randint(0, T, (input_image.shape[0],)).to(device)  # Random time steps
        xt = diffusion_model(input_image, noise, t, T)
        
        # Denoising step: predict the target (original) image from noisy image
        reconstructed = model(xt)
        loss = criterion(reconstructed, target_image)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss