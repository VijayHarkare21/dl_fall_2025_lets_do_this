"""
VICReg Training Script
=====================================
- Backbone: ResNeXt-50
- Expander: 8192
- Epochs: 100
- Feature: AUTO-RESUME (Saves/Loads 'last_checkpoint.pth')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torchvision.models import resnext50_32x4d
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from pathlib import Path
from tqdm import tqdm
import math
import random
import os
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Architecture
    'backbone': 'resnext50_32x4d', 
    'projection_dim': 8192,
    
    # Loss Weights
    'sim_coeff': 25.0,
    'std_coeff': 25.0,
    'cov_coeff': 1.0,
    
    # Training
    'epochs': 100,
    'batch_size': 256,
    'base_lr': 0.02,
    'weight_decay': 1e-6,
    'warmup_epochs': 10,
    
    # Data
    'data_dir': '/teamspace/studios/this_studio/data/train',
    'input_size': 96,
    'num_workers': 8,
    
    # Checkpointing
    'checkpoint_dir': './checkpoints_vicreg_optimized',
    'resume_file': 'last_checkpoint.pth',  # file we look for to resume
    'log_file': 'training_log_optimized.csv'
}

# ============================================================================
# AUGMENTATIONS
# ============================================================================

class Solarization(object):
    def __init__(self, p):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        return img

class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() <= self.p:
            return img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(self.radius_min, self.radius_max)
                )
            )
        return img

class VICRegAugmentation:
    def __init__(self, input_size=96):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5),
            Solarization(p=0.1), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, x):
        return self.transform(x)

# ============================================================================
# DATASET
# ============================================================================

class SSLDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        print(f"Scanning {image_dir}...")
        self.image_paths = sorted(list(self.image_dir.rglob('*.jpg')) + list(self.image_dir.rglob('*.png')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}. Check your path!")
            
        print(f"Found {len(self.image_paths)} images.")
        self.transform = VICRegAugmentation(CONFIG['input_size'])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            x1 = self.transform(img)
            x2 = self.transform(img)
            return x1, x2
        except Exception:
            return torch.randn(3, CONFIG['input_size'], CONFIG['input_size']), \
                   torch.randn(3, CONFIG['input_size'], CONFIG['input_size'])

# ============================================================================
# MODEL
# ============================================================================

class VICReg(nn.Module):
    def __init__(self, mlp_dim=8192):
        super().__init__()
        self.backbone = resnext50_32x4d(weights=None)
        self.embedding_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.expander = nn.Sequential(
            nn.Linear(self.embedding_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim)
        )
        
    def forward(self, x1, x2):
        rep1 = self.backbone(x1)
        z1 = self.expander(rep1)
        rep2 = self.backbone(x2)
        z2 = self.expander(rep2)
        return z1, z2

# ============================================================================
# LOSS
# ============================================================================

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vicreg_loss_func(z1, z2):
    lambd, mu, nu = CONFIG['sim_coeff'], CONFIG['std_coeff'], CONFIG['cov_coeff']
    N, D = z1.shape
    
    sim_loss = F.mse_loss(z1, z2)
    
    std_z1 = torch.sqrt(z1.var(dim=0) + 0.0001)
    std_z2 = torch.sqrt(z2.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    cov_loss = (off_diagonal(cov_z1).pow(2).sum() / D +
                off_diagonal(cov_z2).pow(2).sum() / D)
    
    loss = lambd * sim_loss + mu * std_loss + nu * cov_loss
    return loss, sim_loss, std_loss, cov_loss

# ============================================================================
# UTILS
# ============================================================================

def adjust_learning_rate(optimizer, epoch, args):
    max_epochs = args['epochs']
    warmup_epochs = args['warmup_epochs']
    base_lr = args['base_lr'] * (args['batch_size'] / 256)
    
    if epoch < warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG
    }, path)

# ============================================================================
# MAIN
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = checkpoint_dir / CONFIG['resume_file']
    
    # Init Logging
    if not os.path.exists(CONFIG['log_file']):
        with open(CONFIG['log_file'], "w") as f:
            f.write("Epoch,Batch,TotalLoss,InvLoss,StdLoss,CovLoss,LR\n")
    
    print(f"Initializing VICReg...")
    
    dataset = SSLDataset(CONFIG['data_dir'])
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=True, 
        drop_last=True
    )
    
    model = VICReg(mlp_dim=CONFIG['projection_dim']).to(device)
    optimizer = SGD(model.parameters(), lr=CONFIG['base_lr'], weight_decay=CONFIG['weight_decay'], momentum=0.9)
    
    # --- RESUME LOGIC ---
    start_epoch = 1
    if resume_path.exists():
        print(f"\nFound checkpoint: {resume_path}")
        print("Resuming training...")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Resumed from Epoch {checkpoint['epoch']}")
    else:
        print("\nNo checkpoint found. Starting from scratch.")

    print(f"Starting training loop from Epoch {start_epoch} to {CONFIG['epochs']}...")
    
    start_time = time.time()
    
    for epoch in range(start_epoch, CONFIG['epochs'] + 1):
        lr = adjust_learning_rate(optimizer, epoch, CONFIG)
        model.train()
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{CONFIG["epochs"]}')
        
        for batch_idx, (x1, x2) in enumerate(pbar):
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            z1, z2 = model(x1, x2)
            loss, sim, std, cov = vicreg_loss_func(z1, z2)
            
            if torch.isnan(loss):
                print("Error: Loss is NaN. Stopping training.")
                return
            
            loss.backward()
            optimizer.step()
            
            # Log periodically
            if batch_idx % 10 == 0:
                with open(CONFIG['log_file'], "a") as f:
                    f.write(f"{epoch},{batch_idx},{loss.item():.4f},{sim.item():.4f},{std.item():.4f},{cov.item():.4f},{lr:.6f}\n")
            
            pbar.set_postfix({
                'L': f'{loss.item():.2f}',
                'Std': f'{std.item():.2f}',
                'LR': f'{lr:.4f}'
            })
            
        # --- SAVE CHECKPOINT EVERY EPOCH ---
        # 1. Save the "Resume" checkpoint (overwritten every epoch)
        save_checkpoint(model, optimizer, epoch, resume_path)
        
        # 2. Save the "Permanent" checkpoint (every 10 epochs)
        if epoch % 10 == 0 or epoch == CONFIG['epochs']:
            perm_path = checkpoint_dir / f'encoder_epoch{epoch}.pth'
            # Save only backbone for evaluation convenience
            torch.save(model.backbone.state_dict(), perm_path)
            print(f"Saved permanent encoder to {perm_path}")
            
        elapsed = (time.time() - start_time) / 3600
        print(f"Epoch {epoch} complete. Total session time: {elapsed:.2f} hours.")

if __name__ == "__main__":
    main()