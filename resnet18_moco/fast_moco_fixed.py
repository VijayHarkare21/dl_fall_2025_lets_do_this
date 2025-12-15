
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import subprocess
import kornia.augmentation as K


class GPUAugmentation(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            K.RandomHorizontalFlip(p=0.5),
            K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                       std=torch.tensor([0.229, 0.224, 0.225]))
        )
    
    def forward(self, x):
        return self.aug(x)


class FastJPGDataset(Dataset):
    """Load JPGs directly - handles 500K+ files!"""
    def __init__(self, image_dir='/teamspace/studios/this_studio/data/train', max_images=500000):
        print(f"Loading image paths from {image_dir}...")
        
        # Use 'find' command - handles unlimited files
        result = subprocess.run(
            ['find', image_dir, '-name', '*.jpg', '-type', 'f'],
            capture_output=True,
            text=True,
            check=True
        )
        
        self.image_paths = [p for p in result.stdout.strip().split('\n') if p]
        
        print(f"Found {len(self.image_paths)} total images")
        
        # Limit to max_images
        if max_images and max_images < len(self.image_paths):
            self.image_paths = sorted(self.image_paths)[:max_images]
            print(f"Using first {len(self.image_paths)} images")
        
        print(f"Dataset ready: {len(self.image_paths)} images")
        
        self.to_tensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.to_tensor(img)


class MoCo(nn.Module):
    def __init__(self, dim=128, K=2048, m=0.999, T=0.2):
        super().__init__()
        self.K, self.m, self.T = K, m, T
        
        self.aug_q = GPUAugmentation()
        self.aug_k = GPUAugmentation()
        
        self.encoder_q = resnet18(weights=None)
        dim_mlp = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
        )
        
        self.encoder_k = resnet18(weights=None)
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim)
        )
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, images):
        im_q = self.aug_q(images)
        with torch.no_grad():
            im_k = self.aug_k(images)
        
        q = F.normalize(self.encoder_q(im_q), dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.encoder_k(im_k), dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        self._dequeue_and_enqueue(k)
        return logits, labels


def train():
    batch_size = 256
    epochs = 50
    lr = 0.03
    
    print("="*80)
    print("MoCo Training - 500K Images")
    print("="*80)
    
    model = MoCo(dim=128, K=2048).cuda()
    print("ResNet18 MoCo with GPU augmentations")
    
    dataset = FastJPGDataset(max_images=500000)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    print(f"DataLoader: {len(loader)} batches/epoch")
    print("="*80)
    
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs('checkpoints_fixed', exist_ok=True)
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f'Epoch {epoch}')
        for images in pbar:
            images = images.cuda()
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                logits, labels = model(images)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f'\nEpoch {epoch}: Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f}\n')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, f'checkpoints_fixed/epoch{epoch+1}.pth')
            print(f"Checkpoint saved: epoch{epoch+1}.pth\n")
    
    torch.save(model.encoder_q.state_dict(), 'checkpoints_fixed/encoder_final.pth')
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)


if __name__ == '__main__':
    train()