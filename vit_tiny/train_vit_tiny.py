import os
import argparse
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, ConcatDataset, DistributedSampler
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
from PIL import Image, ImageFilter
import numpy as np

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

torch.autograd.set_detect_anomaly(False)

# ==================== UTILITY FUNCTIONS ====================

def count_parameters(model, rank=0):
    """Count and display model parameters"""
    if rank != 0:
        return
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    print("\n" + "="*50)
    print("  MODEL PARAMETER CHECK")
    print(f"  Total parameters:      {total/1e6:.2f} M")
    print(f"  Trainable parameters:  {trainable/1e6:.2f} M")
    print(f"  Frozen parameters:     {frozen/1e6:.2f} M")
    print("="*50 + "\n")
    
    if trainable > 100 * 1e6:
        print(f"⚠️  WARNING: Trainable params exceed 100M parameter cap!")
        print(f"   Current trainable: {trainable/1e6:.2f}M")
        print(f"   Please reduce model size!")
    else:
        print(f"✓ Model is within 100M parameter limit ({trainable/1e6:.2f}M)")

def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        is_distributed = True
        print(f"Rank {rank}/{world_size} initialized.")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        torch.cuda.set_device(0)
        is_distributed = False
        print("Running in non-distributed mode.")
        
    return rank, local_rank, world_size, is_distributed

# ==================== DATA TRANSFORMS ====================

class GaussianBlur:
    """Gaussian blur augmentation"""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x: Image.Image) -> Image.Image:
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))

def get_ssl_transforms(image_size=96):
    """MoCo v3 standard augmentation pipeline"""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

class TwoViewTransform:
    """Generate two augmented views of the same image"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x: Image.Image):
        return [self.base_transform(x), self.base_transform(x)]

# ==================== DATASET ====================

class FlatFolderDataset(Dataset):
    """Load images from a flat folder structure"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = [
            os.path.join(root, f) for f in os.listdir(root) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.samples) == 0:
            print(f"WARNING: No images found in {root}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert('RGB')
            if self.transform:
                return self.transform(img)
            return img
        except Exception:
            return self.__getitem__(random.randint(0, len(self)-1))

# ==================== MODEL ARCHITECTURE ====================

class SmallVisionTransformer(nn.Module):
    """
    Small Vision Transformer encoder (optimized for <100M params)
    
    Default config (~22M params):
    - hidden_dim: 384
    - num_layers: 8
    - num_heads: 6
    - mlp_dim: 1536
    """
    def __init__(
        self,
        image_size=96,
        patch_size=8,
        hidden_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_dim=1536,
    ):
        super().__init__()

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=hidden_dim,
        )

        # Replace classifier head with identity
        self.vit.head = nn.Identity()
        
        self._init_weights()
        self.embed_dim = hidden_dim

    def _init_weights(self):
        """Xavier initialization for better training"""
        for m in self.vit.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.vit(x)  # [B, hidden_dim]


class ProjectionHead(nn.Module):
    """3-layer MLP projection head (MoCo v3 paper specification)"""
    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        # Reduced hidden_dim from 4096 to 2048 to save parameters
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Linear(hidden_dim, out_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PredictionHead(nn.Module):
    """2-layer MLP prediction head (MoCo v3 paper specification)"""
    def __init__(self, in_dim: int = 256, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        # Reduced hidden_dim from 4096 to 2048 to save parameters
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ==================== MOCO V3 MODEL ====================

def momentum_update(model_q: nn.Module, model_k: nn.Module, m: float):
    """EMA update for momentum encoder"""
    with torch.no_grad():
        for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
            p_k.data = p_k.data * m + p_q.data * (1.0 - m)


class MoCoV3(nn.Module):
    """
    MoCo v3: Momentum Contrast for Self-Supervised Visual Representation Learning
    
    Paper: "An Empirical Study of Training Self-Supervised Vision Transformers"
    https://arxiv.org/abs/2104.02057
    
    Architecture:
    - Query encoder: ViT + projection head + prediction head
    - Key encoder: ViT + projection head (momentum updated, no gradient)
    - Contrastive learning with InfoNCE loss
    - Symmetrized loss: ctr(q1, k2) + ctr(q2, k1)
    
    Total parameters: ~50M (well under 100M limit)
    """
    def __init__(
        self,
        encoder_dim: int = 384,
        proj_dim: int = 256,
        proj_hidden_dim: int = 2048,
        pred_hidden_dim: int = 2048,
        m: float = 0.99,
        T: float = 0.2,
    ):
        super().__init__()

        self.m = m
        self.T = T

        # Query encoder (trainable)
        self.encoder_q = SmallVisionTransformer()
        self.proj_q = ProjectionHead(encoder_dim, proj_hidden_dim, proj_dim)
        self.predictor = PredictionHead(proj_dim, pred_hidden_dim, proj_dim)

        # Key encoder (momentum updated, no grad)
        self.encoder_k = SmallVisionTransformer()
        self.proj_k = ProjectionHead(encoder_dim, proj_hidden_dim, proj_dim)

        # Initialize momentum encoder
        self._init_momentum_encoder()

    def _init_momentum_encoder(self):
        """Initialize momentum encoder as copy of query encoder"""
        for p_k, p_q in zip(self.encoder_k.parameters(), self.encoder_q.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

        for p_k, p_q in zip(self.proj_k.parameters(), self.proj_q.parameters()):
            p_k.data.copy_(p_q.data)
            p_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of key encoder"""
        momentum_update(self.encoder_q, self.encoder_k, self.m)
        momentum_update(self.proj_q, self.proj_k, self.m)

    def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss (MoCo v3 paper, Algorithm 1).
        
        Args:
            q: queries [N, C] (L2-normalized)
            k: keys [N, C] (L2-normalized)
        
        Returns:
            loss: scalar loss value
        
        Loss formulation:
            For each query q_i, the positive key is k_i (diagonal)
            All other keys k_j (j != i) are negatives
            
            Loss = -log(exp(q_i·k_i/τ) / sum_j(exp(q_i·k_j/τ)))
        """
        # Compute similarity matrix: [N, N]
        logits = torch.matmul(q, k.t()) / self.T
        
        # Labels: positive pairs are on diagonal
        labels = torch.arange(q.size(0), device=q.device, dtype=torch.long)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        # Scale by 2*tau (as per paper Algorithm 1)
        loss = 2.0 * self.T * loss
        
        return loss

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Forward pass with two augmented views.
        
        Args:
            x1: first view [B, 3, H, W]
            x2: second view [B, 3, H, W]
        
        Returns:
            dict: {'loss': total_loss}
        """
        # === Query branch (with gradient) ===
        feat_q1 = self.encoder_q(x1)
        feat_q2 = self.encoder_q(x2)
        
        z_q1 = self.proj_q(feat_q1)
        z_q2 = self.proj_q(feat_q2)
        
        p_q1 = self.predictor(z_q1)
        p_q2 = self.predictor(z_q2)
        
        # L2 normalize predictions
        p_q1 = F.normalize(p_q1, dim=1, p=2)
        p_q2 = F.normalize(p_q2, dim=1, p=2)
        
        # === Key branch (no gradient) ===
        with torch.no_grad():
            # Update momentum encoder
            self._momentum_update()
            
            feat_k1 = self.encoder_k(x1)
            feat_k2 = self.encoder_k(x2)
            
            z_k1 = self.proj_k(feat_k1)
            z_k2 = self.proj_k(feat_k2)
            
            # L2 normalize keys
            z_k1 = F.normalize(z_k1, dim=1, p=2)
            z_k2 = F.normalize(z_k2, dim=1, p=2)
        
        # === Compute symmetrized loss ===
        # q1 predicts k2, q2 predicts k1
        loss = self.contrastive_loss(p_q1, z_k2) + self.contrastive_loss(p_q2, z_k1)
        
        return {"loss": loss}


# ==================== TRAINING ====================

def adjust_learning_rate(optimizer, step, total_steps, base_lr, batch_size, warmup_steps):
    """
    Learning rate schedule (MoCo v3 paper):
    - Linear warmup for warmup_steps
    - Cosine decay after warmup
    - Linear scaling rule: lr = base_lr * batch_size / 256
    """
    # Linear scaling rule
    lr = base_lr * batch_size / 256
    
    if step < warmup_steps:
        # Linear warmup
        lr = lr * step / max(warmup_steps, 1)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def adjust_moco_momentum(base_m, step, total_steps):
    """
    Cosine schedule for MoCo momentum (as in moco-m-cos).
    Starts at base_m and gradually increases towards 1.0.
    """
    if total_steps <= 0:
        return base_m
    # progress in [0, 1]
    progress = step / total_steps
    # cosine schedule from base_m -> 1
    m = 1.0 - (1.0 - base_m) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return m



def main():
    parser = argparse.ArgumentParser(description='MoCo v3 Training')
    
    # Data paths
    parser.add_argument('--data_path1', required=True, help='/scratch/sd6217/dl_project/cc3m_96px/')
    parser.add_argument('--data_path2', required=True, help='/scratch/sd6217/dl_project/pretrain_data2/')
    parser.add_argument('--data_path3', required=True, help='/scratch/sd6217/dl_project/pretrain_data3/')
    parser.add_argument('--data_path4', required=True, help='/scratch/sd6217/dl_project/pretrain_data4/')
    
    # Training hyperparameters (MoCo v3 paper defaults, adjusted for smaller model)
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='Batch size per GPU (paper uses 4096, we use 1024 for smaller model)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (paper: 300 for ImageNet, we use 400 for smaller dataset)')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='Warmup epochs (paper: 40)')
    parser.add_argument('--base_lr', type=float, default=1.0e-4,
                        help='Base learning rate (paper: 1.0e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='Weight decay (paper: 0.1)')
    

    parser.add_argument('--max_samples', type=int, default=None,
                        help='If set, cap the number of training samples (per epoch) to this value.')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='If set, override the total number of epochs to this value.')
    parser.add_argument('--max_batch_size', type=int, default=None,
                        help='If set, cap batch size per GPU to this value.')
    
    # Model hyperparameters
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Momentum coefficient for key encoder (paper: 0.99)')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='Temperature for contrastive loss (paper: 0.2)')
    
    # System
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', default='/scratch/sd6217/dl_project/checkpoints')
    parser.add_argument('--wandb_project', type=str, default='mocov3-ssl')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--log_freq', type=int, default=50)
    
    args = parser.parse_args()

    # *** NEW: apply optional caps ***
    if args.max_batch_size is not None:
        args.batch_size = min(args.batch_size, args.max_batch_size)

    if args.max_epochs is not None:
        args.epochs = min(args.epochs, args.max_epochs)
    
    rank, local_rank, world_size, is_distributed = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # WandB initialization
    if rank == 0 and args.wandb_project and HAS_WANDB:
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Create model
    if rank == 0:
        print("\n" + "="*60)
        print("Creating MoCo v3 Model")
        print("="*60)
    
    model = MoCoV3(
        encoder_dim=384,
        proj_dim=256,
        proj_hidden_dim=2048,  # Reduced from 4096 to stay under 100M
        pred_hidden_dim=2048,  # Reduced from 4096 to stay under 100M
        m=args.momentum,
        T=args.temperature,
    )
    
    count_parameters(model, rank)
    
    # Convert to distributed
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=False
        )
    
    # Data transforms
    tfm = TwoViewTransform(get_ssl_transforms(96))
    
    # Load datasets from 4 paths
    datasets = []
    for i, data_path in enumerate([args.data_path1, args.data_path2, args.data_path3, args.data_path4], 1):
        if os.path.exists(data_path):
            ds = FlatFolderDataset(data_path, transform=tfm)
            datasets.append(ds)
            if rank == 0:
                print(f"Dataset {i}: {len(ds):,} images from {data_path}")
        else:
            if rank == 0:
                print(f"WARNING: Dataset path {data_path} does not exist!")
    
    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")
    
    full_ds = ConcatDataset(datasets)
    if rank == 0:
        print(f"\nTotal dataset size: {len(full_ds):,} images\n")

    # *** NEW: optionally cap dataset size ***
    if args.max_samples is not None and args.max_samples > 0:
        from torch.utils.data import Subset
        max_n = min(args.max_samples, len(full_ds))
        indices = list(range(max_n))
        if rank == 0:
            print(f"Using only {max_n:,} samples out of {len(full_ds):,} due to --max_samples")
        full_ds = Subset(full_ds, indices)
    
    # DataLoader
    if is_distributed:
        sampler = DistributedSampler(full_ds, shuffle=True)
    else:
        sampler = None
    
    loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    
    # Optimizer (AdamW as per paper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,  # Will be scaled by learning rate schedule
        weight_decay=args.weight_decay
    )
    
    # Training setup
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    
    if rank == 0:
        print("="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Method: MoCo v3")
        print(f"Batch size (per GPU): {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps:,}")
        print(f"Base learning rate: {args.base_lr}")
        print(f"Effective learning rate: {args.base_lr * args.batch_size / 256:.6f}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Warmup steps: {warmup_steps:,} ({args.warmup_epochs} epochs)")
        print(f"Temperature: {args.temperature}")
        print(f"Momentum: {args.momentum}")
        print("="*60 + "\n")
    
    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    
    if args.resume and os.path.isfile(args.resume):
        if rank == 0:
            print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        
        if is_distributed:
            model.module.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'])
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
        
        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}\n")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0

        # *** NEW: epoch-level timer ***
        epoch_start_time = time.time()
        
        for step, views in enumerate(loader):
            global_batch_size = args.batch_size * world_size  # world_size is already defined above

            curr_lr = adjust_learning_rate(
                optimizer, 
                global_step, 
                total_steps, 
                args.base_lr,
                global_batch_size,
                warmup_steps
            )
            
            # Move data to GPU
            x1, x2 = views[0].cuda(local_rank, non_blocking=True), views[1].cuda(local_rank, non_blocking=True)

            # Select the actual MoCoV3 module (handle DDP or not)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                moco = model.module
            else:
                moco = model


            # Update momentum m with cosine schedule
            moco.m = adjust_moco_momentum(args.momentum, global_step, total_steps)
            
            # Forward pass
            output = model(x1, x2)
            loss = output['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (helps with stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Logging
            if step % args.log_freq == 0 and rank == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] "
                      f"Step [{step}/{steps_per_epoch}] "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {curr_lr:.6f}")
                
                if HAS_WANDB:
                    wandb.log({
                        "loss": loss.item(),
                        "lr": curr_lr,
                        "epoch": epoch,
                        "step": global_step,
                    })
        
        # Epoch summary
        avg_loss = epoch_loss / steps_per_epoch
        if rank == 0:
            epoch_time = time.time() - epoch_start_time   # *** NEW ***
            elapsed = time.time() - start_time            # total time so far

            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{args.epochs}] Summary")
            print(f"{'='*60}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Epoch time: {epoch_time/60:.2f} minutes "
                  f"({epoch_time:.1f} seconds)")         # *** NEW ***
            print(f"Total time elapsed: {elapsed/3600:.2f} hours")
            print(f"{'='*60}\n")

        
        # Save checkpoint
        if rank == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs - 1):
            checkpoint = {
                'epoch': epoch,
                'model': model.module.state_dict() if is_distributed else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'global_step': global_step,
                'args': vars(args),
            }
            
            latest_path = os.path.join(args.output_dir, "mocov3_latest.pth")
            torch.save(checkpoint, latest_path)
            
            if epoch % (args.save_freq * 5) == 0 or epoch == args.epochs - 1:
                epoch_path = os.path.join(args.output_dir, f"mocov3_epoch{epoch}.pth")
                torch.save(checkpoint, epoch_path)
                print(f"✓ Saved checkpoint: {epoch_path}\n")
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time/3600:.2f} hours")
        print(f"{'='*60}\n")
        if HAS_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()
