import os
import argparse
import math
import random
import time
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, DistributedSampler
from torchvision import transforms
from timm import create_model

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    
torch.autograd.set_detect_anomaly(False)

# --- Filter functions for LARS (must be at module level for pickling) ---
def exclude_from_wd_and_lars(param):
    """Return True to SKIP weight decay and LARS adaptation."""
    return True

def include_in_wd_and_lars(param):
    """Return False to APPLY weight decay and LARS adaptation."""
    return False

# --- 1. UTILITIES: COUNTS & CHECKS ---
def print_parameter_count(model, projector, rank=0):
    if rank != 0: return
    
    backbone_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    projector_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    total_params = backbone_params + projector_params
    
    print("\n" + "="*40)
    print(f"  MODEL PARAMETER CHECK")
    print(f"  Backbone (Encoder):   {backbone_params/1e6:.2f} M")
    print(f"  Projector (Head):     {projector_params/1e6:.2f} M")
    print(f"  TOTAL TRAINABLE:      {total_params/1e6:.2f} M")
    print("="*40 + "\n")
    
    # 100M Cap Check
    if backbone_params > 100 * 1e6:
        print(f" WARNING: Backbone exceeds 100M parameter cap!")

def assert_random_init(model, rank=0):
    """
    Verifies that the model is not loading pre-trained weights by checking
    statistics of the first conv layer.
    """
    if rank != 0: return
    
    print(" Verifying Random Initialization...")
    
    # Get the first convolutional layer
    first_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_layer = m
            break
            
    if first_layer is None:
        print("Could not find Conv2d layer to verify.")
        return

    # Check weights statistics
    w = first_layer.weight.data
    mean = w.mean().item()
    std = w.std().item()
    
    print(f"   First Layer Stats -> Mean: {mean:.5f}, Std: {std:.5f}")
    
    # If pretrained on ImageNet, mean is usually very specific or non-zero structure.
    # Random init (Kaiming) usually has mean ~0.0.
    if abs(mean) > 0.1:
        print(" WARNING: Weights look suspicious (Mean is far from 0). Check if pretrained=True?")
    else:
        print(" Weights look random (Mean approx 0).")

# --- 2. DISTRIBUTED SETUP ---
def setup_distributed():
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

# --- 3. OPTIMIZER (LARS) & SCHEDULER ---
class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.005,
                 weight_decay_filter=None, larc_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        larc_adaptation_filter=larc_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None: continue
                grad = p.grad
                
                # 1. Apply Weight Decay (Check filter)
                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    if g['weight_decay'] > 0:
                        grad.add_(p, alpha=g['weight_decay'])
                
                # 2. LARS Adaptation (Check filter)
                # Paper: "biases and batch normalization parameters are excluded from LARS adaptation" 
                if g['larc_adaptation_filter'] is None or not g['larc_adaptation_filter'](p):
                    w_norm = torch.norm(p)
                    g_norm = torch.norm(grad)
                    if w_norm * g_norm > 0:
                        adaptive_lr = g['eta'] * w_norm / (g_norm + 1e-6)
                        grad.mul_(adaptive_lr)
                
                # 3. SGD Step
                p_update = grad
                if g['momentum'] > 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(g['momentum']).add_(grad)
                    p_update = buf

                p.add_(p_update, alpha=-g['lr'])

        return loss

def adjust_learning_rate(optimizer, step, total_steps, args):
    """Linear Warmup + Cosine Decay (per-group scaling via lr_scale)."""
    warmup_steps = args.warmup_epochs * args.steps_per_epoch

    if step < warmup_steps:
        lr = args.lr * step / max(warmup_steps, 1)
    else:
        step -= warmup_steps
        max_steps = max(total_steps - warmup_steps, 1)
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = args.lr * 0.001
        lr = args.lr * q + end_lr * (1 - q)

    for param_group in optimizer.param_groups:
        scale = param_group.get('lr_scale', 1.0)
        param_group['lr'] = lr * scale

    return lr

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# --- 4. LOSS FUNCTIONS ---
class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3):
        super().__init__()
        self.lambda_coeff = lambda_coeff
        self.batch_size = batch_size
        
        # For logging
        self.last_on_diag = 0.0
        self.last_off_diag = 0.0

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        eps = 1e-6
        
        # Normalize (these are fine - not in-place)
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + eps)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + eps)
        
        # Compute correlation matrix
        local_batch_size = z1.shape[0]
        c = torch.mm(z1_norm.T, z2_norm)
        
        # # Distributed synchronization
        # # IMPORTANT: all_reduce is in-place, so we work with it carefully
        # if dist.is_initialized():
        #     # dist.all_reduce(c, op=dist.ReduceOp.SUM)
        #     c = torch.distributed.nn.functional.all_reduce(c)
        #     c = c / self.batch_size  # Non-inplace division (creates new tensor)
        # else:
        #     c = c / local_batch_size  # Non-inplace division
        
        # Only call all_reduce when we actually have multiple GPUs
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if world_size > 1:
            c = torch.distributed.nn.functional.all_reduce(c)
            c = c / self.batch_size
        else:
            c = c / local_batch_size
        
        # Compute loss terms WITHOUT in-place operations
        on_diag = (torch.diagonal(c) - 1).pow(2).sum()    # subtract, not add_()
        off_diag = self.off_diagonal(c).pow(2).sum()       # pow(), not pow_()
        
        # Store for logging
        self.last_on_diag = on_diag.item()
        self.last_off_diag = off_diag.item()
        
        return on_diag + self.lambda_coeff * off_diag

class EmpSSLLoss(nn.Module):
    def __init__(self, eps=0.2, lambda_inv=200.0):
        super().__init__()
        # In the paper, epsilon^2 is set to 0.2. We treat eps as that epsilon^2.
        self.eps = eps
        self.lambda_inv = lambda_inv
        
        # For logging
        self.last_tcr = 0.0
        self.last_inv = 0.0

    def total_coding_rate(self, Z, batch_size):
        n, d = Z.shape
        I = torch.eye(d, device=Z.device)
        scalar = d / (batch_size * self.eps)
        matrix = I + scalar * (Z.T @ Z)
        
        # slogdet returns (sign, logabsdet), more stable than logdet
        sign, logabsdet = torch.linalg.slogdet(matrix)
        
        # For a covariance-like matrix, determinant should be positive
        # but numerical issues can make sign negative
        if sign.item() <= 0:
            # Fallback: add small regularization
            matrix = matrix + 1e-6 * I
            sign, logabsdet = torch.linalg.slogdet(matrix)
        
        return 0.5 * logabsdet

    def forward(self, z, num_patches):
        """
        z: [num_patches * batch_size, dim]
           (constructed by concatenating all patches for patch 0,
            then all patches for patch 1, etc.)
        """
        total_len, dim = z.shape
        batch_size = total_len // num_patches

        # Correct view ordering: [num_patches, batch_size, dim]
        z_views = z.view(num_patches, batch_size, dim)

        # 1. Expansion: TCR for each view, then average
        tcr_sum = 0.0
        for i in range(num_patches):
            tcr_sum += self.total_coding_rate(z_views[i], batch_size)
        # loss_exp = -(tcr_sum / num_patches)
        avg_tcr = tcr_sum / num_patches
        loss_exp = -avg_tcr

        # 2. Invariance: cosine similarity to the mean representation per image
        # Mean over views: [1, batch_size, dim]
        z_bar = z_views.mean(dim=0, keepdim=True)

        z_views_norm = torch.nn.functional.normalize(z_views, dim=2)
        z_bar_norm = torch.nn.functional.normalize(z_bar, dim=2)

        # Cosine similarity: [num_patches, batch_size]
        cosine_sim = (z_views_norm * z_bar_norm).sum(dim=2)
        loss_inv = (1.0 - cosine_sim).mean()

        # Store for logging (detached to avoid affecting gradients)
        self.last_tcr = avg_tcr.item()
        self.last_inv = loss_inv.item()

        return loss_exp + self.lambda_inv * loss_inv


# --- 5. DATA & TRANSFORMS ---
class Solarization:
    """Solarization augmentation mentioned in EMP-SSL / Barlow Twins"""
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            return transforms.functional.solarize(img, threshold=128)
        return img

class SSLTransformBuilder:
    """
    Base builder to create a transform pipeline with specific probabilities.
    """
    def __init__(self, size=96, blur_p=0.1, solar_p=0.1):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # Kernel size adapted for 96x96 (approx 10% of size). 
            # Standard ImageNet uses 23 for 224x224.
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=blur_p),
            Solarization(p=solar_p),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        return self.transform(x)

class BarlowTwinsTransform:
    """
    Asymmetric Transforms for Barlow Twins (following BYOL params).
    View 1: Blur p=1.0, Solarization p=0.0
    View 2: Blur p=0.1, Solarization p=0.2
    """
    def __init__(self, size=96):
        # View A: Always Blur, Never Solarize
        self.transform_a = SSLTransformBuilder(size, blur_p=1.0, solar_p=0.0)
        # View B: Weak Blur, Weak Solarization
        self.transform_b = SSLTransformBuilder(size, blur_p=0.1, solar_p=0.2)
        
    def __call__(self, x):
        return [self.transform_a(x), self.transform_b(x)]

class EmpSSLTransform:
    def __init__(self, size=96, num_patches=20):
        # Paper: "Fixed-size patch extraction... rather than RandomResizedCrop"
        # For 96x96 input, we might extract smaller patches (e.g. 32 or 48) 
        # and upsample, or crop 64 and upsample.
        # Assuming we want to force the "Bag of Features" effect:
        crop_size = 48 # Arbitrary choice, usually smaller than image
        
        self.transform = transforms.Compose([
            # 1. Fixed Size Crop (Random Location)
            transforms.RandomCrop(crop_size),
            # 2. Resize back to model input size (96)
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            
            # Standard Augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # Symmetric Augs per EMP-SSL paper
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9)], p=0.1),
            Solarization(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.num_patches = num_patches
        
    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_patches)]

class FlatFolderDataset(Dataset):
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

# --- 6. MODEL FACTORY ---
def get_model(arch, method, dim=None): # dim is flexible now
    # 1. ENCODER
    if arch == 'convnextv2_tiny':
        model = create_model('convnextv2_tiny', pretrained=False, num_classes=0)
        model.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            LayerNorm(96, eps=1e-6, data_format="channels_first")
        )
        prev_dim = 768
    elif arch == 'wideresnet':
        model = create_model('wide_resnet50_2', pretrained=False, num_classes=0)
        prev_dim = 2048
    else:
        raise ValueError("Unknown arch")

    # 2. PROJECTOR
    if method == 'barlow':
        # Barlow Twins: 3 layers, High Dim (8192 in paper, using 4096 here for 96x96 input)
        # Paper: 
        proj_dim = 4096 
        out_dim = 4096 
        projector = nn.Sequential(
            nn.Linear(prev_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False) 
        )
    elif method == 'empssl':
        # EMP-SSL: 2 layers, 4096 hidden, 512 output
        # Paper: "2 linear layers with respectively 4096 hidden units and 512 output units" 
        hidden_dim = 4096
        out_dim = 512
        projector = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False)
        )
    
    return model, projector

# --- 7. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='convnextv2_tiny', choices=['convnextv2_tiny', 'wideresnet'])
    parser.add_argument('--method', default='barlow', choices=['barlow', 'empssl'])
    parser.add_argument('--data_main', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/train')
    parser.add_argument('--data_extra', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/extra_data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100) # set to 10 for EMP SSL
    parser.add_argument('--warmup_epochs', type=int, default=10) # set to 1 for EMP SSL
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num_patches', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--output_dir', default='./checkpoints')
    parser.add_argument('--wandb_project', type=str, default='ssl-vision-backbone')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=50, help='Logging frequency (steps)')
    parser.add_argument('--max_samples', type=int, default=None, help='Debug: Limit total dataset size')
    args = parser.parse_args()
    
    rank, local_rank, world_size, is_distributed = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # WandB
    if rank == 0 and args.wandb_project and HAS_WANDB:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Create Model
    model, projector = get_model(args.arch, args.method)
    
    # --- CHECKS (Rank 0 only) ---
    assert_random_init(model, rank)
    print_parameter_count(model, projector, rank)
    # ----------------------------

    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # projector = nn.SyncBatchNorm.convert_sync_batchnorm(projector)
    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        projector = nn.SyncBatchNorm.convert_sync_batchnorm(projector)
    model.cuda(local_rank)
    projector.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    # projector = torch.nn.parallel.DistributedDataParallel(projector, device_ids=[local_rank])
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        projector = torch.nn.parallel.DistributedDataParallel(projector, device_ids=[local_rank])
    
    # Load Data
    if args.method == 'barlow':
        tfm = BarlowTwinsTransform(size=96)
    else:
        tfm = EmpSSLTransform(size=96, num_patches=args.num_patches)
        
    ds1 = FlatFolderDataset(args.data_main, transform=tfm)
    ds2 = FlatFolderDataset(args.data_extra, transform=tfm)
    full_ds = ConcatDataset([ds1, ds2])

    if args.max_samples is not None and args.max_samples < len(full_ds):
        # Create a subset of indices [0, 1, ... max_samples-1]
        indices = list(range(args.max_samples))
        full_ds = torch.utils.data.Subset(full_ds, indices)
        if rank == 0:
            print(f" DEBUG MODE: Truncating dataset to {len(full_ds)} samples.")
    
    # sampler = DistributedSampler(full_ds, shuffle=True)
    if is_distributed:
        sampler = DistributedSampler(full_ds, shuffle=True)
    else:
        sampler = None  # DataLoader will use default RandomSampler
    loader = DataLoader(
        full_ds, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # Calculate LR
    effective_bs = args.batch_size * world_size
    # Base LR 0.2 (safer than 0.3)
    # if args.lr is None:
    #     args.lr = 0.2 * (effective_bs / 256.0)
    base_lr = 0.2 * (effective_bs / 256.0)
    args.lr = base_lr
    
    bn_bias_lr_scale = 0.024
    weight_decay_backbone = 1.5e-6 if args.method == 'barlow' else 1e-4

    param_groups = []

    # Backbone / encoder
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = ('bias' in name) or ('bn' in name) or ('norm' in name)
        if is_norm_or_bias:
            # Exclude from LARS & WD, smaller LR
            lr_scale = bn_bias_lr_scale
            param_groups.append({
                'params': [p],
                'weight_decay_filter': exclude_from_wd_and_lars,
                'larc_adaptation_filter': exclude_from_wd_and_lars,
                'lr': base_lr * lr_scale,
                'lr_scale': lr_scale,
                'weight_decay': 0.0,
                'momentum': 0.9,
            })
        else:
            lr_scale = 1.0
            param_groups.append({
                'params': [p],
                'weight_decay_filter': include_in_wd_and_lars,
                'larc_adaptation_filter': include_in_wd_and_lars,
                'lr': base_lr * lr_scale,
                'lr_scale': lr_scale,
                'weight_decay': weight_decay_backbone,
                'momentum': 0.9,
            })
    
    # Projector (same pattern)
    for name, p in projector.named_parameters():
        if not p.requires_grad:
            continue
        is_norm_or_bias = ('bias' in name) or ('bn' in name) or ('norm' in name)
        if is_norm_or_bias:
            lr_scale = bn_bias_lr_scale
            param_groups.append({
                'params': [p],
                'weight_decay_filter': exclude_from_wd_and_lars,
                'larc_adaptation_filter': exclude_from_wd_and_lars,
                'lr': base_lr * lr_scale,
                'lr_scale': lr_scale,
                'weight_decay': 0.0,
                'momentum': 0.9,
            })
        else:
            lr_scale = 1.0
            param_groups.append({
                'params': [p],
                'weight_decay_filter': include_in_wd_and_lars,
                'larc_adaptation_filter': include_in_wd_and_lars,
                'lr': base_lr * lr_scale,
                'lr_scale': lr_scale,
                'weight_decay': weight_decay_backbone,
                'momentum': 0.9,
            })

    optimizer = LARS(param_groups, lr=base_lr)

    wd = weight_decay_backbone


    # params = list(model.parameters()) + list(projector.parameters())
    # optimizer = LARS(params, lr=args.lr, weight_decay=wd)

    if rank == 0:
        print(f"Effective Batch Size: {effective_bs}")
        print(f"Max LR (LARS): {args.lr:.4f}")
        print(f"Weight Decay: {wd} (Specific to {args.method})")
        if args.method == 'barlow':
            print("Augmentations: Asymmetric (View A: Blur=1.0/Sol=0.0, View B: Blur=0.1/Sol=0.2)")
        else:
            print("Augmentations: Symmetric (Blur=0.1, Sol=0.1)")
    
    if args.method == 'barlow': criterion = BarlowTwinsLoss(effective_bs)
    else: criterion = EmpSSLLoss()
        
    args.steps_per_epoch = len(loader)
    total_steps = args.epochs * args.steps_per_epoch

    if rank == 0:
        print(f"Effective Batch Size: {effective_bs}")
        print(f"Max LR (LARS): {args.lr:.4f}")

    start_epoch = 0
    global_step = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"rank {rank}: ==> Loading checkpoint '{args.resume}'")
            # Map location is crucial for distributed training to avoid GPU overload on rank 0
            checkpoint = torch.load(args.resume, map_location=f'cuda:{local_rank}')
            
            # 1. Load Start Epoch
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint.get('global_step', start_epoch * len(loader))
            
            # 2. Load Weights (Handle DDP 'module.' prefix if needed, though strictly saving/loading 'module' is standard)
            # model.module.load_state_dict(checkpoint['model_state'])
            # projector.module.load_state_dict(checkpoint['projector_state'])
            if is_distributed:
                model.module.load_state_dict(checkpoint['model_state'])
                projector.module.load_state_dict(checkpoint['projector_state'])
            else:
                model.load_state_dict(checkpoint['model_state'])
                projector.load_state_dict(checkpoint['projector_state'])
            
            # 3. Load Optimizer
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            print(f"rank {rank}: ==> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"rank {rank}: ==> No checkpoint found at '{args.resume}'")
    
    total_training_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        # sampler.set_epoch(epoch)
        if is_distributed:
            sampler.set_epoch(epoch)
        model.train()
        projector.train()
        
        for step, inputs in enumerate(loader):
            curr_lr = adjust_learning_rate(optimizer, global_step, total_steps, args)
            optimizer.zero_grad()
            
            # if args.method == 'barlow':
            #     x1, x2 = inputs[0].cuda(local_rank, non_blocking=True), inputs[1].cuda(local_rank, non_blocking=True)
            #     z1, z2 = projector(model(x1)), projector(model(x2))
            #     loss = criterion(z1, z2)
            if args.method == 'barlow':
                x1 = inputs[0].cuda(local_rank, non_blocking=True)
                x2 = inputs[1].cuda(local_rank, non_blocking=True)
                
                # Concatenate both views and process together
                x_combined = torch.cat([x1, x2], dim=0)
                z_combined = projector(model(x_combined))
                
                # Split the output back into two views
                z1, z2 = z_combined.chunk(2, dim=0)
                
                loss = criterion(z1, z2)
            else:
                inputs_flat = torch.cat(inputs, dim=0).cuda(local_rank, non_blocking=True)
                z = projector(model(inputs_flat))
                loss = criterion(z, args.num_patches)
                
            loss.backward()
            optimizer.step()
            global_step += 1
            
            if step % args.log_freq == 0 and rank == 0:
                if args.method == 'barlow':
                    print(f"Epoch [{epoch}] Step [{step}] "
                          f"Loss: {loss.item():.4f} | "
                          f"OnDiag: {criterion.last_on_diag:.4f} | "
                          f"OffDiag: {criterion.last_off_diag:.4f} | "
                          f"LR: {curr_lr:.6f}")
                    if args.wandb_project and HAS_WANDB:
                        wandb.log({
                            "loss": loss.item(), 
                            "on_diag": criterion.last_on_diag,
                            "off_diag": criterion.last_off_diag,
                            "lr": curr_lr, 
                            "epoch": epoch
                        })
                else:
                    print(f"Epoch [{epoch}] Step [{step}] "
                          f"Loss: {loss.item():.4f} | "
                          f"TCR: {criterion.last_tcr:.4f} | "
                          f"Inv: {criterion.last_inv:.4f} | "
                          f"LR: {curr_lr:.6f}")
                    if args.wandb_project and HAS_WANDB:
                        wandb.log({
                            "loss": loss.item(), 
                            "tcr": criterion.last_tcr, 
                            "inv": criterion.last_inv, 
                            "lr": curr_lr, 
                            "epoch": epoch
                        })

        epoch_time = time.time() - start_time
        if rank == 0 and HAS_WANDB:
            wandb.log({
                "epoch_duration": epoch_time, 
                "epoch": epoch
            })
            # Optional: Log total time elapsed so far
            wandb.log({"total_time_elapsed": time.time() - total_training_start})

        # Save Checkpoint
        if rank == 0:
            # Create the checkpoint dictionary
            checkpoint_state = {
                'epoch': epoch,
                'model_state': model.module.state_dict() if is_distributed else model.state_dict(),
                'projector_state': projector.module.state_dict() if is_distributed else projector.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'args': vars(args),
                'global_step': global_step
            }

            # 1. Save "Latest" (Always overwrite)
            latest_path = os.path.join(args.output_dir, f"{args.method}_{args.arch}_latest.pth")
            torch.save(checkpoint_state, latest_path)
            
            # 2. Save History (Per-epoch or specific frequency)
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                history_path = os.path.join(args.output_dir, f"{args.method}_{args.arch}_epoch{epoch}.pth")
                torch.save(checkpoint_state, history_path)
                print(f"Saved Checkpoints: {latest_path} (Latest) and {history_path} (History)")
            else:
                 print(f"Saved Checkpoint: {latest_path} (Latest)")

    if rank == 0:
        if args.wandb_project and HAS_WANDB: wandb.finish()
        print("Done.")

if __name__ == "__main__":
    main()