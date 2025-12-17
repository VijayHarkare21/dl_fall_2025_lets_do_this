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
from torch import optim
import torch.nn.functional as F

import resnet

from pathlib import Path

import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

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

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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

def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass

# --- 3. OPTIMIZER (LARS) & SCHEDULER ---
class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss

def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

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

# --- 7. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='resnet50', choices=['resnet50'])
    parser.add_argument('--method', default='vicreg', choices=['vicreg'])
    parser.add_argument('--data_main', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/train')
    parser.add_argument('--main_max_samples', type=int, default=100000)
    parser.add_argument('--train_data_type', default='birds', choices=['birds', 'scenes', 'general'])
    parser.add_argument('--birds_1', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/inat_new')
    parser.add_argument('--birds_2', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/nabirds_new')
    parser.add_argument('--birds_3', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/birds525')
    parser.add_argument('--birds_4', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/birdsnap')
    parser.add_argument('--scenes_data', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/places_new')
    parser.add_argument('--general_data', type=Path, default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/openimages_new')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--output_dir', type=Path, default='./checkpoints')
    parser.add_argument('--wandb_project', type=str, default='ssl-vision-backbone')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=50, help='Logging frequency (steps)')
    parser.add_argument('--max_samples', type=int, default=None, help='Debug: Limit total dataset size')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    if args.rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True) 
    # WandB
    if args.rank == 0 and args.wandb_project and HAS_WANDB:
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