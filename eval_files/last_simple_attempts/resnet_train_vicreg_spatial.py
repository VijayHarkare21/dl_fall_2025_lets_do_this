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
from pathlib import Path

import resnet

# Try importing wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    
torch.autograd.set_detect_anomaly(True)

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
        param_group['lr'] = lr

    return lr

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class VicRegLoss(nn.Module):
    def __init__(self, args, effective_bs):
        super().__init__()
        self.args = args
        self.num_features = 4096
        self.effective_bs = effective_bs

        self.last_repr_loss = 0.0
        self.last_std_loss = 0.0
        self.last_cov_loss = 0.0

    def forward(self, x, y):

        repr_loss = torch.nn.functional.mse_loss(x, y)

        if dist.is_available() and dist.is_initialized():
            x = torch.cat(FullGatherLayer.apply(x), dim=0)
            y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(torch.nn.functional.relu(1 - std_x)) / 2 + torch.mean(torch.nn.functional.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.effective_bs - 1)
        cov_y = (y.T @ y) / (self.effective_bs - 1)
        cov_loss = off_diagonal(cov_x).pow(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        self.last_repr_loss = repr_loss.item()
        self.last_std_loss = std_loss.item()
        self.last_cov_loss = cov_loss.item()
        return loss


from PIL import ImageOps, ImageFilter
import numpy as np
from torchvision.transforms import InterpolationMode


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                Solarization(p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return [x1, x2]

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

def Projector(mlp, embedding):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

# --- 6. MODEL FACTORY ---
def get_model(arch, method, dim=None): # dim is flexible now
    # 1. ENCODER
    if arch == 'resnet50':
        model, prev_dim = resnet.__dict__[arch](
            zero_init_residual=True
        )
        model.maxpool = nn.Identity()
        # prev_dim = 2048

    # 2. PROJECTOR
    if method == 'vicreg':
        projector = Projector(mlp="4096-4096-4096", embedding=prev_dim)
    
    return model, projector

def exclude_bias_and_norm(p):
    return p.ndim == 1

# --- 7. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='resnet50', choices=['resnet50'])
    parser.add_argument('--method', default='vicreg', choices=['vicreg'])
    parser.add_argument('--data_main', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/train')
    parser.add_argument('--main_max_samples', type=int, default=0)
    parser.add_argument('--train_data_type', default='birds', choices=['birds', 'scenes', 'general'])
    parser.add_argument('--birds_1', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/inat_new')
    parser.add_argument('--birds_2', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/nabirds')
    parser.add_argument('--birds_3', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/birds525')
    parser.add_argument('--birds_4', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/birdsnap')
    parser.add_argument('--scenes_data', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/places_new')
    parser.add_argument('--general_data', default='/scratch/vjh9526/dl-fall-2025/pretrain_data/all_data/openimages_new')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--base_lr', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--output_dir', default='/scratch/vjh9526/dl-fall-2025/checkpoints_resnet_vicreg')
    parser.add_argument('--wandb_project', type=str, default='ssl-vision-backbone')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save frequency (epochs)')
    parser.add_argument('--log_freq', type=int, default=50, help='Logging frequency (steps)')
    parser.add_argument('--max_samples', type=int, default=None, help='Debug: Limit total dataset size')
    
    parser.add_argument("--sim_coeff", type=float, default=15.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std_coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov_coeff", type=float, default=2.0,
                        help='Covariance regularization loss coefficient')
    
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    
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
    if args.method == 'vicreg':
        tfm = TrainTransform()
        
    ds_lists = []
    if args.main_max_samples > 0:
        ds_main = torch.utils.data.Subset(FlatFolderDataset(args.data_main, transform=tfm), list(range(args.main_max_samples)))
        ds_lists.append(ds_main)

    if args.train_data_type == "birds":
        ds1 = FlatFolderDataset(args.birds_1, transform=tfm)
        ds2 = FlatFolderDataset(args.birds_2, transform=tfm)
        ds3 = FlatFolderDataset(args.birds_3, transform=tfm)
        ds4 = FlatFolderDataset(args.birds_4, transform=tfm)
        ds_lists.extend([ds1, ds2, ds3, ds4])
    elif args.train_data_type == "scenes":
        ds1 = FlatFolderDataset(args.scenes_data, transform=tfm)
        ds_lists.append(ds1)
    elif args.train_data_type == "general":
        ds1 = FlatFolderDataset(args.general_data, transform=tfm)
        ds_lists.append(ds1)
    else:
        raise ValueError(f"Unknown train_data_type: {args.train_data_type}")

    full_ds = torch.utils.data.ConcatDataset(ds_lists)

    if args.max_samples is not None and args.max_samples < len(full_ds):
        # Create a subset of indices [0, 1, ... max_samples-1]
        indices = list(range(args.max_samples))
        full_ds = torch.utils.data.Subset(full_ds, indices)
        if rank == 0:
            print(f" DEBUG MODE: Truncating dataset to {len(full_ds)} samples.")
    
    print(f"Dataset size: {len(full_ds)}")
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

    base_lr = args.base_lr * (effective_bs / 256.0)
    args.lr = base_lr
    
    weight_decay_backbone = args.wd
    
    # use model and projector parameters in optimizer
    params = list(model.parameters()) + list(projector.parameters())
    optimizer = LARS(
        params,
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    wd = weight_decay_backbone


    # params = list(model.parameters()) + list(projector.parameters())
    # optimizer = LARS(params, lr=args.lr, weight_decay=wd)

    if rank == 0:
        print(f"Effective Batch Size: {effective_bs}")
        print(f"Max LR (LARS): {args.lr:.4f}")
        print(f"Weight Decay: {wd} (Specific to {args.method})")
        if args.method == 'vicreg':
            print("Augmentations: Standard, according to official implementation")
    
    if args.method == 'vicreg': criterion = VicRegLoss(args, effective_bs)
        
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
            
            if args.method == 'vicreg':
                x1 = inputs[0].cuda(local_rank, non_blocking=True)
                x2 = inputs[1].cuda(local_rank, non_blocking=True)
                
                # Concatenate both views and process together
                z1 = projector(model(x1))
                z2 = projector(model(x2))
                
                if rank == 0 and step % args.log_freq == 0:
                    with torch.no_grad():
                        print(
                            f"Feat norms: z1 {z1.detach().norm(dim=1).mean():.3f}, "
                            f"z2 {z2.detach().norm(dim=1).mean():.3f}"
                        )
                
                loss = criterion(z1, z2)
                
            loss.backward()
            optimizer.step()
            global_step += 1
            
            if step % args.log_freq == 0 and rank == 0:
                if args.method == 'vicreg':
                    print(f"Epoch [{epoch}] Step [{step}] "
                          f"Loss: {loss.item():.4f} | "
                          f"Repr Loss: {criterion.last_repr_loss:.4f} | "
                          f"Std Loss: {criterion.last_std_loss:.4f} | "
                          f"Cov Loss: {criterion.last_cov_loss:.4f}")
                    if args.wandb_project and HAS_WANDB:
                        wandb.log({
                            "loss": loss.item(), 
                            "repr_loss": criterion.last_repr_loss,
                            "std_loss": criterion.last_std_loss,
                            "cov_loss": criterion.last_cov_loss,
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