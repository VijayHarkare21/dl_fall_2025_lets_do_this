"""
Distributed Training Utilities
==============================

This module provides utilities for distributed data parallel (DDP) training
using PyTorch's native distributed package. The setup is designed to work
seamlessly with torchrun, which handles process spawning and environment
variable setup automatically.

Usage with torchrun:
    # Single GPU
    python scripts/train.py --config configs/dinov2.yaml
    
    # Multi-GPU (e.g., 2 GPUs)
    torchrun --nproc_per_node=2 scripts/train.py --config configs/dinov2.yaml
    
    # Multi-node (advanced)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d ...

Key Concepts:
- World Size: Total number of processes (GPUs) across all nodes
- Rank: Global process ID (0 to world_size-1)
- Local Rank: Process ID within current node (0 to nproc_per_node-1)
- Process Group: Communication group for collective operations

The code automatically detects whether it's running in distributed mode
by checking for torchrun environment variables.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def is_distributed() -> bool:
    """
    Check if running in distributed mode.
    
    torchrun sets several environment variables that we can check.
    Returns True if any of the standard distributed env vars are set.
    """
    return (
        os.environ.get('WORLD_SIZE', None) is not None or
        os.environ.get('RANK', None) is not None or
        os.environ.get('LOCAL_RANK', None) is not None
    )


def get_world_size() -> int:
    """
    Get total number of processes in the distributed group.
    
    Returns 1 if not in distributed mode.
    """
    if is_distributed():
        # First check if process group is initialized
        if dist.is_initialized():
            return dist.get_world_size()
        # Fall back to environment variable (before init_process_group)
        return int(os.environ.get('WORLD_SIZE', 1))
    return 1


def get_rank() -> int:
    """
    Get global rank of current process.
    
    Returns 0 if not in distributed mode (single process is rank 0).
    """
    if is_distributed():
        # First check if process group is initialized
        if dist.is_initialized():
            return dist.get_rank()
        # Fall back to environment variable (before init_process_group)
        return int(os.environ.get('RANK', 0))
    return 0


def get_local_rank() -> int:
    """
    Get local rank (GPU index on current node).
    
    This is used to assign each process to a specific GPU.
    Returns 0 if not in distributed mode.
    """
    if is_distributed():
        return int(os.environ.get('LOCAL_RANK', 0))
    return 0


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0).
    
    Use this to guard operations that should only happen once:
    - Logging to W&B
    - Saving checkpoints
    - Printing progress
    - Evaluation
    """
    return get_rank() == 0


def setup_distributed() -> None:
    """
    Initialize the distributed process group.
    
    This should be called at the start of training before creating
    any models or data loaders. torchrun sets the necessary environment
    variables (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK).
    
    The function:
    1. Detects if running in distributed mode
    2. Sets the CUDA device for this process
    3. Initializes the NCCL backend for GPU communication
    """
    if not is_distributed():
        print("[Distributed] Running in single-GPU mode")
        return
    
    # Get distributed info from environment
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group with NCCL backend (optimized for GPU)
    # Specify device_id to avoid PyTorch warning about guessing device
    dist.init_process_group(
        backend='nccl',
        init_method='env://',  # Use environment variables set by torchrun
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f'cuda:{local_rank}'),
    )
    
    # Synchronize all processes before proceeding
    dist.barrier()
    
    if is_main_process():
        print(f"[Distributed] Initialized: world_size={world_size}, backend=nccl")


def cleanup_distributed() -> None:
    """
    Clean up the distributed process group.
    
    Call this at the end of training to properly shut down
    distributed communication.
    """
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()
        if is_main_process():
            print("[Distributed] Process group destroyed")


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average a tensor across all processes.
    
    Useful for aggregating metrics like loss across GPUs.
    Each process contributes its local value, and all processes
    receive the mean.
    
    Args:
        tensor: Local tensor to average (will be modified in-place)
    
    Returns:
        Tensor containing the mean across all processes
    """
    if not is_distributed() or not dist.is_initialized():
        return tensor
    
    # Clone to avoid modifying the original
    tensor = tensor.clone()
    
    # Sum across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # Divide by world size to get mean
    tensor /= get_world_size()
    
    return tensor


def all_gather_tensors(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes and concatenate.
    
    Useful for gathering features or predictions from all GPUs
    for evaluation (e.g., k-NN needs all features together).
    
    Args:
        tensor: Local tensor of shape (N, ...) where N can vary per process
    
    Returns:
        Concatenated tensor from all processes (total_N, ...)
    """
    if not is_distributed() or not dist.is_initialized():
        return tensor
    
    world_size = get_world_size()
    
    # Get tensor size from all processes (they may differ)
    local_size = torch.tensor([tensor.shape[0]], device=tensor.device)
    sizes_list = [torch.zeros(1, dtype=torch.long, device=tensor.device) 
                  for _ in range(world_size)]
    dist.all_gather(sizes_list, local_size)
    sizes = [int(s.item()) for s in sizes_list]
    max_size = max(sizes)
    
    # Pad tensor to max size for gathering
    if tensor.shape[0] < max_size:
        padding = torch.zeros(
            max_size - tensor.shape[0], *tensor.shape[1:],
            dtype=tensor.dtype, device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=0)
    
    # Gather padded tensors from all processes
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    
    # Remove padding and concatenate
    gathered = [g[:s] for g, s in zip(gathered, sizes)]
    return torch.cat(gathered, dim=0)


def broadcast_object(obj, src: int = 0):
    """
    Broadcast a Python object from source rank to all processes.
    
    Useful for sharing configuration, random seeds, or other
    small objects that need to be consistent across processes.
    
    Args:
        obj: Python object to broadcast (only used on src rank)
        src: Source rank to broadcast from
    
    Returns:
        The broadcasted object (same on all processes)
    """
    if not is_distributed() or not dist.is_initialized():
        return obj
    
    # Wrap in list for broadcast_object_list
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def reduce_dict(metrics_dict: dict, average: bool = True) -> dict:
    """
    Reduce a dictionary of metrics across all processes.
    
    Convenient wrapper for reducing multiple metrics at once.
    
    Args:
        metrics_dict: Dictionary of metric_name -> tensor
        average: If True, compute mean; if False, compute sum
    
    Returns:
        Dictionary with reduced metrics
    """
    if not is_distributed() or not dist.is_initialized():
        return metrics_dict
    
    world_size = get_world_size()
    reduced = {}
    
    for key, value in metrics_dict.items():
        if isinstance(value, torch.Tensor):
            tensor = value.clone()
        else:
            tensor = torch.tensor(value, device='cuda')
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        if average:
            tensor /= world_size
        
        reduced[key] = tensor.item() if tensor.numel() == 1 else tensor
    
    return reduced


class DistributedSampler:
    """
    Wrapper info for torch.utils.data.distributed.DistributedSampler.
    
    This is just documentation - use PyTorch's built-in DistributedSampler:
    
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        
        # Important: call set_epoch() at the start of each epoch
        # to ensure proper shuffling across epochs
        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            for batch in dataloader:
                ...
    
    The sampler ensures each GPU sees a different subset of the data,
    and drop_last=True ensures all GPUs have the same number of batches.
    """
    pass


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Distributed Utilities Test")
    print("=" * 60)
    
    print("\n--- Environment Check (before setup) ---")
    print(f"Is distributed: {is_distributed()}")
    print(f"World size: {get_world_size()}")
    print(f"Rank: {get_rank()}")
    print(f"Local rank: {get_local_rank()}")
    print(f"Is main process: {is_main_process()}")
    
    # Setup distributed (should be called early in real training)
    print("\n--- Setup Distributed ---")
    setup_distributed()
    
    print("\n--- Environment Check (after setup) ---")
    print(f"World size: {get_world_size()}")
    print(f"Rank: {get_rank()}")
    print(f"Local rank: {get_local_rank()}")
    print(f"Is main process: {is_main_process()}")
    
    # Test collective operations
    print("\n--- Collective Operations Test ---")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{get_local_rank()}')
    else:
        device = torch.device('cpu')
    
    # Each process has a different tensor value based on rank
    tensor = torch.tensor([float(get_rank() + 1)], device=device)
    print(f"[Rank {get_rank()}] Before all_reduce: {tensor}")
    
    reduced = all_reduce_mean(tensor)
    print(f"[Rank {get_rank()}] After all_reduce_mean: {reduced}")
    
    # Test all_gather
    local_tensor = torch.tensor([get_rank() * 10, get_rank() * 10 + 1], device=device)
    gathered = all_gather_tensors(local_tensor)
    if is_main_process():
        print(f"[Rank {get_rank()}] Gathered tensors: {gathered}")
    
    # Test broadcast
    if get_rank() == 0:
        obj = {'key': 'value', 'num': 42}
    else:
        obj = None
    broadcasted = broadcast_object(obj, src=0)
    print(f"[Rank {get_rank()}] Broadcasted object: {broadcasted}")
    
    # Test reduce_dict
    metrics = {
        'loss': torch.tensor(0.5 + get_rank() * 0.1, device=device),
        'acc': torch.tensor(0.8 + get_rank() * 0.05, device=device)
    }
    reduced_metrics = reduce_dict(metrics)
    if is_main_process():
        print(f"[Rank {get_rank()}] Reduced metrics: {reduced_metrics}")
    
    # Cleanup
    print("\n--- Cleanup ---")
    cleanup_distributed()
    
    if get_rank() == 0:  # Use env var since process group is destroyed
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        print("\nTo test distributed mode, run with torchrun:")
        print("  torchrun --nproc_per_node=2 utils/distributed.py")