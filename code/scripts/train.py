"""
Main Training Script for Self-Supervised Learning
==================================================

This is the main entry point for training SSL models. It handles:
- Command-line argument parsing
- Configuration loading from YAML files
- Trainer instantiation (DINOv2 or MoCo v3)
- Training execution

Usage:
    # Single GPU
    python scripts/train.py --config configs/dinov2_vit_small.yaml
    
    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 scripts/train.py --config configs/dinov2_vit_small.yaml
    
    # Override config values via CLI
    python scripts/train.py --config configs/dinov2_vit_small.yaml --lr 1e-4 --batch_size 64
    
    # Quick test with small data
    python scripts/train.py --config configs/dinov2_vit_small.yaml --max_samples 1000 --epochs 2
    
    # DINOv2 with iBOT disabled (DINO v1 style)
    python scripts/train.py --data_dir /path/to/data --method dinov2 --no_ibot

The script supports both YAML configuration files and command-line overrides.
CLI arguments take precedence over config file values.
"""

import argparse
import os
import sys
import yaml
import random
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trainers.dinov2_trainer import DINOv2Trainer
from trainers.moco_trainer import MoCoV3Trainer
from utils.distributed import is_main_process, get_rank, get_world_size


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SSL models (DINOv2, MoCo v3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML configuration file'
    )
    
    # Required arguments (if no config file)
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to training data directory'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs',
        help='Path to save checkpoints and logs'
    )
    
    # Method selection
    parser.add_argument(
        '--method', type=str, default='dinov2',
        choices=['dinov2', 'moco'],
        help='SSL method to use'
    )
    
    # Model configuration
    parser.add_argument(
        '--backbone', type=str, default='vit_small',
        choices=['vit_small', 'vit_tiny', 'convnext_tiny', 'convnext_small', 'convnext_base'],
        help='Backbone architecture'
    )
    parser.add_argument(
        '--img_size', type=int, default=96,
        help='Input image size'
    )
    parser.add_argument(
        '--patch_size', type=int, default=8,
        help='Patch size for ViT'
    )
    
    # CFM configuration
    parser.add_argument(
        '--use_cfm', action='store_true',
        help='Enable Conditional Feature Modulation'
    )
    parser.add_argument(
        '--no_cfm', action='store_true',
        help='Disable Conditional Feature Modulation'
    )
    parser.add_argument(
        '--cfm_context_dim', type=int, default=256,
        help='CFM context vector dimension'
    )
    parser.add_argument(
        '--cfm_hidden_dim', type=int, default=128,
        help='CFM hidden dimension'
    )
    parser.add_argument(
        '--cfm_curriculum_start', type=float, default=0.2,
        help='CFM curriculum start (fraction of total epochs, e.g., 0.2 = 20%%)'
    )
    parser.add_argument(
        '--cfm_curriculum_full', type=float, default=0.6,
        help='CFM curriculum full strength (fraction of total epochs, e.g., 0.6 = 60%%)'
    )
    parser.add_argument(
        '--no_cfm_curriculum', action='store_true',
        help='Disable CFM curriculum (train CFM from start)'
    )
    
    # Training configuration
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size per GPU'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Base learning rate'
    )
    parser.add_argument(
        '--min_lr', type=float, default=1e-6,
        help='Minimum learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.04,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=10,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--max_grad_norm', type=float, default=1.0,
        help='Maximum gradient norm for clipping'
    )
    
    # DINOv2-specific
    parser.add_argument(
        '--n_global_crops', type=int, default=2,
        help='Number of global crops (DINOv2)'
    )
    parser.add_argument(
        '--n_local_crops', type=int, default=6,
        help='Number of local crops (DINOv2)'
    )
    parser.add_argument(
        '--global_crop_size', type=int, default=96,
        help='Global crop size'
    )
    parser.add_argument(
        '--local_crop_size', type=int, default=48,
        help='Local crop size'
    )
    parser.add_argument(
        '--student_temp', type=float, default=0.1,
        help='Student temperature (DINOv2)'
    )
    parser.add_argument(
        '--teacher_temp', type=float, default=0.04,
        help='Teacher temperature (DINOv2)'
    )
    parser.add_argument(
        '--center_momentum', type=float, default=0.9,
        help='Center momentum (DINOv2)'
    )
    
    # DINOv2 iBOT (masked image modeling)
    parser.add_argument(
        '--use_ibot', action='store_true', default=True,
        help='Use iBOT masked patch prediction (DINOv2)'
    )
    parser.add_argument(
        '--no_ibot', action='store_true',
        help='Disable iBOT (use DINO v1 style, CLS token only)'
    )
    parser.add_argument(
        '--ibot_loss_weight', type=float, default=1.0,
        help='Weight for iBOT loss'
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.3,
        help='Ratio of patches to mask for iBOT (0.0 to 1.0)'
    )
    
    # DINOv2 KoLeo regularizer
    parser.add_argument(
        '--use_koleo', action='store_true', default=True,
        help='Use KoLeo regularizer for uniform feature distribution'
    )
    parser.add_argument(
        '--no_koleo', action='store_true',
        help='Disable KoLeo regularizer'
    )
    parser.add_argument(
        '--koleo_loss_weight', type=float, default=0.1,
        help='Weight for KoLeo loss'
    )
    
    # DINOv2 loss weights
    parser.add_argument(
        '--dino_loss_weight', type=float, default=1.0,
        help='Weight for DINO CLS token loss'
    )
    
    # MoCo v3-specific
    parser.add_argument(
        '--temperature', type=float, default=0.2,
        help='InfoNCE temperature (MoCo v3)'
    )
    
    # Common SSL parameters
    parser.add_argument(
        '--ema_momentum', type=float, default=0.996,
        help='EMA momentum for teacher/key encoder'
    )
    
    # Head configuration
    parser.add_argument(
        '--head_out_dim', type=int, default=16384,
        help='Output dimension of projection head (DINOv2 prototypes)'
    )
    parser.add_argument(
        '--head_hidden_dim', type=int, default=1024,
        help='Hidden dimension of projection head'
    )
    parser.add_argument(
        '--head_bottleneck_dim', type=int, default=256,
        help='Bottleneck dimension (DINOv2)'
    )
    
    # Data loading
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--max_samples', type=int, default=None,
        help='Limit number of training samples (for debugging)'
    )
    
    # Logging and checkpointing
    parser.add_argument(
        '--wandb_project', type=str, default='ssl-vision-backbone',
        help='Weights & Biases project name'
    )
    parser.add_argument(
        '--run_name', type=str, default=None,
        help='Run name (auto-generated if not provided)'
    )
    parser.add_argument(
        '--use_wandb', action='store_true', default=True,
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--no_wandb', action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--log_every', type=int, default=50,
        help='Log every N steps'
    )
    parser.add_argument(
        '--save_every', type=int, default=10,
        help='Save checkpoint every N epochs'
    )
    
    # Resumption
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--auto_resume', action='store_true', default=True,
        help='Automatically resume from latest checkpoint'
    )
    parser.add_argument(
        '--no_auto_resume', action='store_true',
        help='Disable automatic resumption'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config_and_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration from file with command-line arguments.
    CLI arguments take precedence over config file values.
    """
    # Start with config file values
    merged = dict(config)
    
    # Override with CLI arguments (only if explicitly provided)
    args_dict = vars(args)
    
    for key, value in args_dict.items():
        # Skip None values (not provided) and the config path itself
        if value is None or key == 'config':
            continue
        
        # Handle special boolean flags
        if key == 'no_cfm' and value:
            merged['use_cfm'] = False
            continue
        if key == 'no_wandb' and value:
            merged['use_wandb'] = False
            continue
        if key == 'no_auto_resume' and value:
            merged['auto_resume'] = False
            continue
        if key == 'no_ibot' and value:
            merged['use_ibot'] = False
            continue
        if key == 'no_koleo' and value:
            merged['use_koleo'] = False
            continue
        
        # Skip negative flag keys
        if key.startswith('no_'):
            continue
        
        # Override config value
        merged[key] = value
    
    return merged


def build_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration dictionary from command-line arguments only."""
    config = vars(args).copy()
    
    # Remove the config path
    config.pop('config', None)
    
    # Handle boolean flags
    if config.get('no_cfm'):
        config['use_cfm'] = False
    if config.get('no_wandb'):
        config['use_wandb'] = False
    if config.get('no_auto_resume'):
        config['auto_resume'] = False
    if config.get('no_ibot'):
        config['use_ibot'] = False
    if config.get('no_koleo'):
        config['use_koleo'] = False
    if config.get('no_cfm_curriculum'):
        config['cfm_curriculum_disabled'] = True
    
    # Remove negative flags
    for key in list(config.keys()):
        if key.startswith('no_'):
            del config[key]
    
    # Build nested config for head
    config['head'] = {
        'out_dim': config.pop('head_out_dim', 16384),
        'hidden_dim': config.pop('head_hidden_dim', 1024),
        'bottleneck_dim': config.pop('head_bottleneck_dim', 256),
    }
    
    # Build nested config for CFM with curriculum settings
    cfm_curriculum_start = config.pop('cfm_curriculum_start', 0.2)
    cfm_curriculum_full = config.pop('cfm_curriculum_full', 0.6)
    
    # If CFM curriculum was disabled, set start and full to 0
    if config.pop('cfm_curriculum_disabled', False):
        cfm_curriculum_start = 0.0
        cfm_curriculum_full = 0.0
    
    config['cfm'] = {
        'context_dim': config.pop('cfm_context_dim', 256),
        'hidden_dim': config.pop('cfm_hidden_dim', 128),
        'input_size': config.get('local_crop_size', 48),
        'curriculum_start': cfm_curriculum_start,
        'curriculum_full': cfm_curriculum_full,
    }
    
    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For full determinism (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def validate_config(config: Dict[str, Any]):
    """Validate configuration and check required fields."""
    required_fields = ['data_dir']
    
    for field in required_fields:
        if field not in config or config[field] is None:
            raise ValueError(f"Required configuration field missing: {field}")
    
    # Check data directory exists
    data_dir = Path(config['data_dir'])
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    # Validate method
    method = config.get('method', 'dinov2')
    if method not in ['dinov2', 'moco']:
        raise ValueError(f"Unknown method: {method}. Choose from: dinov2, moco")
    
    # Validate backbone
    backbone = config.get('backbone', 'vit_small')
    valid_backbones = ['vit_small', 'vit_tiny', 'convnext_tiny', 'convnext_small', 'convnext_base']
    if backbone not in valid_backbones:
        raise ValueError(f"Unknown backbone: {backbone}. Choose from: {', '.join(valid_backbones)}")
    
    # Validate mask_ratio
    mask_ratio = config.get('mask_ratio', 0.3)
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be between 0 and 1, got {mask_ratio}")


def generate_run_name(config: Dict[str, Any]) -> str:
    """Generate a descriptive run name."""
    method = config.get('method', 'dinov2')
    backbone = config.get('backbone', 'vit_small')
    use_cfm = config.get('use_cfm', False)
    use_ibot = config.get('use_ibot', True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    parts = [method, backbone]
    if use_cfm:
        parts.append("cfm")
    if method == 'dinov2' and not use_ibot:
        parts.append("no_ibot")
    
    return "_".join(parts) + f"_{timestamp}"


def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format."""
    print("\n" + "=" * 60)
    print("Configuration")
    print("=" * 60)
    
    # Group related configs
    groups = {
        'Method': ['method', 'backbone', 'img_size', 'patch_size'],
        'CFM': ['use_cfm', 'cfm'],
        'Training': ['epochs', 'batch_size', 'lr', 'min_lr', 'weight_decay', 'warmup_epochs'],
        'Data': ['data_dir', 'num_workers', 'max_samples'],
        'Augmentation': ['n_global_crops', 'n_local_crops', 'global_crop_size', 'local_crop_size'],
        'SSL': ['ema_momentum', 'student_temp', 'teacher_temp', 'temperature'],
        'DINOv2': ['use_ibot', 'mask_ratio', 'ibot_loss_weight', 'use_koleo', 'koleo_loss_weight', 'dino_loss_weight'],
        'Logging': ['output_dir', 'wandb_project', 'run_name', 'use_wandb'],
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in config:
                value = config[key]
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load and merge configuration
    if args.config is not None:
        config = load_config(args.config)
        config = merge_config_and_args(config, args)
    else:
        config = build_config_from_args(args)
    
    # Validate configuration
    validate_config(config)
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Generate run name if not provided
    if config.get('run_name') is None:
        config['run_name'] = generate_run_name(config)
    
    # Print configuration (only on main process)
    # Note: distributed not initialized yet, so we check env vars
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print_config(config)
    
    # Select trainer based on method
    method = config.get('method', 'dinov2')
    
    if method == 'dinov2':
        config['aug_type'] = 'multicrop'
        trainer = DINOv2Trainer(config)
    elif method == 'moco':
        config['aug_type'] = 'twoview'
        trainer = MoCoV3Trainer(config)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Start training
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
    
    trainer.train()
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)


if __name__ == "__main__":
    main()