# """
# EMP-SSL Training Script
# Train ResNet-50 with EMP-SSL on 500K dataset

# Simple training script for Lightning.ai or local machines

# Usage:
#     python train_empssl.py --config configs/empssl_base.yaml
# """

# import os
# import sys
# import time
# import argparse
# import yaml
# from pathlib import Path
# from datetime import datetime

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast as cuda_autocast, GradScaler
# try:
#     from torch.amp import autocast
# except ImportError:
#     # Fallback for older PyTorch
#     from torch.cuda.amp import autocast

# from model_empssl import create_empssl_model
# from utils_empssl import (
#     EMPSSLLoss, 
#     get_empssl_transforms,
#     AverageMeter,
#     adjust_learning_rate,
#     save_checkpoint,
#     load_checkpoint
# )

# from data_pipeline import SSLDataset


# class EMPSSLTrainer:
#     """
#     EMP-SSL training orchestrator
#     Handles training loop, checkpointing, and logging
#     """
    
#     def __init__(self, config):
#         """
#         Args:
#             config: Configuration dictionary
#         """
#         self.config = config
        
#         # Setup device (prioritize CUDA > MPS > CPU)
#         if torch.cuda.is_available():
#             self.device = torch.device('cuda')
#         elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#             self.device = torch.device('mps')
#         else:
#             self.device = torch.device('cpu')
#         print(f"Using device: {self.device}")
        
#         # Create experiment directory
#         self.setup_directories()
        
#         # Save configuration
#         self.save_config()
        
#         # Build model, optimizer, etc.
#         self.build_model()
#         self.build_dataloader()
#         self.build_optimizer()
#         self.build_loss()
        
#         # Mixed precision training (MPS has limited AMP support)
#         self.use_amp = config.get('use_amp', True) and self.device.type != 'mps'
#         if self.use_amp:
#             if self.device.type == 'cuda':
#                 self.scaler = GradScaler()
#             else:
#                 self.scaler = None  # MPS doesn't support GradScaler
#         else:
#             self.scaler = None
        
#         # Training state
#         self.start_epoch = 0
#         self.global_step = 0
#         self.best_loss = float('inf')
        
#     def setup_directories(self):
#         """Create directories for checkpoints and logs"""
#         exp_name = self.config.get('experiment_name', 'empssl_experiment')
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         self.exp_dir = Path(self.config.get('output_dir', './experiments')) / f"{exp_name}_{timestamp}"
        
#         self.exp_dir.mkdir(parents=True, exist_ok=True)
#         self.checkpoint_dir = self.exp_dir / 'checkpoints'
#         self.checkpoint_dir.mkdir(exist_ok=True)
#         self.log_file = self.exp_dir / 'training.log'
        
#         # Create log file
#         with open(self.log_file, 'w') as f:
#             f.write(f"EMP-SSL Training Log - {timestamp}\n")
#             f.write("="*80 + "\n\n")
    
#     def save_config(self):
#         """Save configuration to experiment directory"""
#         config_path = self.exp_dir / 'config.yaml'
#         with open(config_path, 'w') as f:
#             yaml.dump(self.config, f, default_flow_style=False)
#         self.log(f"Configuration saved to {config_path}")
    
#     def log(self, message, print_console=True):
#         """Log message to file and optionally console"""
#         timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         log_message = f"[{timestamp}] {message}"
        
#         if print_console:
#             print(log_message)
        
#         with open(self.log_file, 'a') as f:
#             f.write(log_message + '\n')
    
#     def build_model(self):
#         """Build EMP-SSL model"""
#         self.log("Building EMP-SSL ResNet-50 model...")
        
#         model, param_info = create_empssl_model(
#             projection_hidden_dim=self.config.get('projection_hidden_dim', 2048),
#             projection_output_dim=self.config.get('projection_output_dim', 128)
#         )
        
#         self.model = model.to(self.device)
        
#         # Log parameter info
#         self.log(f"Model parameters: {param_info['total']:,}")
#         self.log(f"  Encoder: {param_info['encoder']:,}")
#         self.log(f"  Projector: {param_info['projector']:,}")
    
#     def build_dataloader(self):
#         """Build training dataloader"""
#         self.log("Building dataloader...")
        
#         # Get transforms
#         transform = get_empssl_transforms(
#             image_size=self.config.get('image_size', 96),
#             patch_size=self.config.get('patch_size', 32),
#             num_patches=self.config.get('num_patches', 200),
#             is_training=True
#         )
        
#         # Create dataset
#         # Use pre-computed patches if available (much faster!)
#         precomputed_dir = self.config.get('precomputed_patches_dir', None)
#         if precomputed_dir:
#             self.log(f"Using pre-computed patches from: {precomputed_dir}")
#             # No transform needed - patches already extracted
#             transform = None
        
#         dataset = SSLDataset(
#             dataset_name=self.config.get('dataset_name', 'tsbpp/fall2025_deeplearning'),
#             split='train',
#             transform=transform,
#             cache_dir=self.config.get('cache_dir', None),
#             num_samples=self.config.get('num_samples', None),  # For testing with subset
#             precomputed_patches_dir=precomputed_dir
#         )
        
#         self.log(f"Dataset size: {len(dataset):,} images")
        
#         # Custom collate function for patches
#         from data_pipeline import collate_patches
        
#         # Create dataloader with custom collate function
#         num_workers = self.config.get('num_workers', 0)
#         self.train_loader = DataLoader(
#             dataset,
#             batch_size=self.config.get('batch_size', 256),
#             shuffle=True,
#             num_workers=num_workers,
#             pin_memory=True if num_workers > 0 else False,  # Disable pin_memory if no workers
#             drop_last=True,
#             collate_fn=collate_patches,  # Custom collate for list of patches
#             persistent_workers=False,  # Disable persistent workers to avoid hangs
#             prefetch_factor=2 if num_workers > 0 else None  # Only use prefetch with workers
#         )
        
#         self.log(f"Dataloader created: {len(self.train_loader)} batches per epoch")
    
#     def build_optimizer(self):
#         """Build optimizer and learning rate scheduler"""
#         self.log("Building optimizer...")
        
#         lr = self.config.get('learning_rate', 0.3)
#         weight_decay = self.config.get('weight_decay', 1e-4)
#         optimizer_type = self.config.get('optimizer', 'lars').lower()  # Default to LARS
        
#         if optimizer_type == 'lars':
#             # LARS optimizer (common for contrastive learning)
#             # Fall back to SGD if LARS not available
#             try:
#                 from torchlars import LARS
#                 base_optimizer = torch.optim.SGD(
#                     self.model.parameters(),
#                     lr=lr,
#                     momentum=0.9,
#                     weight_decay=weight_decay
#                 )
#                 self.optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
#                 self.log("Using LARS optimizer")
#             except ImportError:
#                 self.optimizer = torch.optim.SGD(
#                     self.model.parameters(),
#                     lr=lr,
#                     momentum=0.9,
#                     weight_decay=weight_decay
#                 )
#                 self.log("Using SGD optimizer (LARS not available, install torchlars)")
#         elif optimizer_type == 'sgd':
#             self.optimizer = torch.optim.SGD(
#                 self.model.parameters(),
#                 lr=lr,
#                 momentum=0.9,
#                 weight_decay=weight_decay
#             )
#             self.log("Using SGD optimizer")
#         elif optimizer_type == 'adamw':
#             self.optimizer = torch.optim.AdamW(
#                 self.model.parameters(),
#                 lr=lr,
#                 weight_decay=weight_decay,
#                 betas=(0.9, 0.999)
#             )
#             self.log("Using AdamW optimizer")
#         else:
#             raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose from: lars, sgd, adamw")
    
#     def build_loss(self):
#         """Build loss function"""
#         self.criterion = EMPSSLLoss(
#             lambda_inv=self.config.get('lambda_inv', 200.0),
#             epsilon_sq=self.config.get('epsilon_sq', 0.2),
#             projection_dim=self.config.get('projection_output_dim', 128)
#         )
#         self.log(f"Loss: EMP-SSL (TCR + Invariance)")
#         self.log(f"  λ (lambda_inv): {self.config.get('lambda_inv', 200.0)}")
#         self.log(f"  ε² (epsilon_sq): {self.config.get('epsilon_sq', 0.2)}")
    
#     def train_epoch(self, epoch):
#         """Train for one epoch"""
#         self.model.train()
        
#         # Metrics
#         loss_meter = AverageMeter()
#         tcr_meter = AverageMeter()
#         inv_meter = AverageMeter()
#         batch_time = AverageMeter()
#         data_time = AverageMeter()
        
#         end = time.time()
        
#         # Get num_patches: use from precomputed patches metadata if available, else from config
#         if hasattr(self.train_loader.dataset, 'precomputed_num_patches'):
#             num_patches = self.train_loader.dataset.precomputed_num_patches
#             self.log(f"Using num_patches from precomputed patches: {num_patches}")
#         else:
#             num_patches = self.config.get('num_patches', 200)
#             self.log(f"Using num_patches from config: {num_patches}")
        
#         # Test first batch to catch errors early
#         self.log("Testing first batch...")
#         try:
#             self.log("  Creating dataloader iterator...")
#             dataloader_iter = iter(self.train_loader)
#             self.log("  Getting first batch...")
#             test_batch = next(dataloader_iter)
#             self.log(f"✓ First batch loaded successfully: {len(test_batch)} patch tensors")
#             self.log(f"  Batch shapes: {[p.shape for p in test_batch]}")
#         except Exception as e:
#             self.log(f"✗ ERROR loading first batch: {e}")
#             import traceback
#             self.log(traceback.format_exc())
#             raise
        
#         self.log("Starting training loop...")
#         self.log(f"  Dataloader length: {len(self.train_loader)} batches")
#         self.log(f"  About to iterate dataloader...")
        
#         # Force Python to flush output
#         import sys
#         sys.stdout.flush()
        
#         try:
#             dataloader_iter = iter(self.train_loader)
#             self.log(f"  Dataloader iterator created, getting first batch...")
#             for batch_idx, patches_list in enumerate(dataloader_iter):
#                 if batch_idx == 0:
#                     self.log(f"Processing first batch (batch_idx={batch_idx})...")
#                 # Measure data loading time
#                 data_time.update(time.time() - end)
                
#                 if batch_idx == 0:
#                     self.log(f"  Data loading time: {data_time.val:.3f}s")
                
#                 # Adjust learning rate
#                 current_lr = adjust_learning_rate(
#                     self.optimizer,
#                     epoch + batch_idx / len(self.train_loader),
#                     warmup_epochs=self.config.get('warmup_epochs', 10),
#                     total_epochs=self.config.get('epochs', 100),
#                     base_lr=self.config.get('learning_rate', 0.3),
#                     min_lr=self.config.get('min_lr', 0.0)
#                 )
                
#                 if batch_idx == 0:
#                     self.log(f"  Learning rate adjusted: {current_lr:.6f}")
                
#                 # patches_list is a list of n_patches tensors, each [B, 3, H, W]
#                 # Stack all patches: [B * n_patches, 3, H, W]
#                 batch_size = patches_list[0].size(0)
#                 if batch_idx == 0:
#                     self.log(f"  Batch size: {batch_size}, num_patches: {num_patches}")
#                     self.log(f"  Stacking patches...")
#                 all_patches = torch.cat(patches_list, dim=0).to(self.device)
#                 if batch_idx == 0:
#                     self.log(f"  Patches stacked and moved to GPU: {all_patches.shape}")
                
#                 # Forward pass with mixed precision
#                 try:
#                     if batch_idx == 0:
#                         self.log(f"  Starting forward pass...")
#                         torch.cuda.synchronize()  # Ensure GPU is ready
#                         self.log(f"  GPU synchronized, calling model...")
                    
#                     if self.device.type == 'mps':
#                         # MPS: use autocast with device='mps' or disable AMP
#                         if batch_idx == 0:
#                             self.log(f"  Calling model.forward()...")
#                         projections = self.model(all_patches)  # [B * n_patches, projection_dim]
#                         if batch_idx == 0:
#                             torch.cuda.synchronize()
#                             self.log(f"  Model forward pass complete: {projections.shape}")
#                             self.log(f"  Computing loss...")
#                         loss, tcr_loss, inv_loss = self.criterion(projections, num_patches)
#                         if batch_idx == 0:
#                             self.log(f"  Loss computed: {loss.item():.4f}")
#                     else:
#                         # CUDA/CPU: use autocast
#                         if batch_idx == 0:
#                             self.log(f"  Entering autocast context...")
#                         with autocast(enabled=self.use_amp, device_type=self.device.type):
#                             if batch_idx == 0:
#                                 self.log(f"  Calling model.forward()...")
#                             projections = self.model(all_patches)  # [B * n_patches, projection_dim]
#                             if batch_idx == 0:
#                                 torch.cuda.synchronize()
#                                 self.log(f"  Model forward pass complete: {projections.shape}")
#                                 self.log(f"  Computing loss...")
#                             loss, tcr_loss, inv_loss = self.criterion(projections, num_patches)
#                             if batch_idx == 0:
#                                 self.log(f"  Loss computed: {loss.item():.4f}")
                    
#                     # Check for NaN immediately
#                     if torch.isnan(loss) or torch.isinf(loss):
#                         self.log(f"ERROR: NaN/Inf loss at batch {batch_idx}! TCR: {tcr_loss}, Inv: {inv_loss}")
#                         raise ValueError(f"NaN/Inf loss detected at batch {batch_idx}")
#                 except Exception as e:
#                     self.log(f"ERROR in forward pass at batch {batch_idx}: {e}")
#                     import traceback
#                     self.log(traceback.format_exc())
#                     raise
                
#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 if self.use_amp and self.scaler is not None:
#                     self.scaler.scale(loss).backward()
#                     # Gradient clipping to prevent explosion
#                     self.scaler.unscale_(self.optimizer)
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                     self.scaler.step(self.optimizer)
#                     self.scaler.update()
#                 else:
#                     loss.backward()
#                     # Gradient clipping to prevent explosion
#                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                     self.optimizer.step()
                
#                 # Update metrics
#                 loss_meter.update(loss.item(), batch_size)
#                 tcr_meter.update(tcr_loss.item(), batch_size)
#                 inv_meter.update(inv_loss.item(), batch_size)
#                 batch_time.update(time.time() - end)
#                 end = time.time()
                
#                 self.global_step += 1
                
#                 # Save checkpoint periodically during training (for resume safety)
#                 checkpoint_batch_interval = self.config.get('checkpoint_batch_interval', None)
#                 if checkpoint_batch_interval and (batch_idx + 1) % checkpoint_batch_interval == 0:
#                     self.save_checkpoint_state(
#                         epoch, 
#                         loss_meter.avg, 
#                         is_best=False  # Don't overwrite best, just save progress
#                     )
#                     self.log(f"Mid-epoch checkpoint saved at batch {batch_idx + 1}")
                
#                 # Log progress
#                 if batch_idx % self.config.get('log_interval', 50) == 0:
#                     self.log(
#                         f"Epoch [{epoch}/{self.config.get('epochs', 100)}] "
#                         f"Batch [{batch_idx}/{len(self.train_loader)}] "
#                         f"Loss: {loss_meter.avg:.4f} (TCR: {tcr_meter.avg:.4f}, Inv: {inv_meter.avg:.4f}) "
#                         f"LR: {current_lr:.6f} "
#                         f"Time: {batch_time.avg:.3f}s "
#                         f"Data: {data_time.avg:.3f}s"
#                     )
#         except Exception as e:
#             self.log(f"ERROR in training loop: {e}")
#             import traceback
#             self.log(traceback.format_exc())
#             raise
        
#         return loss_meter.avg
    
#     def save_checkpoint_state(self, epoch, loss, is_best=False):
#         """Save training checkpoint"""
#         state = {
#             'epoch': epoch,
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'loss': loss,
#             'config': self.config,
#             'global_step': self.global_step
#         }
        
#         # Save latest checkpoint
#         checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
#         save_checkpoint(state, checkpoint_path)
        
#         # Save epoch checkpoint every N epochs
#         if epoch % self.config.get('checkpoint_interval', 10) == 0:
#             checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
#             save_checkpoint(state, checkpoint_path)
        
#         # Save best checkpoint
#         if is_best:
#             checkpoint_path = self.checkpoint_dir / 'checkpoint_best.pth'
#             save_checkpoint(state, checkpoint_path)
#             self.log(f"Best checkpoint saved! Loss: {loss:.4f}")
    
#     def load_checkpoint_state(self, checkpoint_path):
#         """Load training checkpoint"""
#         self.log(f"Loading checkpoint from {checkpoint_path}")
        
#         checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
#         # Load model and optimizer state
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
#         # Load training state
#         self.start_epoch = checkpoint['epoch'] + 1
#         self.global_step = checkpoint.get('global_step', 0)
#         self.best_loss = checkpoint.get('loss', float('inf'))
        
#         self.log(f"Resumed from epoch {checkpoint['epoch']}")
    
#     def train(self):
#         """Main training loop"""
#         self.log("="*80)
#         self.log("Starting EMP-SSL training")
#         self.log("="*80)
#         self.log(f"Total epochs: {self.config.get('epochs', 100)}")
#         self.log(f"Batch size: {self.config.get('batch_size', 256)}")
#         self.log(f"Learning rate: {self.config.get('learning_rate', 0.3)}")
#         self.log(f"Device: {self.device}")
#         self.log(f"Output directory: {self.exp_dir}")
#         self.log("="*80)
        
#         # Resume from checkpoint if specified
#         if self.config.get('resume_from'):
#             self.load_checkpoint_state(self.config['resume_from'])
        
#         total_epochs = self.config.get('epochs', 100)
        
#         self.log(f"Starting epoch loop: {self.start_epoch} to {total_epochs-1}")
        
#         for epoch in range(self.start_epoch, total_epochs):
#             epoch_start = time.time()
#             self.log(f"Starting epoch {epoch}...")
            
#             # Train one epoch
#             try:
#                 avg_loss = self.train_epoch(epoch)
#             except Exception as e:
#                 self.log(f"ERROR in train_epoch: {e}")
#                 import traceback
#                 self.log(traceback.format_exc())
#                 raise
            
#             epoch_time = time.time() - epoch_start
            
#             # Log epoch summary
#             self.log("="*80)
#             self.log(f"Epoch {epoch} completed in {epoch_time/60:.2f} minutes")
#             self.log(f"Average loss: {avg_loss:.4f}")
#             self.log("="*80)
            
#             # Save checkpoint
#             is_best = avg_loss < self.best_loss
#             if is_best:
#                 self.best_loss = avg_loss
            
#             self.save_checkpoint_state(epoch, avg_loss, is_best)
        
#         self.log("Training completed!")


# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='EMP-SSL Training')
#     parser.add_argument('--config', type=str, required=True, help='Path to config file')
#     return parser.parse_args()


# def main():
#     args = parse_args()
    
#     # Load configuration
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Create trainer and start training
#     trainer = EMPSSLTrainer(config)
#     trainer.train()


# if __name__ == "__main__":
#     main()

"""
EMP-SSL Training Script - FIXED VERSION
Train ResNet-50 with EMP-SSL on 500K dataset

Updates:
1. Fixed Critical Data Ordering Bug (Transpose Fix)
2. Uses local LARS optimizer from utils_empssl
"""

import os
import sys
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as cuda_autocast, GradScaler

# Handle AMP import for different PyTorch versions
try:
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import autocast

from model_empssl import create_empssl_model
from utils_empssl import (
    EMPSSLLoss, 
    get_empssl_transforms,
    AverageMeter,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    LARS  # Importing local LARS from your utils update
)

from data_pipeline import SSLDataset, collate_patches


class EMPSSLTrainer:
    """
    EMP-SSL training orchestrator
    Handles training loop, checkpointing, and logging
    """
    
    def __init__(self, config):
        self.config = config
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Create experiment directory
        self.setup_directories()
        
        # Save configuration
        self.save_config()
        
        # Build components
        self.build_model()
        self.build_dataloader()
        self.build_optimizer()
        self.build_loss()
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', True) and self.device.type != 'mps'
        if self.use_amp:
            self.scaler = GradScaler() if self.device.type == 'cuda' else None
        else:
            self.scaler = None
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def setup_directories(self):
        exp_name = self.config.get('experiment_name', 'empssl_experiment')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(self.config.get('output_dir', './experiments')) / f"{exp_name}_{timestamp}"
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_file = self.exp_dir / 'training.log'
        
        with open(self.log_file, 'w') as f:
            f.write(f"EMP-SSL Training Log - {timestamp}\n")
            f.write("="*80 + "\n\n")
    
    def save_config(self):
        config_path = self.exp_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        self.log(f"Configuration saved to {config_path}")
    
    def log(self, message, print_console=True):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def build_model(self):
        self.log("Building EMP-SSL ResNet-50 model...")
        
        model, param_info = create_empssl_model(
            projection_hidden_dim=self.config.get('projection_hidden_dim', 2048),
            projection_output_dim=self.config.get('projection_output_dim', 128),
            small_image_mode=True # Ensuring 96x96 optimization is on
        )
        
        self.model = model.to(self.device)
        
        self.log(f"Model parameters: {param_info['total']:,}")
        self.log(f"  Encoder: {param_info['encoder']:,}")
        self.log(f"  Projector: {param_info['projector']:,}")
    
    def build_dataloader(self):
        self.log("Building dataloader...")
        
        # Get transforms (only needed if not using pre-computed patches)
        transform = get_empssl_transforms(
            image_size=self.config.get('image_size', 96),
            patch_size=self.config.get('patch_size', 32),
            num_patches=self.config.get('num_patches', 200),
            is_training=True
        )
        
        precomputed_dir = self.config.get('precomputed_patches_dir', None)
        if precomputed_dir:
            self.log(f"Using pre-computed patches from: {precomputed_dir}")
            transform = None
        
        dataset = SSLDataset(
            dataset_name=self.config.get('dataset_name', 'tsbpp/fall2025_deeplearning'),
            split='train',
            transform=transform,
            cache_dir=self.config.get('cache_dir', None),
            num_samples=self.config.get('num_samples', None),
            precomputed_patches_dir=precomputed_dir
        )
        
        self.log(f"Dataset size: {len(dataset):,} images")
        
        num_workers = self.config.get('num_workers', 4)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 256),
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False,
            drop_last=True,
            collate_fn=collate_patches,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        self.log(f"Dataloader created: {len(self.train_loader)} batches per epoch")
    
    def build_optimizer(self):
        self.log("Building optimizer...")
        
        lr = self.config.get('learning_rate', 0.3)
        weight_decay = self.config.get('weight_decay', 1e-4)
        optimizer_type = self.config.get('optimizer', 'lars').lower()
        
        if optimizer_type == 'lars':
            # Use Local LARS from utils_empssl
            self.optimizer = LARS(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                eta=0.001
            )
            self.log("Using Local LARS optimizer (from utils)")
            
        elif optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
            self.log("Using SGD optimizer")
            
        elif optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
            self.log("Using AdamW optimizer")
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def build_loss(self):
        self.criterion = EMPSSLLoss(
            lambda_inv=self.config.get('lambda_inv', 200.0),
            epsilon_sq=self.config.get('epsilon_sq', 0.2),
            projection_dim=self.config.get('projection_output_dim', 128)
        )
        self.log(f"Loss: EMP-SSL (TCR + Invariance)")
    
    def train_epoch(self, epoch):
        self.model.train()
        
        loss_meter = AverageMeter()
        tcr_meter = AverageMeter()
        inv_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        
        # Determine num_patches
        if hasattr(self.train_loader.dataset, 'precomputed_num_patches'):
            num_patches = self.train_loader.dataset.precomputed_num_patches
        else:
            num_patches = self.config.get('num_patches', 200)
        
        if epoch == 0:
            self.log(f"Training with num_patches={num_patches}")

        for batch_idx, patches_list in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            
            # Adjust LR
            current_lr = adjust_learning_rate(
                self.optimizer,
                epoch + batch_idx / len(self.train_loader),
                warmup_epochs=self.config.get('warmup_epochs', 10),
                total_epochs=self.config.get('epochs', 100),
                base_lr=self.config.get('learning_rate', 0.3),
                min_lr=self.config.get('min_lr', 0.0)
            )
            
            # =================================================================
            # CRITICAL FIX: TRANSPOSE BATCH
            # =================================================================
            # Input patches_list is [Patch1_Batch, Patch2_Batch, ...]
            # Each element is [B, 3, H, W]
            
            # 1. Get dimensions
            batch_size = patches_list[0].size(0)
            _, c, h, w = patches_list[0].shape
            
            # 2. Stack to [num_patches, B, 3, H, W]
            stacked_patches = torch.stack(patches_list, dim=0)
            
            # 3. Transpose to [B, num_patches, 3, H, W]
            # This ensures patches from the same image are adjacent in memory
            stacked_patches = stacked_patches.transpose(0, 1)
            
            # 4. Flatten to [B * num_patches, 3, H, W]
            all_patches = stacked_patches.reshape(-1, c, h, w).to(self.device)
            
            if batch_idx == 0 and epoch == 0:
                 self.log(f"Batch Structure Verified: [B={batch_size}, Patches={num_patches}] -> Reshaped to {all_patches.shape}")

            # =================================================================
            
            # Forward pass
            try:
                if self.device.type == 'mps':
                    # MPS doesn't support autocast well with some ops
                    projections = self.model(all_patches)
                    loss, tcr_loss, inv_loss = self.criterion(projections, num_patches)
                else:
                    with autocast(enabled=self.use_amp, device_type=self.device.type):
                        projections = self.model(all_patches)
                        loss, tcr_loss, inv_loss = self.criterion(projections, num_patches)
                
                # Nan check
                if torch.isnan(loss) or torch.isinf(loss):
                    self.log(f"WARNING: NaN/Inf loss at batch {batch_idx}. Skipping.")
                    continue
                    
            except Exception as e:
                self.log(f"Error in forward pass: {e}")
                raise
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Update meters
            loss_meter.update(loss.item(), batch_size)
            tcr_meter.update(tcr_loss.item(), batch_size)
            inv_meter.update(inv_loss.item(), batch_size)
            batch_time.update(time.time() - end)
            end = time.time()
            
            self.global_step += 1
            
            # Mid-epoch checkpoint
            chk_interval = self.config.get('checkpoint_batch_interval', None)
            if chk_interval and (batch_idx + 1) % chk_interval == 0:
                self.save_checkpoint_state(epoch, loss_meter.avg, is_best=False)
            
            # Logging
            if batch_idx % self.config.get('log_interval', 50) == 0:
                self.log(
                    f"Epoch [{epoch}/{self.config.get('epochs', 100)}] "
                    f"Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} "
                    f"(TCR: {tcr_meter.avg:.4f}, Inv: {inv_meter.avg:.4f}) "
                    f"LR: {current_lr:.4f} "
                    f"Time: {batch_time.avg:.3f}s"
                )
        
        return loss_meter.avg
    
    def save_checkpoint_state(self, epoch, loss, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        # Save latest
        save_checkpoint(state, self.checkpoint_dir / 'checkpoint_latest.pth')
        
        # Save by epoch
        if epoch % self.config.get('checkpoint_interval', 10) == 0:
            save_checkpoint(state, self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth')
        
        # Save best
        if is_best:
            save_checkpoint(state, self.checkpoint_dir / 'checkpoint_best.pth')
            self.log(f"Best checkpoint saved! Loss: {loss:.4f}")
    
    def train(self):
        self.log("="*80)
        self.log("Starting EMP-SSL training (FIXED VERSION)")
        self.log("="*80)
        
        if self.config.get('resume_from'):
            self.load_checkpoint_state(self.config['resume_from'])
        
        for epoch in range(self.start_epoch, self.config.get('epochs', 100)):
            epoch_start = time.time()
            
            avg_loss = self.train_epoch(epoch)
            
            epoch_time = time.time() - epoch_start
            self.log(f"Epoch {epoch} done in {epoch_time/60:.2f} mins. Avg Loss: {avg_loss:.4f}")
            
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            self.save_checkpoint_state(epoch, avg_loss, is_best)
            
        self.log("Training completed!")

    def load_checkpoint_state(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))
        self.log(f"Resumed from epoch {checkpoint['epoch']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = EMPSSLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()