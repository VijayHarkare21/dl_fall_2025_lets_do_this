"""
Base Trainer for Self-Supervised Learning
==========================================

This module provides an abstract base class for SSL training that handles:
- Distributed training setup (single/multi-GPU via torchrun)
- Training loop with epoch/step management
- Checkpoint saving/loading and training resumption
- Logging to W&B and console
- Learning rate scheduling
- CFM (Conditional Feature Modulation) integration

Subclasses (DINOv2Trainer, MoCoV3Trainer) implement the specific:
- Model construction (backbone + heads + teacher/momentum encoder)
- Forward pass and loss computation
- EMA/momentum updates

Design Principles:
- CFM is a first-class feature, toggled via `use_cfm` config
- All logging/checkpointing happens only on rank 0
- Training can be resumed from any checkpoint
- Compatible with both single-GPU and multi-GPU (torchrun) execution

Usage:
    trainer = DINOv2Trainer(config)
    trainer.train()
"""

import os
import math
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports - use absolute imports for when running as part of package
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import vit_small, vit_tiny, convnext_tiny, convnext_small, convnext_base
from models.cfm import CFMNetwork
from data.datasets import get_pretrain_dataloader
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_local_rank,
    get_world_size,
    all_reduce_mean,
)
from utils.logging_utils import Logger, MetricTracker, Timer


class BaseTrainer(ABC):
    """
    Abstract base class for SSL trainers.
    
    Handles the common training infrastructure while delegating
    method-specific logic (loss computation, model updates) to subclasses.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary containing all hyperparameters.
                   Required keys vary by subclass, but common ones include:
                   - output_dir: Where to save checkpoints and logs
                   - data_dir: Path to training data
                   - epochs: Number of training epochs
                   - batch_size: Per-GPU batch size
                   - lr: Base learning rate
                   - use_cfm: Whether to use Conditional Feature Modulation
                   - backbone: 'vit_small' or 'vit_tiny'
                   - wandb_project: W&B project name
        """
        self.config = config
        
        # Setup distributed training (must be first)
        setup_distributed()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')
        
        # Initialize logger (only logs on main process)
        self.logger = Logger(
            project=config.get('wandb_project', 'ssl-vision-backbone'),
            name=config.get('run_name', None),
            config=config,
            output_dir=config.get('output_dir', './outputs'),
            use_wandb=config.get('use_wandb', True),
        )
        
        # Build components (order matters)
        self._build_backbone()
        self._build_cfm()
        self._build_dataloader()
        self._build_model()  # Subclass implements: heads, teacher, etc.
        self._build_cfm_curriculum()  # After dataloader since we need steps_per_epoch
        self._build_optimizer()
        self._build_scheduler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        
        # Try to resume from checkpoint if exists
        self._maybe_resume()
        
        # Log model info and print parameter summary
        self._log_model_info()
        self._print_param_summary()
    
    def _log_gpu_info(self):
        """Log GPU information and memory usage."""
        if not self.is_main or not torch.cuda.is_available():
            return
        
        print("\n" + "=" * 60)
        print("GPU Information")
        print("=" * 60)
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {total_memory:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Free: {total_memory - reserved:.2f} GB")
        
        print("=" * 60 + "\n")
    
    def _build_backbone(self):
        """Build the backbone network."""
        backbone_name = self.config.get('backbone', 'vit_small')
        img_size = self.config.get('img_size', 96)
        patch_size = self.config.get('patch_size', 8)
        
        # Track backbone type for downstream handling (e.g., iBOT compatibility)
        self.backbone_type = 'vit' if backbone_name.startswith('vit') else 'convnext'
        
        if backbone_name == 'vit_small':
            self.backbone = vit_small(img_size=img_size, patch_size=patch_size)
        elif backbone_name == 'vit_tiny':
            self.backbone = vit_tiny(img_size=img_size, patch_size=patch_size)
        elif backbone_name == 'convnext_tiny':
            self.backbone = convnext_tiny(img_size=img_size)
        elif backbone_name == 'convnext_small':
            self.backbone = convnext_small(img_size=img_size)
        elif backbone_name == 'convnext_base':
            self.backbone = convnext_base(img_size=img_size)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.backbone = self.backbone.to(self.device)
        self.embed_dim = self.backbone.embed_dim
        
        # Get number of blocks (ViT uses 'depth', ConvNeXt uses 'total_blocks')
        if hasattr(self.backbone, 'depth'):
            self.num_blocks = self.backbone.depth
        elif hasattr(self.backbone, 'total_blocks'):
            self.num_blocks = self.backbone.total_blocks
        else:
            self.num_blocks = self.backbone.num_blocks
        
        if self.is_main:
            print(f"[Trainer] Built backbone: {backbone_name} (type: {self.backbone_type})")
            print(f"  Embed dim: {self.embed_dim}")
            print(f"  Num blocks: {self.num_blocks}")
            # Handle different parameter counting methods
            if hasattr(self.backbone, 'get_num_params'):
                print(f"  Parameters: {self.backbone.get_num_params():,}")
            else:
                num_params = sum(p.numel() for p in self.backbone.parameters())
                print(f"  Parameters: {num_params:,}")
    
    def _build_cfm(self):
        """Build Conditional Feature Modulation network if enabled."""
        self.use_cfm = self.config.get('use_cfm', False)
        self.cfm_curriculum = None
        if not self.use_cfm:
            self.cfm = None
            if self.is_main:
                print("[Trainer] CFM disabled")
            return
        
        cfm_config = self.config.get('cfm', {})
        context_dim = cfm_config.get('context_dim', 256)
        hidden_dim = cfm_config.get('hidden_dim', 128)
        input_size = cfm_config.get('input_size', 48)
        
        # Check if backbone provides per-block dimensions (ConvNeXt style)
        # or uses uniform dimension (ViT style)
        if hasattr(self.backbone, 'block_dims'):
            # ConvNeXt: use per-block dimensions
            feature_dims = self.backbone.block_dims
            self.cfm = CFMNetwork(
                feature_dims=feature_dims,
                context_dim=context_dim,
                input_size=input_size,
                hidden_dim=hidden_dim,
            )
            if self.is_main:
                print(f"[Trainer] Built CFM with per-block dims: {feature_dims[:3]}...{feature_dims[-3:]}")
        else:
            # ViT: use uniform dimension
            self.cfm = CFMNetwork(
                num_blocks=self.num_blocks,
                feature_dim=self.embed_dim,
                context_dim=context_dim,
                input_size=input_size,
                hidden_dim=hidden_dim,
            )
            if self.is_main:
                print(f"[Trainer] Built CFM with uniform dim: {self.embed_dim} x {self.num_blocks} blocks")
        
        self.cfm = self.cfm.to(self.device)
        
        if self.is_main:
            print(f"[Trainer] CFM parameters: {self.cfm.get_num_params():,}")
    
    def _build_cfm_curriculum(self):
        """
        Build CFM curriculum scheduler after dataloader is created.
        
        Must be called after _build_dataloader since we need steps_per_epoch.
        """
        if not self.use_cfm:
            return
        
        cfm_config = self.config.get('cfm', {})
        
        # Get curriculum settings with defaults
        # Default: start CFM at 20% of training, full strength at 60%
        cfm_start_epoch = cfm_config.get('curriculum_start', 0.2)
        cfm_full_epoch = cfm_config.get('curriculum_full', 0.6)
        
        # Check if curriculum is disabled (start at 0 means train CFM from beginning)
        if cfm_start_epoch == 0 and cfm_full_epoch == 0:
            self.cfm_curriculum = None
            if self.is_main:
                print("[Trainer] CFM curriculum disabled - training CFM from start")
            return
        
        self.cfm_curriculum = CFMCurriculumScheduler(
            total_epochs=self.config.get('epochs', 100),
            steps_per_epoch=self.steps_per_epoch,
            cfm_start_epoch=cfm_start_epoch,
            cfm_full_epoch=cfm_full_epoch,
        )
        
        if self.is_main:
            print(f"[Trainer] CFM curriculum enabled:")
            print(f"  Phase 1 (backbone only): epochs 0-{self.cfm_curriculum.cfm_start_epoch}")
            print(f"  Phase 2 (CFM ramp-up): epochs {self.cfm_curriculum.cfm_start_epoch}-{self.cfm_curriculum.cfm_full_epoch}")
            print(f"  Phase 3 (full CFM): epochs {self.cfm_curriculum.cfm_full_epoch}+")
    
    @abstractmethod
    def _build_model(self):
        """
        Build the complete model (heads, teacher, etc.).
        
        Subclasses must implement this to set up:
        - Projection/prediction heads
        - Teacher/momentum encoder (for DINO/MoCo)
        - Any other method-specific components
        
        After building, wrap student model in DDP if distributed.
        """
        pass
    
    def _build_dataloader(self):
        """Build the training dataloader."""
        data_dir = self.config['data_dir']
        batch_size = self.config.get('batch_size', 64)
        num_workers = self.config.get('num_workers', 4)
        
        # Augmentation type depends on SSL method
        aug_type = self.config.get('aug_type', 'multicrop')
        n_global_crops = self.config.get('n_global_crops', 2)
        n_local_crops = self.config.get('n_local_crops', 6)
        global_crop_size = self.config.get('global_crop_size', 96)
        local_crop_size = self.config.get('local_crop_size', 48)
        
        # For debugging/quick tests
        max_samples = self.config.get('max_samples', None)
        
        # Create dataloader (handles distributed sampler internally if needed)
        self.dataloader = get_pretrain_dataloader(
            root_dir=data_dir,
            batch_size=batch_size,
            aug_type=aug_type,
            n_global_crops=n_global_crops,
            n_local_crops=n_local_crops,
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            num_workers=num_workers,
            max_samples=max_samples,
            seed=self.config.get('seed', 42),
        )
        
        # For distributed training, we need to use DistributedSampler
        # Recreate with distributed sampler if needed
        if is_distributed():
            from data.datasets import SSLPretrainDataset, collate_multicrop, collate_twoview
            from data.augmentations import get_augmentation
            
            transform = get_augmentation(
                aug_type=aug_type,
                global_crop_size=global_crop_size,
                local_crop_size=local_crop_size,
                n_global_crops=n_global_crops,
                n_local_crops=n_local_crops,
            )
            
            dataset = SSLPretrainDataset(
                root_dir=data_dir,
                transform=transform,
                max_samples=max_samples,
            )
            
            self.sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True,
            )
            
            collate_fn = collate_multicrop if aug_type == 'multicrop' else collate_twoview
            
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=self.sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True,
                persistent_workers=num_workers > 0,
            )
        else:
            self.sampler = None
        
        self.steps_per_epoch = len(self.dataloader)
        
        if self.is_main:
            dataset_size = len(self.dataloader.dataset)
            self.logger.log_dataset_info(
                name="pretrain",
                num_samples=dataset_size,
                split="train"
            )
            print(f"[Trainer] DataLoader: {self.steps_per_epoch} steps/epoch")
    
    def _build_optimizer(self):
        """Build the optimizer."""
        # Collect parameters from all trainable components
        param_groups = self._get_param_groups()
        
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.04)
        optimizer_name = self.config.get('optimizer', 'adamw')
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        if self.is_main:
            print(f"[Trainer] Optimizer: {optimizer_name}, LR: {lr}, WD: {weight_decay}")
    
    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for the optimizer.
        
        Subclasses can override to use different LR for different components.
        Default: all parameters with same LR.
        """
        params = []
        
        # Backbone parameters
        params.extend(self.backbone.parameters())
        
        # CFM parameters (if enabled)
        if self.cfm is not None:
            params.extend(self.cfm.parameters())
        
        return [{'params': params}]
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        epochs = self.config.get('epochs', 100)
        warmup_epochs = self.config.get('warmup_epochs', 10)
        min_lr = self.config.get('min_lr', 1e-6)
        
        # Total steps
        total_steps = epochs * self.steps_per_epoch
        warmup_steps = warmup_epochs * self.steps_per_epoch
        
        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(warmup_steps, 1)
            else:
                # Cosine decay
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return max(min_lr / self.config.get('lr', 1e-4), 
                          0.5 * (1 + math.cos(math.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        if self.is_main:
            print(f"[Trainer] Scheduler: cosine with {warmup_epochs} warmup epochs")
    
    def _maybe_resume(self):
        """Resume training from checkpoint if available."""
        resume_path = self.config.get('resume', None)
        
        # Check for auto-resume from latest checkpoint
        if resume_path is None and self.config.get('auto_resume', True):
            latest_path = self.logger.get_resume_path()
            if latest_path is not None:
                resume_path = latest_path
        
        if resume_path is not None and Path(resume_path).exists():
            if self.is_main:
                print(f"[Trainer] Resuming from {resume_path}")
            
            checkpoint = self.logger.load_checkpoint(
                resume_path,
                model=self.backbone,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            
            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('step', 0)
            self.best_metric = checkpoint.get('best_metric', None)
            
            # Load CFM if present
            if self.cfm is not None and 'cfm_state_dict' in checkpoint:
                self.cfm.load_state_dict(checkpoint['cfm_state_dict'])
            
            # Load method-specific state (teacher, etc.)
            self._load_checkpoint_hook(checkpoint)
            
            if self.is_main:
                print(f"[Trainer] Resumed from epoch {self.epoch}, step {self.global_step}")
    
    def _load_checkpoint_hook(self, checkpoint: Dict[str, Any]):
        """
        Hook for subclasses to load additional state from checkpoint.
        
        Override in subclass to load teacher model, centering buffer, etc.
        """
        pass
    
    def _save_checkpoint_hook(self) -> Dict[str, Any]:
        """
        Hook for subclasses to save additional state to checkpoint.
        
        Override in subclass to save teacher model, centering buffer, etc.
        """
        return {}
    
    def _log_model_info(self):
        """Log information about all model components."""
        if not self.is_main:
            return
        
        self.logger.log_model_info(self.backbone, "backbone")
        if self.cfm is not None:
            self.logger.log_model_info(self.cfm, "cfm")
    
    def _print_param_summary(self):
        """
        Print a summary of all model parameters.
        
        Called after all components are built to show total counts.
        """
        if not self.is_main:
            return
        
        print("\n" + "=" * 60)
        print("Model Parameter Summary")
        print("=" * 60)
        
        # Backbone
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Backbone: {backbone_params:,} total, {backbone_trainable:,} trainable")
        
        # CFM
        if self.cfm is not None:
            cfm_params = sum(p.numel() for p in self.cfm.parameters())
            cfm_trainable = sum(p.numel() for p in self.cfm.parameters() if p.requires_grad)
            print(f"CFM: {cfm_params:,} total, {cfm_trainable:,} trainable")
        else:
            cfm_params = 0
            cfm_trainable = 0
        
        # Heads (to be filled by subclass)
        head_params, head_trainable = self._get_head_param_counts()
        if head_params > 0:
            print(f"Heads: {head_params:,} total, {head_trainable:,} trainable")
        
        # Total
        total_params = backbone_params + cfm_params + head_params
        total_trainable = backbone_trainable + cfm_trainable + head_trainable
        
        print("-" * 60)
        print(f"TOTAL: {total_params:,} parameters")
        print(f"TRAINABLE: {total_trainable:,} parameters")
        print("=" * 60 + "\n")
    
    def _get_head_param_counts(self) -> tuple:
        """
        Get parameter counts for heads. Override in subclass.
        
        Returns:
            Tuple of (total_params, trainable_params)
        """
        return 0, 0
    
    def _wrap_ddp(self, model: nn.Module) -> nn.Module:
        """Wrap model in DistributedDataParallel if distributed."""
        if is_distributed():
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
            )
        return model
    
    def _get_cfm_modulations(self, images: torch.Tensor) -> Optional[List[Dict[str, torch.Tensor]]]:
        """
        Get CFM modulations for the input images.
        
        Args:
            images: Input images (B, C, H, W)
        
        Returns:
            List of modulation dicts if CFM enabled, else None
        """
        if self.cfm is None:
            return None
        return self.cfm(images)
    
    def _get_current_cfm_weight(self) -> float:
        """
        Get the current CFM curriculum weight for this training step.
        
        Returns:
            Weight between 0 and 1 (0 = no CFM, 1 = full CFM)
        """
        if not self.use_cfm:
            return 0.0
        
        if self.cfm_curriculum is None:
            # No curriculum - always use full CFM
            return 1.0
        
        return self.cfm_curriculum.get_cfm_weight(self.global_step)
    
    def _should_train_cfm(self) -> bool:
        """
        Check if CFM should receive gradients at the current step.
        
        During Phase 1 of curriculum, CFM parameters should be frozen.
        """
        if not self.use_cfm or self.cfm is None:
            return False
        
        if self.cfm_curriculum is None:
            # No curriculum - always train CFM
            return True
        
        return self.cfm_curriculum.should_train_cfm(self.global_step)
    
    def _update_cfm_training_state(self):
        """
        Update CFM training state based on curriculum.
        
        Freezes/unfreezes CFM parameters based on current phase.
        """
        if self.cfm is None:
            return
        
        should_train = self._should_train_cfm()
        
        # Get the actual module (handle DDP wrapper)
        cfm_module = self.cfm.module if hasattr(self.cfm, 'module') else self.cfm
        
        for param in cfm_module.parameters():
            param.requires_grad = should_train
    
    @abstractmethod
    def _train_step(self, batch: Any) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data from dataloader
        
        Returns:
            Dictionary of loss values (will be logged)
        """
        pass
    
    def train(self):
        """Main training loop."""
        torch.autograd.set_detect_anomaly(False)
        self._log_gpu_info()
        epochs = self.config.get('epochs', 100)
        log_every = self.config.get('log_every', 50)
        save_every = self.config.get('save_every', 10)  # epochs
        
        if self.is_main:
            print(f"\n[Trainer] Starting training for {epochs} epochs")
            print(f"  Device: {self.device}")
            print(f"  World size: {self.world_size}")
            print(f"  Batch size (per GPU): {self.config.get('batch_size', 64)}")
            print(f"  Effective batch size: {self.config.get('batch_size', 64) * self.world_size}")
            if self.use_cfm:
                print(f"  CFM: Enabled with curriculum" if self.cfm_curriculum else "  CFM: Enabled (no curriculum)")
        
        timer = Timer()
        metric_tracker = MetricTracker()
        
        for epoch in range(self.epoch, epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler (ensures different shuffle each epoch)
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            
            # Training epoch
            self.backbone.train()
            if self.cfm is not None:
                self.cfm.train()
            self._set_train_mode()  # Subclass sets heads, etc. to train mode
            
            metric_tracker.reset()
            epoch_start = time.time()
            
            # Track CFM phase at start of epoch
            if self.is_main and self.use_cfm and self.cfm_curriculum is not None:
                phase = self.cfm_curriculum.get_phase(self.global_step)
                cfm_weight = self._get_current_cfm_weight()
                print(f"\n[Epoch {epoch+1}] CFM phase: {phase}, weight: {cfm_weight:.3f}")
            
            for step, batch in enumerate(self.dataloader):
                timer.start('step')
                
                # Update CFM training state based on curriculum
                self._update_cfm_training_state()
                
                # Update current CFM weight for use in subclass _train_step
                self.current_cfm_weight = self._get_current_cfm_weight()
                
                # Training step (implemented by subclass)
                losses = self._train_step(batch)
                
                # Update metrics
                batch_size = self._get_batch_size(batch)
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item()
                    metric_tracker.update(k, v, batch_size)
                
                # Update scheduler
                self.scheduler.step()
                self.global_step += 1
                
                timer.stop('step')
                
                # Logging
                if self.is_main and (step + 1) % log_every == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    metrics = metric_tracker.compute()
                    metrics['lr'] = lr
                    metrics['epoch'] = epoch
                    
                    # Add CFM weight to metrics if using CFM
                    if self.use_cfm:
                        metrics['cfm_weight'] = self.current_cfm_weight
                    
                    self.logger.log(metrics, step=self.global_step)
                    self.logger.log_console(
                        f"Epoch {epoch+1}/{epochs}",
                        metrics={'loss': metrics.get('loss', 0), 'lr': lr},
                        step=self.global_step,
                    )
            
            # End of epoch
            epoch_time = time.time() - epoch_start
            epoch_metrics = metric_tracker.compute()
            
            # Aggregate metrics across GPUs
            if is_distributed():
                for k, v in epoch_metrics.items():
                    tensor = torch.tensor(v, device=self.device)
                    epoch_metrics[k] = all_reduce_mean(tensor).item()
            
            if self.is_main:
                epoch_metrics['epoch_time'] = epoch_time
                if self.use_cfm:
                    epoch_metrics['cfm_weight'] = self.current_cfm_weight
                self.logger.log(epoch_metrics, step=self.global_step, epoch=epoch)
                print(f"\n[Epoch {epoch+1}/{epochs}] Loss: {epoch_metrics.get('loss', 0):.4f}, "
                      f"Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if self.is_main and (epoch + 1) % save_every == 0:
                additional_state = self._save_checkpoint_hook()
                if self.cfm is not None:
                    cfm_module = self.cfm.module if hasattr(self.cfm, 'module') else self.cfm
                    additional_state['cfm_state_dict'] = cfm_module.state_dict()
                additional_state['best_metric'] = self.best_metric
                
                self.logger.save_checkpoint(
                    model=self.backbone,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    step=self.global_step,
                    metrics=epoch_metrics,
                    scheduler=self.scheduler,
                    additional_state=additional_state,
                )
        
        # Training complete
        if self.is_main:
            self.logger.finish()
            print("\n[Trainer] Training complete!")
        
        cleanup_distributed()
    
    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from batch (handles different batch formats)."""
        if isinstance(batch, (list, tuple)):
            # Multi-crop or two-view: first element is a tensor
            first = batch[0]
            if isinstance(first, torch.Tensor):
                return first.shape[0]
            elif isinstance(first, (list, tuple)):
                return first[0].shape[0]
        elif isinstance(batch, torch.Tensor):
            return batch.shape[0]
        return self.config.get('batch_size', 64)
    
    @abstractmethod
    def _set_train_mode(self):
        """Set all method-specific components to training mode."""
        pass


class EMAModel:
    """
    Exponential Moving Average (EMA) of model parameters.
    
    Used for the teacher model in DINO and the momentum encoder in MoCo.
    The EMA provides stable targets during training and is updated each step.
    
    Update rule: ema_param = momentum * ema_param + (1 - momentum) * param
    """
    
    def __init__(
        self,
        model: nn.Module,
        momentum: float = 0.996,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: Model to track (parameters will be copied)
            momentum: EMA momentum (higher = slower updates)
            device: Device to store EMA parameters
        """
        self.momentum = momentum
        self.device = device
        
        # Create a copy of the model for EMA
        self.ema_model = self._copy_model(model)
        self.ema_model.requires_grad_(False)  # EMA model is never trained directly
        
        if device is not None:
            self.ema_model = self.ema_model.to(device)
    
    def _copy_model(self, model: nn.Module) -> nn.Module:
        """Create a deep copy of the model."""
        import copy
        return copy.deepcopy(model)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA parameters.
        
        Args:
            model: Source model with updated parameters
        """
        # Handle DDP wrapper
        if hasattr(model, 'module'):
            model = model.module
        
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.momentum).add_(param.data, alpha=1 - self.momentum)
    
    def set_momentum(self, momentum: float):
        """Update the momentum value (for scheduling)."""
        self.momentum = momentum
    
    def forward(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'ema_model': self.ema_model.state_dict(),
            'momentum': self.momentum,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.momentum = state_dict['momentum']


def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
    warmup_value: float = 0.0,
) -> List[float]:
    """
    Create a cosine annealing schedule with optional warmup.
    
    Args:
        base_value: Value after warmup
        final_value: Final value at end of training
        epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        warmup_epochs: Number of warmup epochs
        warmup_value: Starting value during warmup
    
    Returns:
        List of values for each training step
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    
    schedule = []
    
    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            value = warmup_value + (base_value - warmup_value) * step / max(warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            value = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        schedule.append(value)
    
    return schedule


class CFMCurriculumScheduler:
    """
    Curriculum scheduler for CFM (Conditional Feature Modulation).
    
    The curriculum has three phases:
    
    Phase 1 - Backbone Only (epochs 0 to cfm_start_epoch):
        CFM weight = 0. Only the backbone trains with standard SSL.
        This allows the backbone to learn stable base representations first.
    
    Phase 2 - CFM Ramp-up (epochs cfm_start_epoch to cfm_full_epoch):
        CFM weight linearly increases from 0 to 1.
        Modulation: x_mod = (1 - w) * x_norm + w * (gamma * x_norm + beta)
        This gradually introduces CFM's influence.
    
    Phase 3 - Full CFM (epochs cfm_full_epoch onwards):
        CFM weight = 1. Full modulation is applied.
        Both backbone and CFM train together.
    
    This curriculum ensures the backbone first learns robust features before
    CFM learns to adapt them for different distributions.
    """
    
    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        cfm_start_epoch: float = 0.2,  # Fraction or absolute epoch
        cfm_full_epoch: float = 0.6,   # Fraction or absolute epoch
    ):
        """
        Args:
            total_epochs: Total number of training epochs
            steps_per_epoch: Number of steps per epoch
            cfm_start_epoch: When to start introducing CFM (fraction of total or absolute)
            cfm_full_epoch: When CFM reaches full strength (fraction of total or absolute)
        """
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        
        # Convert fractions to absolute epochs if needed
        if cfm_start_epoch < 1.0:
            self.cfm_start_epoch = int(cfm_start_epoch * total_epochs)
        else:
            self.cfm_start_epoch = int(cfm_start_epoch)
        
        if cfm_full_epoch < 1.0:
            self.cfm_full_epoch = int(cfm_full_epoch * total_epochs)
        else:
            self.cfm_full_epoch = int(cfm_full_epoch)
        
        # Convert to steps
        self.cfm_start_step = self.cfm_start_epoch * steps_per_epoch
        self.cfm_full_step = self.cfm_full_epoch * steps_per_epoch
        self.ramp_steps = self.cfm_full_step - self.cfm_start_step
        
    def get_cfm_weight(self, step: int) -> float:
        """
        Get the CFM weight for a given training step.
        
        Args:
            step: Current global training step
        
        Returns:
            CFM weight between 0 and 1
        """
        if step < self.cfm_start_step:
            # Phase 1: Backbone only
            return 0.0
        elif step < self.cfm_full_step:
            # Phase 2: Linear ramp-up
            progress = (step - self.cfm_start_step) / max(self.ramp_steps, 1)
            return progress
        else:
            # Phase 3: Full CFM
            return 1.0
    
    def get_phase(self, step: int) -> str:
        """Get the current training phase name."""
        if step < self.cfm_start_step:
            return "backbone_only"
        elif step < self.cfm_full_step:
            return "cfm_rampup"
        else:
            return "full_cfm"
    
    def should_train_cfm(self, step: int) -> bool:
        """Check if CFM should receive gradients at this step."""
        return step >= self.cfm_start_step
    
    def __repr__(self) -> str:
        return (f"CFMCurriculumScheduler("
                f"start_epoch={self.cfm_start_epoch}, "
                f"full_epoch={self.cfm_full_epoch}, "
                f"total_epochs={self.total_epochs})")


def apply_cfm_modulation_with_weight(
    x_normalized: torch.Tensor,
    modulation: Dict[str, torch.Tensor],
    cfm_weight: float,
) -> torch.Tensor:
    """
    Apply CFM modulation with curriculum weight.
    
    When cfm_weight = 0: returns x_normalized (no modulation)
    When cfm_weight = 1: returns gamma * x_normalized + beta (full modulation)
    When 0 < cfm_weight < 1: interpolates between the two
    
    Args:
        x_normalized: Normalized features (B, N, D) or (B, D)
        modulation: Dict with 'gamma' and 'beta' tensors
        cfm_weight: Weight between 0 and 1
    
    Returns:
        Modulated features
    """
    if cfm_weight == 0.0:
        return x_normalized
    
    gamma = modulation.get('gamma', None)
    beta = modulation.get('beta', None)
    
    # Compute full modulation
    x_modulated = x_normalized
    if gamma is not None:
        if gamma.dim() == 2 and x_normalized.dim() == 3:
            gamma = gamma.unsqueeze(1)
        x_modulated = x_modulated * gamma
    if beta is not None:
        if beta.dim() == 2 and x_normalized.dim() == 3:
            beta = beta.unsqueeze(1)
        x_modulated = x_modulated + beta
    
    if cfm_weight == 1.0:
        return x_modulated
    
    # Interpolate between no modulation and full modulation
    return (1 - cfm_weight) * x_normalized + cfm_weight * x_modulated


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Base Trainer Test")
    print("=" * 60)
    
    # Test EMAModel
    print("\n--- EMAModel Test ---")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleModel()
    ema = EMAModel(model, momentum=0.99)
    
    # Initial params should be equal
    for p1, p2 in zip(model.parameters(), ema.ema_model.parameters()):
        assert torch.allclose(p1, p2), "Initial params should match"
    print("[PASS] Initial EMA params match source")
    
    # Update model params
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    
    # Update EMA
    ema.update(model)
    print("[PASS] EMA update works")
    
    # Test state dict
    state = ema.state_dict()
    ema2 = EMAModel(SimpleModel(), momentum=0.5)
    ema2.load_state_dict(state)
    assert ema2.momentum == 0.99, "Momentum should be loaded"
    print("[PASS] EMA state dict save/load works")
    
    # Test cosine scheduler
    print("\n--- Cosine Scheduler Test ---")
    schedule = cosine_scheduler(
        base_value=1.0,
        final_value=0.1,
        epochs=10,
        steps_per_epoch=100,
        warmup_epochs=2,
        warmup_value=0.0,
    )
    
    print(f"Schedule length: {len(schedule)}")
    print(f"Warmup end value: {schedule[199]:.4f} (should be ~1.0)")
    print(f"Final value: {schedule[-1]:.4f} (should be ~0.1)")
    
    assert len(schedule) == 1000, "Should have 10*100=1000 steps"
    assert abs(schedule[199] - 1.0) < 0.01, "Warmup should end at base_value"
    assert abs(schedule[-1] - 0.1) < 0.01, "Should end at final_value"
    print("[PASS] Cosine scheduler works correctly")
    
    # Test CFMCurriculumScheduler
    print("\n--- CFMCurriculumScheduler Test ---")
    cfm_scheduler = CFMCurriculumScheduler(
        total_epochs=100,
        steps_per_epoch=100,
        cfm_start_epoch=0.2,  # 20%
        cfm_full_epoch=0.6,   # 60%
    )
    
    print(f"Scheduler: {cfm_scheduler}")
    print(f"Start epoch: {cfm_scheduler.cfm_start_epoch}")
    print(f"Full epoch: {cfm_scheduler.cfm_full_epoch}")
    
    # Test Phase 1 (backbone only)
    step_phase1 = 500  # Epoch 5, step 0
    weight_phase1 = cfm_scheduler.get_cfm_weight(step_phase1)
    phase1 = cfm_scheduler.get_phase(step_phase1)
    assert weight_phase1 == 0.0, f"Phase 1 weight should be 0, got {weight_phase1}"
    assert phase1 == "backbone_only", f"Phase should be backbone_only, got {phase1}"
    print(f"[PASS] Phase 1 (step {step_phase1}): weight={weight_phase1}, phase={phase1}")
    
    # Test Phase 2 (ramp-up) - middle of ramp
    step_phase2 = 4000  # Epoch 40, middle of 20-60 ramp
    weight_phase2 = cfm_scheduler.get_cfm_weight(step_phase2)
    phase2 = cfm_scheduler.get_phase(step_phase2)
    assert 0 < weight_phase2 < 1, f"Phase 2 weight should be between 0 and 1, got {weight_phase2}"
    assert phase2 == "cfm_rampup", f"Phase should be cfm_rampup, got {phase2}"
    print(f"[PASS] Phase 2 (step {step_phase2}): weight={weight_phase2:.3f}, phase={phase2}")
    
    # Test Phase 3 (full CFM)
    step_phase3 = 8000  # Epoch 80
    weight_phase3 = cfm_scheduler.get_cfm_weight(step_phase3)
    phase3 = cfm_scheduler.get_phase(step_phase3)
    assert weight_phase3 == 1.0, f"Phase 3 weight should be 1, got {weight_phase3}"
    assert phase3 == "full_cfm", f"Phase should be full_cfm, got {phase3}"
    print(f"[PASS] Phase 3 (step {step_phase3}): weight={weight_phase3}, phase={phase3}")
    
    # Test apply_cfm_modulation_with_weight
    print("\n--- CFM Modulation with Weight Test ---")
    x = torch.randn(4, 10, 384)  # (B, N, D)
    modulation = {
        'gamma': torch.ones(4, 384) * 2.0,  # Scale by 2
        'beta': torch.ones(4, 384) * 0.5,   # Shift by 0.5
    }
    
    # Weight = 0 (no modulation)
    out_w0 = apply_cfm_modulation_with_weight(x, modulation, cfm_weight=0.0)
    assert torch.allclose(out_w0, x), "Weight 0 should return unchanged input"
    print("[PASS] Weight 0: no modulation applied")
    
    # Weight = 1 (full modulation)
    out_w1 = apply_cfm_modulation_with_weight(x, modulation, cfm_weight=1.0)
    expected_w1 = x * 2.0 + 0.5
    assert torch.allclose(out_w1, expected_w1), "Weight 1 should apply full modulation"
    print("[PASS] Weight 1: full modulation applied")
    
    # Weight = 0.5 (interpolation)
    out_w05 = apply_cfm_modulation_with_weight(x, modulation, cfm_weight=0.5)
    expected_w05 = 0.5 * x + 0.5 * (x * 2.0 + 0.5)
    assert torch.allclose(out_w05, expected_w05), "Weight 0.5 should interpolate"
    print("[PASS] Weight 0.5: interpolation works")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nNote: Full trainer testing requires data and is done via train.py")