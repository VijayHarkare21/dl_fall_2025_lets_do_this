"""
Logging Utilities for SSL Training
==================================

This module provides utilities for:
1. Weights & Biases (W&B) integration
2. Metric logging and tracking
3. Checkpoint management (save/load)
4. Training state management

Design Principles:
- Only the main process (rank 0) logs to W&B to avoid duplicate logs
- Checkpoints include full training state for resumption
- Metrics are aggregated across GPUs before logging
- All logs include step/epoch information for reproducibility

Usage:
    logger = Logger(
        project="ssl-vision",
        config=config_dict,
        enabled=True,
    )
    
    # Log metrics
    logger.log({'loss': 0.5, 'lr': 1e-4}, step=100)
    
    # Save checkpoint
    logger.save_checkpoint(model, optimizer, epoch, step, metrics)
    
    # Load checkpoint
    state = logger.load_checkpoint(path)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch
import torch.nn as nn

from .distributed import is_main_process, get_rank, get_world_size


class Logger:
    """
    Unified logging interface for SSL training.
    
    Handles W&B logging, console output, and checkpoint management.
    Automatically handles distributed training by only logging on rank 0.
    """
    
    def __init__(
        self,
        project: str = "ssl-vision-backbone",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
        enabled: bool = True,
        use_wandb: bool = True,
        log_every_n_steps: int = 50,
    ):
        """
        Args:
            project: W&B project name
            name: Run name (auto-generated if None)
            config: Configuration dictionary to log
            output_dir: Directory for checkpoints and local logs
            enabled: If False, disable all logging (for debugging)
            use_wandb: Whether to use W&B (can disable for local testing)
            log_every_n_steps: Frequency of console logging
        """
        self.project = project
        self.config = config or {}
        self.enabled = enabled and is_main_process()
        self.use_wandb = use_wandb
        self.log_every_n_steps = log_every_n_steps
        
        # Generate run name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"run_{timestamp}"
        self.name = name
        
        # Setup output directory
        self.output_dir = Path(output_dir) / name
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = self.output_dir / "checkpoints"
            self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize W&B
        self.wandb_run = None
        if self.enabled and self.use_wandb:
            self._init_wandb()
        
        # Metric history for local tracking
        self.metric_history = []
        self.start_time = time.time()
        
        if self.enabled:
            self._log_config()
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                dir=str(self.output_dir),
                resume="allow",
            )
            print(f"[Logger] W&B initialized: {wandb.run.url}")
        except ImportError:
            print("[Logger] wandb not installed, skipping W&B logging")
            self.use_wandb = False
        except Exception as e:
            print(f"[Logger] Failed to initialize W&B: {e}")
            self.use_wandb = False
    
    def _log_config(self):
        """Log configuration to file."""
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"[Logger] Config saved to {config_path}")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics to W&B and console.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Global training step
            epoch: Current epoch
            commit: Whether to commit the log (set False to batch logs)
        """
        if not self.enabled:
            return
        
        # Add step/epoch to metrics
        log_dict = dict(metrics)
        if step is not None:
            log_dict['step'] = step
        if epoch is not None:
            log_dict['epoch'] = epoch
        
        # Log to W&B
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.log(log_dict, step=step, commit=commit)
        
        # Store in history
        log_dict['timestamp'] = time.time() - self.start_time
        self.metric_history.append(log_dict)
    
    def log_console(
        self,
        message: str,
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """
        Print a formatted message to console.
        
        Args:
            message: Main message
            metrics: Optional metrics to display
            step: Global step
            epoch: Current epoch
        """
        if not self.enabled:
            return
        
        # Build prefix
        parts = []
        if epoch is not None:
            parts.append(f"Epoch {epoch}")
        if step is not None:
            parts.append(f"Step {step}")
        
        prefix = " | ".join(parts)
        if prefix:
            prefix = f"[{prefix}] "
        
        # Build metrics string
        metrics_str = ""
        if metrics:
            metrics_parts = [f"{k}: {v:.8f}" if isinstance(v, float) else f"{k}: {v}" 
                           for k, v in metrics.items()]
            metrics_str = " | " + " | ".join(metrics_parts)
        
        print(f"{prefix}{message}{metrics_str}")
    
    def log_model_info(self, model: nn.Module, name: str = "model"):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
            name: Model name for logging
        """
        if not self.enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        info = {
            f"{name}/total_params": total_params,
            f"{name}/trainable_params": trainable_params,
            f"{name}/frozen_params": frozen_params,
        }
        
        print(f"\n[Logger] {name} Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.config.update(info, allow_val_change=True)
    
    def log_dataset_info(
        self,
        name: str,
        num_samples: int,
        num_classes: Optional[int] = None,
        split: Optional[str] = None,
    ):
        """
        Log dataset information.
        
        Args:
            name: Dataset name
            num_samples: Number of samples
            num_classes: Number of classes (if labeled)
            split: Data split (train/val/test)
        """
        if not self.enabled:
            return
        
        prefix = f"{name}/{split}" if split else name
        
        info = {f"{prefix}/num_samples": num_samples}
        if num_classes is not None:
            info[f"{prefix}/num_classes"] = num_classes
        
        print(f"[Logger] Dataset {prefix}: {num_samples:,} samples", end="")
        if num_classes is not None:
            print(f", {num_classes} classes", end="")
        print()
        
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.config.update(info, allow_val_change=True)
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Any] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        is_best: bool = False,
    ) -> Optional[Path]:
        """
        Save training checkpoint.
        
        Args:
            model: Model to save (handles DDP wrapper automatically)
            optimizer: Optimizer state
            epoch: Current epoch
            step: Global step
            metrics: Current metrics
            scheduler: Optional LR scheduler
            additional_state: Any additional state to save
            filename: Custom filename (default: checkpoint_epoch{N}.pt)
            is_best: If True, also save as best_model.pt
        
        Returns:
            Path to saved checkpoint (None if not main process)
        """
        if not self.enabled:
            return None
        
        # Handle DDP wrapper
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        # Build checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'config': self.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_state is not None:
            checkpoint.update(additional_state)
        
        # Save checkpoint
        if filename is None:
            filename = f"checkpoint_epoch{epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"[Logger] Checkpoint saved: {checkpoint_path}")
        
        # Save as best model if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[Logger] Best model saved: {best_path}")
        
        # Also save latest (for easy resumption)
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        path: Union[str, Path],
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: str = 'cpu',
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
            model: Model to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            map_location: Device to map tensors to
        
        Returns:
            Checkpoint dictionary
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=map_location)
        
        if model is not None:
            # Handle DDP wrapper
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Logger] Loaded model state from {path}")
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"[Logger] Loaded optimizer state from {path}")
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"[Logger] Loaded scheduler state from {path}")
        
        return checkpoint
    
    def get_resume_path(self) -> Optional[Path]:
        """
        Get path to latest checkpoint for resuming training.
        
        Returns:
            Path to latest.pt if exists, else None
        """
        if not self.enabled:
            return None
        
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return latest_path
        return None
    
    def finish(self):
        """
        Finalize logging (call at end of training).
        
        Saves metric history and closes W&B run.
        """
        if not self.enabled:
            return
        
        # Save metric history
        history_path = self.output_dir / "metrics_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metric_history, f, indent=2)
        print(f"[Logger] Metrics history saved: {history_path}")
        
        # Close W&B
        if self.use_wandb and self.wandb_run is not None:
            import wandb
            wandb.finish()
            print("[Logger] W&B run finished")
        
        # Print summary
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"[Logger] Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")


class MetricTracker:
    """
    Track and aggregate metrics over time.
    
    Useful for computing epoch-level averages from batch-level metrics.
    
    Usage:
        tracker = MetricTracker()
        
        for batch in dataloader:
            loss = compute_loss(batch)
            tracker.update('loss', loss.item(), batch_size)
        
        epoch_metrics = tracker.compute()
        tracker.reset()
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self._sum = {}
        self._count = {}
    
    def update(self, name: str, value: float, count: int = 1):
        """
        Update a metric with a new value.
        
        Args:
            name: Metric name
            value: Metric value (will be weighted by count)
            count: Weight/count for this value (e.g., batch size)
        """
        if name not in self._sum:
            self._sum[name] = 0.0
            self._count[name] = 0
        
        self._sum[name] += value * count
        self._count[name] += count
    
    def compute(self) -> Dict[str, float]:
        """
        Compute weighted averages of all tracked metrics.
        
        Returns:
            Dictionary of metric_name -> average_value
        """
        return {
            name: self._sum[name] / max(self._count[name], 1)
            for name in self._sum
        }
    
    def get(self, name: str) -> float:
        """Get current average for a specific metric."""
        if name not in self._sum:
            return 0.0
        return self._sum[name] / max(self._count[name], 1)


class Timer:
    """
    Simple timer for profiling training steps.
    
    Usage:
        timer = Timer()
        
        timer.start('forward')
        output = model(input)
        timer.stop('forward')
        
        timer.start('backward')
        loss.backward()
        timer.stop('backward')
        
        print(timer.summary())
    """
    
    def __init__(self):
        self._starts = {}
        self._totals = {}
        self._counts = {}
    
    def start(self, name: str):
        """Start timing a section."""
        self._starts[name] = time.time()
    
    def stop(self, name: str):
        """Stop timing a section and accumulate."""
        if name not in self._starts:
            return
        
        elapsed = time.time() - self._starts[name]
        
        if name not in self._totals:
            self._totals[name] = 0.0
            self._counts[name] = 0
        
        self._totals[name] += elapsed
        self._counts[name] += 1
    
    def get_average(self, name: str) -> float:
        """Get average time for a section in milliseconds."""
        if name not in self._totals:
            return 0.0
        return (self._totals[name] / self._counts[name]) * 1000
    
    def summary(self) -> str:
        """Get formatted summary of all timings."""
        parts = [f"{name}: {self.get_average(name):.1f}ms" 
                for name in self._totals]
        return " | ".join(parts)
    
    def reset(self):
        """Reset all timings."""
        self._starts.clear()
        self._totals.clear()
        self._counts.clear()


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("=" * 60)
    print("Logging Utilities Test")
    print("=" * 60)
    
    # Create temp directory for test outputs
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Test Logger (without W&B)
        print("\n--- Logger Test ---")
        logger = Logger(
            project="test-project",
            name="test-run",
            config={'lr': 0.001, 'batch_size': 32},
            output_dir=temp_dir,
            use_wandb=False,  # Disable W&B for testing
        )
        
        # Log some metrics
        logger.log({'loss': 0.5, 'acc': 0.8}, step=100, epoch=1)
        logger.log({'loss': 0.4, 'acc': 0.85}, step=200, epoch=1)
        logger.log_console("Training progress", {'loss': 0.4}, step=200, epoch=1)
        
        # Test model info logging
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
        
        dummy_model = DummyModel()
        logger.log_model_info(dummy_model, "test_model")
        
        # Test dataset info logging
        logger.log_dataset_info("pretrain", num_samples=500000, split="train")
        logger.log_dataset_info("CUB200", num_samples=5994, num_classes=200, split="train")
        
        # Test checkpoint save/load
        print("\n--- Checkpoint Test ---")
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=0.001)
        
        checkpoint_path = logger.save_checkpoint(
            model=dummy_model,
            optimizer=optimizer,
            epoch=5,
            step=1000,
            metrics={'loss': 0.3, 'acc': 0.9},
            is_best=True,
        )
        
        # Load checkpoint
        loaded = logger.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from epoch {loaded['epoch']}, step {loaded['step']}")
        print(f"Loaded metrics: {loaded['metrics']}")
        
        # Finish logging
        logger.finish()
        
        # Test MetricTracker
        print("\n--- MetricTracker Test ---")
        tracker = MetricTracker()
        tracker.update('loss', 0.5, count=32)
        tracker.update('loss', 0.4, count=32)
        tracker.update('loss', 0.3, count=32)
        tracker.update('acc', 0.8, count=32)
        tracker.update('acc', 0.85, count=32)
        
        metrics = tracker.compute()
        print(f"Tracked metrics: {metrics}")
        print(f"Loss average: {tracker.get('loss'):.4f}")
        
        tracker.reset()
        print(f"After reset: {tracker.compute()}")
        
        # Test Timer
        print("\n--- Timer Test ---")
        timer = Timer()
        
        for _ in range(3):
            timer.start('computation')
            time.sleep(0.01)  # Simulate work
            timer.stop('computation')
        
        print(f"Timer summary: {timer.summary()}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("\nCleaned up temporary files.")