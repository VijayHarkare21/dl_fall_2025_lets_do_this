"""
Trainers Module
===============

Contains training recipes for different SSL methods.

- BaseTrainer: Abstract base class with common training infrastructure
- DINOv2Trainer: Self-distillation with multi-crop, iBOT, and KoLeo
- MoCoV3Trainer: Contrastive learning with momentum encoder

All trainers support:
- Single and multi-GPU training via torchrun
- CFM (Conditional Feature Modulation) integration with curriculum training
- Checkpoint saving/loading and training resumption
- W&B logging

DINOv2 Components:
- DINOLoss: CLS token self-distillation loss
- iBOTLoss: Masked patch token prediction loss
- KoLeoLoss: Uniform feature distribution regularizer

CFM Curriculum Training:
- Phase 1 (0-20% of training): Backbone only, CFM disabled
- Phase 2 (20-60% of training): Gradual CFM introduction (weight ramps 0 to 1)
- Phase 3 (60-100% of training): Full CFM modulation
"""

from .base_trainer import (
    BaseTrainer, 
    EMAModel, 
    cosine_scheduler, 
    CFMCurriculumScheduler,
    apply_cfm_modulation_with_weight,
)
from .dinov2_trainer import DINOv2Trainer, DINOLoss, iBOTLoss, KoLeoLoss
from .moco_trainer import MoCoV3Trainer, InfoNCELoss

__all__ = [
    # Base
    'BaseTrainer',
    'EMAModel',
    'cosine_scheduler',
    'CFMCurriculumScheduler',
    'apply_cfm_modulation_with_weight',
    # DINOv2
    'DINOv2Trainer',
    'DINOLoss',
    'iBOTLoss',
    'KoLeoLoss',
    # MoCo v3
    'MoCoV3Trainer',
    'InfoNCELoss',
]