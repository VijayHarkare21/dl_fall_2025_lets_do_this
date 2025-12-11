"""
Models Module
=============

Contains model architecture definitions only (no training logic).

Backbones:
- VisionTransformer (ViT): Attention-based, good for fine-grained features
- ConvNeXtV2: Pure convolutional, faster training, designed for SSL

Heads:
- DINOv2Head: Projection head for DINO self-distillation
- iBOTHead: Head for masked patch prediction
- MoCoV3Head/Predictor: Projection and prediction heads for MoCo v3

CFM (Conditional Feature Modulation):
- CFMNetwork: Adaptive feature modulation for distribution robustness
"""

from .vit import VisionTransformer, vit_small, vit_tiny
from .convnext import (
    ConvNeXtV2,
    convnext_tiny,
    convnext_small,
    convnext_base,
)
from .heads import DINOv2Head, iBOTHead, MoCoV3Head, MoCoV3Predictor
from .cfm import ContextEncoder, ModulationPredictor, CFMNetwork, CFMWrapper

__all__ = [
    # ViT Backbones
    'VisionTransformer',
    'vit_small',
    'vit_tiny',
    # ConvNeXt-V2 Backbones
    'ConvNeXtV2',
    'convnext_tiny',
    'convnext_small',
    'convnext_base',
    # Heads
    'DINOv2Head',
    'iBOTHead',
    'MoCoV3Head',
    'MoCoV3Predictor',
    # CFM
    'ContextEncoder',
    'ModulationPredictor', 
    'CFMNetwork',
    'CFMWrapper',
]