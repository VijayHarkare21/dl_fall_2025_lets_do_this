"""
Data Module
===========

Contains dataset classes and augmentation pipelines for SSL training and evaluation.
"""

from .augmentations import (
    GaussianBlur,
    Solarization,
    MultiCropAugmentation,
    TwoViewAugmentation,
    EvalAugmentation,
    get_augmentation,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .datasets import (
    SSLPretrainDataset,
    EvalDataset,
    collate_multicrop,
    collate_twoview,
    get_pretrain_dataloader,
    get_eval_dataloader,
)

__all__ = [
    # Augmentations
    'GaussianBlur',
    'Solarization',
    'MultiCropAugmentation',
    'TwoViewAugmentation',
    'EvalAugmentation',
    'get_augmentation',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    # Datasets
    'SSLPretrainDataset',
    'EvalDataset',
    'collate_multicrop',
    'collate_twoview',
    'get_pretrain_dataloader',
    'get_eval_dataloader',
]