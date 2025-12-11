"""
Dataset Classes for Self-Supervised Learning
=============================================

This module provides dataset classes for:
1. SSL Pretraining: Loads unlabeled images from a folder
2. Evaluation: Loads labeled images from train/val/test splits with CSV labels

Dataset Structures Expected:

Pretrain set:
    train/
        image1.jpg
        image2.jpg
        ...

Evaluation sets (CUB200, miniImageNet):
    data_root/
        train/
        val/
        test/
        train_labels.csv    (columns: filename, class_id)
        val_labels.csv      (columns: filename, class_id)
        test_images.csv     (columns: filename)
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .augmentations import get_augmentation, EvalAugmentation


class SSLPretrainDataset(Dataset):
    """
    Dataset for self-supervised pretraining.
    
    Loads images from a folder without labels. Returns augmented views
    based on the specified augmentation strategy (multi-crop for DINOv2,
    two-view for MoCo v3).
    
    The dataset scans the folder for supported image formats and caches
    the file list for efficient repeated access.
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Path to directory containing images
            transform: Augmentation transform (e.g., MultiCropAugmentation)
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Scan for image files
        self.image_paths = self._scan_images()
        
        # Optionally limit samples
        if max_samples is not None and max_samples < len(self.image_paths):
            self.image_paths = self.image_paths[:max_samples]
        
        print(f"[SSLPretrainDataset] Found {len(self.image_paths)} images in {root_dir}")
    
    def _scan_images(self) -> List[Path]:
        """Scan directory for supported image files."""
        image_paths = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            image_paths.extend(self.root_dir.glob(f'*{ext}'))
            image_paths.extend(self.root_dir.glob(f'*{ext.upper()}'))
        
        # Sort for reproducibility
        image_paths = sorted(image_paths)
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.root_dir}")
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        """
        Load and transform an image.
        
        Args:
            idx: Image index
        
        Returns:
            If transform returns multiple views (multi-crop): List of tensors
            If transform returns two views (MoCo): Tuple of tensors
            If transform returns single tensor (eval): Single tensor
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Failed to load {img_path}: {e}")
            # Return a random different image instead of crashing
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transform
        if self.transform is not None:
            return self.transform(img)
        
        return img


class EvalDataset(Dataset):
    """
    Dataset for k-NN evaluation on labeled test sets.
    
    Loads images and labels from the specified split (train/val/test).
    Uses deterministic augmentation (resize + center crop) for consistent
    feature extraction.
    
    For test split without labels, returns -1 as the label placeholder.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        crop_size: int = 96,
    ):
        """
        Args:
            root_dir: Path to dataset root (containing train/, val/, test/, and CSVs)
            split: One of 'train', 'val', 'test'
            transform: Optional transform (defaults to EvalAugmentation)
            crop_size: Crop size for evaluation transform
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or EvalAugmentation(crop_size=crop_size)
        
        # Load metadata
        self.image_dir = self.root_dir / split
        self.filenames, self.labels = self._load_metadata()
        
        print(f"[EvalDataset] Loaded {len(self.filenames)} images from {split} split")
        if self.labels is not None:
            n_classes = len(set(self.labels))
            print(f"[EvalDataset] Number of classes: {n_classes}")
    
    def _load_metadata(self) -> Tuple[List[str], Optional[List[int]]]:
        """Load filenames and labels from CSV."""
        if self.split == 'test':
            csv_path = self.root_dir / 'test_images.csv'
            df = pd.read_csv(csv_path)
            filenames = df['filename'].tolist()
            labels = None  # Test set has no labels
        else:
            csv_path = self.root_dir / f'{self.split}_labels.csv'
            df = pd.read_csv(csv_path)
            filenames = df['filename'].tolist()
            labels = df['class_id'].tolist()
        
        return filenames, labels
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load an image with its label.
        
        Args:
            idx: Image index
        
        Returns:
            Tuple of (image_tensor, label, filename)
            For test split, label is -1
        """
        filename = self.filenames[idx]
        img_path = self.image_dir / filename
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[WARNING] Failed to load {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        # Get label (-1 for test split)
        label = self.labels[idx] if self.labels is not None else -1
        
        return img, label, filename


def collate_multicrop(batch):
    """
    Custom collate function for multi-crop augmentation.
    
    Multi-crop returns a list of crops per image. This function
    reorganizes them into a list of batched tensors, one per crop type.
    
    Args:
        batch: List of crop lists, each of length (n_global + n_local)
    
    Returns:
        List of tensors, each of shape (B, C, H, W)
    """
    # batch is a list of lists: [[crop1, crop2, ...], [crop1, crop2, ...], ...]
    n_crops = len(batch[0])
    
    # Stack each crop position across the batch
    return [torch.stack([sample[i] for sample in batch]) for i in range(n_crops)]


def collate_twoview(batch):
    """
    Custom collate function for two-view augmentation.
    
    Args:
        batch: List of (view1, view2) tuples
    
    Returns:
        Tuple of (view1_batch, view2_batch), each of shape (B, C, H, W)
    """
    view1s = torch.stack([sample[0] for sample in batch])
    view2s = torch.stack([sample[1] for sample in batch])
    return view1s, view2s


def get_pretrain_dataloader(
    root_dir: str,
    batch_size: int,
    aug_type: str = 'multicrop',
    n_global_crops: int = 2,
    n_local_crops: int = 6,
    global_crop_size: int = 96,
    local_crop_size: int = 48,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    seed: int = 42,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for SSL pretraining.
    
    Args:
        root_dir: Path to image folder
        batch_size: Batch size
        aug_type: 'multicrop' for DINOv2, 'twoview' for MoCo v3
        n_global_crops: Number of global crops (multi-crop only)
        n_local_crops: Number of local crops (multi-crop only)
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        num_workers: Number of data loading workers
        max_samples: Limit number of samples (for debugging)
        seed: Random seed for shuffling
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader configured for SSL pretraining
    """
    # Get augmentation
    transform = get_augmentation(
        aug_type=aug_type,
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        n_global_crops=n_global_crops,
        n_local_crops=n_local_crops,
    )
    
    # Create dataset
    dataset = SSLPretrainDataset(
        root_dir=root_dir,
        transform=transform,
        max_samples=max_samples,
    )
    
    # Select collate function
    if aug_type == 'multicrop':
        collate_fn = collate_multicrop
    elif aug_type == 'twoview':
        collate_fn = collate_twoview
    else:
        collate_fn = None
    
    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # Important for batch norm and consistent batch sizes
        generator=generator,
        persistent_workers=num_workers > 0,
        **kwargs
    )


def get_eval_dataloader(
    root_dir: str,
    split: str,
    batch_size: int,
    crop_size: int = 96,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for k-NN evaluation.
    
    Args:
        root_dir: Path to dataset root
        split: One of 'train', 'val', 'test'
        batch_size: Batch size
        crop_size: Crop size for evaluation
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader configured for evaluation
    """
    dataset = EvalDataset(
        root_dir=root_dir,
        split=split,
        crop_size=crop_size,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Keep all samples for evaluation
        **kwargs
    )


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import shutil
    
    print("=" * 60)
    print("Dataset Classes Test")
    print("=" * 60)
    
    # Create a temporary directory with dummy images
    temp_dir = tempfile.mkdtemp()
    pretrain_dir = Path(temp_dir) / "train"
    pretrain_dir.mkdir()
    
    eval_dir = Path(temp_dir) / "eval_data"
    eval_dir.mkdir()
    (eval_dir / "train").mkdir()
    (eval_dir / "val").mkdir()
    (eval_dir / "test").mkdir()
    
    try:
        # Create dummy pretrain images
        print("\n--- Creating dummy data ---")
        n_pretrain = 100
        for i in range(n_pretrain):
            img = Image.fromarray(
                (torch.rand(96, 96, 3).numpy() * 255).astype('uint8'),
                mode='RGB'
            )
            img.save(pretrain_dir / f"img_{i:04d}.jpg")
        print(f"Created {n_pretrain} pretrain images")
        
        # Create dummy eval images and CSVs
        n_train, n_val, n_test = 50, 20, 30
        n_classes = 5
        
        train_files, train_labels = [], []
        for i in range(n_train):
            fname = f"train_{i:04d}.jpg"
            img = Image.fromarray(
                (torch.rand(96, 96, 3).numpy() * 255).astype('uint8'),
                mode='RGB'
            )
            img.save(eval_dir / "train" / fname)
            train_files.append(fname)
            train_labels.append(i % n_classes)
        
        val_files, val_labels = [], []
        for i in range(n_val):
            fname = f"val_{i:04d}.jpg"
            img = Image.fromarray(
                (torch.rand(96, 96, 3).numpy() * 255).astype('uint8'),
                mode='RGB'
            )
            img.save(eval_dir / "val" / fname)
            val_files.append(fname)
            val_labels.append(i % n_classes)
        
        test_files = []
        for i in range(n_test):
            fname = f"test_{i:04d}.jpg"
            img = Image.fromarray(
                (torch.rand(96, 96, 3).numpy() * 255).astype('uint8'),
                mode='RGB'
            )
            img.save(eval_dir / "test" / fname)
            test_files.append(fname)
        
        # Create CSVs
        pd.DataFrame({'filename': train_files, 'class_id': train_labels}).to_csv(
            eval_dir / 'train_labels.csv', index=False
        )
        pd.DataFrame({'filename': val_files, 'class_id': val_labels}).to_csv(
            eval_dir / 'val_labels.csv', index=False
        )
        pd.DataFrame({'filename': test_files}).to_csv(
            eval_dir / 'test_images.csv', index=False
        )
        print(f"Created eval data: {n_train} train, {n_val} val, {n_test} test")
        
        # Test SSLPretrainDataset with MultiCrop
        print("\n--- SSLPretrainDataset (MultiCrop) ---")
        multicrop_transform = get_augmentation('multicrop', n_local_crops=4)
        pretrain_dataset = SSLPretrainDataset(
            root_dir=str(pretrain_dir),
            transform=multicrop_transform,
            max_samples=50,
        )
        crops = pretrain_dataset[0]
        print(f"Dataset size: {len(pretrain_dataset)}")
        print(f"Crops per image: {len(crops)}")
        print(f"Global crop shape: {crops[0].shape}")
        print(f"Local crop shape: {crops[2].shape}")
        
        # Test SSLPretrainDataset with TwoView
        print("\n--- SSLPretrainDataset (TwoView) ---")
        twoview_transform = get_augmentation('twoview')
        pretrain_dataset_2v = SSLPretrainDataset(
            root_dir=str(pretrain_dir),
            transform=twoview_transform,
        )
        view1, view2 = pretrain_dataset_2v[0]
        print(f"View 1 shape: {view1.shape}")
        print(f"View 2 shape: {view2.shape}")
        
        # Test EvalDataset
        print("\n--- EvalDataset ---")
        eval_dataset = EvalDataset(root_dir=str(eval_dir), split='train')
        img, label, fname = eval_dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Label: {label}")
        print(f"Filename: {fname}")
        
        # Test with test split (no labels)
        test_dataset = EvalDataset(root_dir=str(eval_dir), split='test')
        img, label, fname = test_dataset[0]
        print(f"Test image label: {label} (should be -1)")
        
        # Test DataLoaders
        print("\n--- DataLoaders ---")
        pretrain_loader = get_pretrain_dataloader(
            root_dir=str(pretrain_dir),
            batch_size=8,
            aug_type='multicrop',
            n_local_crops=4,
            num_workers=0,
            max_samples=50,
        )
        batch = next(iter(pretrain_loader))
        print(f"Pretrain batch: {len(batch)} crops")
        print(f"  Global crops: {batch[0].shape}, {batch[1].shape}")
        print(f"  Local crops: {batch[2].shape}")
        
        eval_loader = get_eval_dataloader(
            root_dir=str(eval_dir),
            split='train',
            batch_size=16,
            num_workers=0,
        )
        images, labels, filenames = next(iter(eval_loader))
        print(f"Eval batch: images {images.shape}, labels {labels.shape}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print("\nCleaned up temporary files.")