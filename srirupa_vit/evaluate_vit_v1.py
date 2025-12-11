"""
Standalone Evaluation Script for SSL Models
============================================

This script evaluates trained SSL models using k-NN classification on
downstream datasets. It loads a checkpoint, extracts features from the
frozen backbone, trains kNN on train split, evaluates on val split,
and generates Kaggle submission from test split.

Dataset Structure Expected:
    data/
    ├── train/              # Training images (with labels)
    ├── val/                # Validation images (with labels)  
    ├── test/               # Test images (NO labels)
    ├── train_labels.csv    # filename, class_id
    ├── val_labels.csv      # filename, class_id
    └── test_images.csv     # filename

Usage:
    # Evaluate and generate submission
    python scripts/evaluate.py \
        --checkpoint outputs/run_name/latest.pt \
        --data_dir /path/to/cub200 \
        --output_csv submission_cub200.csv

    # Evaluate on multiple datasets
    python scripts/evaluate.py \
        --checkpoint outputs/run_name/latest.pt \
        --cub200_dir /path/to/cub200 \
        --miniimagenet_dir /path/to/miniimagenet

The script supports:
- Single and multi-dataset evaluation
- Automatic k selection based on validation accuracy
- Kaggle submission file generation
- ViT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from timm import create_model

def exclude_from_wd_and_lars(param):
    """Return True to SKIP weight decay and LARS adaptation."""
    return True

def include_in_wd_and_lars(param):
    """Return False to APPLY weight decay and LARS adaptation."""
    return False

# =============================================================================
#                          DATASET CLASSES
# =============================================================================

class EvalImageDataset(Dataset):
    """
    Dataset for evaluation that loads images and optional labels.
    
    Handles both labeled splits (train/val) and unlabeled test split.
    """
    
    def __init__(
        self,
        image_dir: str,
        image_list: List[str],
        labels: Optional[List[int]] = None,
        crop_size: int = 96,
    ):
        """
        Args:
            image_dir: Directory containing images
            image_list: List of image filenames
            labels: List of class labels (None for test split)
            crop_size: Size to resize images to
        """
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.crop_size = crop_size
        
        # Simple normalization (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
        
        # Convert to tensor and normalize
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image = (image - self.mean) / self.std
        
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name


def collate_fn_labeled(batch):
    """Collate function for labeled data (train/val)."""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    return images, labels, filenames


def collate_fn_unlabeled(batch):
    """Collate function for unlabeled data (test)."""
    images = torch.stack([item[0] for item in batch])
    filenames = [item[1] for item in batch]
    return images, filenames

def extract_features(images, backbone, device):
    images = images.to(device)
    features = backbone(images)
    features = F.normalize(features, p=2.0, dim=1)
    
    return features.cpu().numpy()

def extract_features_from_dataloader(
    backbone, device,
    dataloader: DataLoader,
    split_name: str = 'train',
    has_labels: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]], List[str]]:
    """
    Extract features from all samples in a dataloader.
    
    Args:
        dataloader: DataLoader for the split
        split_name: Name of split (for progress bar)
        has_labels: Whether the dataloader returns labels
    
    Returns:
        features: numpy array (N, feature_dim)
        labels: list of labels (None if has_labels=False)
        filenames: list of filenames
    """
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if has_labels:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch
        
        # Extract features
        features = extract_features(images, backbone, device)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    print(f"  Extracted {features.shape[0]} features of dimension {features.shape[1]}")
    
    return features, labels, all_filenames


# =============================================================================
#                          KNN EVALUATION
# =============================================================================

def evaluate_knn(
    train_features: np.ndarray,
    train_labels: List[int],
    val_features: np.ndarray,
    val_labels: List[int],
    k_values: List[int],
) -> Dict[str, Any]:
    """
    Evaluate kNN classifier with multiple k values.
    
    Args:
        train_features: Training features (N_train, feature_dim)
        train_labels: Training labels
        val_features: Validation features (N_val, feature_dim)
        val_labels: Validation labels
        k_values: List of k values to try
    
    Returns:
        Dictionary with results for each k and best k info
    """
    print(f"\n{'='*60}")
    print("k-NN Evaluation")
    print('='*60)
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")
    print(f"  Feature dim: {train_features.shape[1]}")
    print(f"  k values: {k_values}")
    
    results = {
        'k_results': {},
        'best_k': None,
        'best_val_accuracy': 0.0,
    }
    
    best_classifier = None
    
    print(f"\n{'k':>5} | {'Train Acc':>10} | {'Val Acc':>10}")
    print("-" * 35)
    
    for k in k_values:
        # Skip if k is larger than training set
        if k > len(train_labels):
            print(f"{k:>5} | {'skipped (k > n_train)':^23}")
            continue
        
        # Train kNN
        classifier = KNeighborsClassifier(
            n_neighbors=k,
            weights='distance',  # Weight by inverse distance
            metric='cosine',     # Cosine similarity for normalized embeddings
            n_jobs=-1,
        )
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        train_acc = classifier.score(train_features, train_labels)
        val_acc = classifier.score(val_features, val_labels)
        
        print(f"{k:>5} | {train_acc*100:>9.2f}% | {val_acc*100:>9.2f}%")
        
        results['k_results'][k] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
        }
        
        # Track best
        if val_acc > results['best_val_accuracy']:
            results['best_val_accuracy'] = val_acc
            results['best_k'] = k
            best_classifier = classifier
    
    print("-" * 35)
    print(f"Best: k={results['best_k']} with val accuracy {results['best_val_accuracy']*100:.2f}%")
    
    return results, best_classifier


def create_submission(
    classifier: KNeighborsClassifier,
    test_features: np.ndarray,
    test_filenames: List[str],
    output_path: str,
    dataset_name: str = "dataset",
) -> None:
    """
    Generate Kaggle submission CSV.
    
    Args:
        classifier: Trained kNN classifier
        test_features: Test features (N_test, feature_dim)
        test_filenames: List of test image filenames
        output_path: Path to save submission CSV
        dataset_name: Name of dataset (for logging)
    """
    print(f"\n{'='*60}")
    print(f"Generating Submission: {dataset_name}")
    print('='*60)
    
    # Generate predictions
    print("Generating predictions on test set...")
    predictions = classifier.predict(test_features)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions.astype(int),
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\nSubmission file created: {output_path}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# =============================================================================
#                          MODEL BUILDING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate SSL models with k-NN classification and generate submission",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Checkpoint
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    
    # Single dataset evaluation
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--dataset_name', type=str, default='dataset',
        help='Name of the dataset (for logging and output naming)'
    )
    
    # Multi-dataset evaluation
    parser.add_argument(
        '--cub200_dir', type=str, default=None,
        help='Path to CUB-200 dataset directory'
    )
    parser.add_argument(
        '--miniimagenet_dir', type=str, default=None,
        help='Path to miniImageNet dataset directory'
    )
    parser.add_argument(
        '--sun397_dir', type=str, default=None,
        help='Path to SUN-397 dataset directory'
    )
    
    # Model configuration (usually inferred from checkpoint)
    parser.add_argument(
        '--backbone', type=str, default=None,
        choices=['convnextv2_tiny', 'wideresnet', 'vit'],
        help='Backbone architecture'
    )
    parser.add_argument(
        '--img_size', type=int, default=96,
        help='Input image size'
    )
    
    # k-NN configuration
    parser.add_argument(
        '--k_values', type=int, nargs='+', default=[1, 3, 5, 10, 20, 50, 100, 200],
        help='k values for k-NN evaluation'
    )
    
    # Data loading
    parser.add_argument(
        '--batch_size', type=int, default=1024,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of data loading workers'
    )
    
    # Output
    parser.add_argument(
        '--output_dir', type=str, default='./submissions',
        help='Directory to save submission files'
    )
    parser.add_argument(
        '--output_csv', type=str, default=None,
        help='Specific path for submission CSV (overrides output_dir naming)'
    )
    parser.add_argument(
        '--output_json', type=str, default=None,
        help='Path to save results JSON'
    )
    
    # Device
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device to run evaluation on'
    )
    
    return parser.parse_args()

from torchvision.models.vision_transformer import VisionTransformer

class SmallVisionTransformer(nn.Module):
    """
    Small Vision Transformer encoder (optimized for <100M params)
    
    Default config (~22M params):
    - hidden_dim: 384
    - num_layers: 8
    - num_heads: 6
    - mlp_dim: 1536
    """
    def __init__(
        self,
        image_size=96,
        patch_size=8,
        hidden_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_dim=1536,
    ):
        super().__init__()

        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=hidden_dim,
        )

        # Replace classifier head with identity
        self.vit.head = nn.Identity()
        
        self._init_weights()
        self.embed_dim = hidden_dim

    def _init_weights(self):
        """Xavier initialization for better training"""
        for m in self.vit.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.vit(x)  # [B, hidden_dim]


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """Load checkpoint and return state dictionary."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    return checkpoint

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def build_backbone(config: Dict[str, Any], device: str) -> nn.Module:
    """Build backbone model from configuration."""
    backbone_name = config.get('backbone', 'convnextv2_tiny')
    img_size = config.get('img_size', 96)
    
    print(f"\nBuilding backbone: {backbone_name}")
    print(f"  Image size: {img_size}")
    
    if backbone_name == 'convnextv2_tiny':
        backbone = create_model('convnextv2_tiny', pretrained=False, num_classes=0)
        backbone.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            LayerNorm(96, eps=1e-6, data_format="channels_first")
        )
    elif backbone_name == 'wideresnet':
        backbone = create_model('wide_resnet50_2', pretrained=False, num_classes=0)
    elif backbone_name == 'vit':
        backbone = SmallVisionTransformer()
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    backbone = backbone.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in backbone.parameters())
    print(f"  Parameters: {num_params:,}")
    
    return backbone


def load_model_weights(
    backbone: nn.Module,
    checkpoint: Dict[str, Any],
) -> None:
    """Load model weights from checkpoint."""
    print("\nLoading model weights...")
    
    # Try different key names for backbone weights
    backbone_keys = ['model', 'model_state', 'model_state_dict', 'backbone', 'backbone_state_dict']
    loaded_backbone = False
    
    for key in backbone_keys:
        if key in checkpoint:
            state_dict = checkpoint.get(key, checkpoint)
            new_state_dict = {}
            for key1, value in state_dict.items():
                if key1.startswith('encoder_q.'):
                    new_key = key1[10:] 
                    new_state_dict[new_key] = value
            backbone.load_state_dict(new_state_dict)
            print(f"  Backbone weights loaded (from '{key}')")
            loaded_backbone = True
            break
    
    if not loaded_backbone:
        print("  WARNING: No backbone weights found in checkpoint!")
        print(f"  Available keys: {list(checkpoint.keys())}")
        
    # Set to eval mode and freeze
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    
    print("All backbone weights frozen.")


def print_model_summary(backbone: nn.Module) -> None:
    """Print model parameter summary."""
    print("\n" + "=" * 60)
    print("Model Summary (Frozen for Evaluation)")
    print("=" * 60)
    
    backbone_params = sum(p.numel() for p in backbone.parameters())
    backbone_trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone: {backbone_params:,} params ({backbone_trainable:,} trainable)")
    
    total = backbone_params
    
    print(f"Total: {total:,} params")
    print("=" * 60)


# =============================================================================
#                          DATASET EVALUATION
# =============================================================================

def evaluate_dataset(
    backbone, device,
    data_dir: str,
    dataset_name: str,
    k_values: List[int],
    batch_size: int,
    num_workers: int,
    crop_size: int,
    output_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate on a single dataset: extract features, train kNN, evaluate, generate submission.
    
    Args:
        extractor: Feature extractor
        data_dir: Path to dataset directory
        dataset_name: Name of dataset
        k_values: List of k values to try
        batch_size: Batch size for feature extraction
        num_workers: Number of data loading workers
        crop_size: Image crop size
        output_csv: Path to save submission CSV (optional)
    
    Returns:
        Dictionary with evaluation results
    """
    data_dir = Path(data_dir)
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"Data directory: {data_dir}")
    print('='*60)
    
    # Load CSV files
    print("\nLoading dataset metadata...")
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_csv = data_dir / 'test_images.csv'
    
    has_test = test_csv.exists()
    if has_test:
        test_df = pd.read_csv(test_csv)
        print(f"  Train: {len(train_df)} images")
        print(f"  Val: {len(val_df)} images")
        print(f"  Test: {len(test_df)} images (unlabeled)")
    else:
        print(f"  Train: {len(train_df)} images")
        print(f"  Val: {len(val_df)} images")
        print(f"  Test: No test_images.csv found")
    
    print(f"  Classes: {train_df['class_id'].nunique()}")
    
    # Create datasets
    train_dataset = EvalImageDataset(
        image_dir=data_dir / 'train',
        image_list=train_df['filename'].tolist(),
        labels=train_df['class_id'].tolist(),
        crop_size=crop_size,
    )
    
    val_dataset = EvalImageDataset(
        image_dir=data_dir / 'val',
        image_list=val_df['filename'].tolist(),
        labels=val_df['class_id'].tolist(),
        crop_size=crop_size,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_labeled,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_labeled,
        pin_memory=True,
    )
    
    # Extract features
    train_features, train_labels, _ = extract_features_from_dataloader(
        backbone, device, train_loader, 'train', has_labels=True
    )
    val_features, val_labels, _ = extract_features_from_dataloader(
        backbone, device, val_loader, 'val', has_labels=True
    )
    
    # Evaluate kNN with different k values
    results, best_classifier = evaluate_knn(
        train_features, train_labels,
        val_features, val_labels,
        k_values,
    )
    
    results['dataset_name'] = dataset_name
    
    # Generate submission if test set exists and output path specified
    if has_test and output_csv is not None:
        test_dataset = EvalImageDataset(
            image_dir=data_dir / 'test',
            image_list=test_df['filename'].tolist(),
            labels=None,
            crop_size=crop_size,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_unlabeled,
            pin_memory=True,
        )
        
        test_features, _, test_filenames = extract_features_from_dataloader(
            backbone, device, test_loader, 'test', has_labels=False
        )
        
        create_submission(
            classifier=best_classifier,
            test_features=test_features,
            test_filenames=test_filenames,
            output_path=output_csv,
            dataset_name=dataset_name,
        )
        
        results['submission_path'] = output_csv
        results['submission_k'] = results['best_k']
    
    return results


# =============================================================================
#                          MAIN
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("SSL Model Evaluation with k-NN")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    # Get configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        if args.backbone is not None:
            config['backbone'] = args.backbone
    else:
        config = {
            'backbone': args.backbone or 'convnextv2_tiny',
            'img_size': args.img_size,
        }
    
    # Build models
    backbone = build_backbone(config, device)
    
    # Load weights
    load_model_weights(backbone, checkpoint)
    
    # Print summary
    print_model_summary(backbone)
    
    crop_size = config.get('img_size', 96)
    all_results = {}
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect datasets to evaluate
    datasets_to_eval = {}
    
    if args.data_dir is not None:
        datasets_to_eval[args.dataset_name] = args.data_dir
    
    if args.cub200_dir is not None:
        datasets_to_eval['CUB200'] = args.cub200_dir
    
    if args.miniimagenet_dir is not None:
        datasets_to_eval['miniImageNet'] = args.miniimagenet_dir
    
    if args.sun397_dir is not None:
        datasets_to_eval['SUN397'] = args.sun397_dir
    
    if not datasets_to_eval:
        print("\nERROR: No dataset specified!")
        print("Use --data_dir or --cub200_dir/--miniimagenet_dir/--sun397_dir")
        return
    
    # Evaluate each dataset
    for name, data_dir in datasets_to_eval.items():
        if not Path(data_dir).exists():
            print(f"\nWARNING: Dataset directory not found: {data_dir}")
            print(f"Skipping {name}")
            continue
        
        # Determine output CSV path
        if args.output_csv is not None and len(datasets_to_eval) == 1:
            output_csv = args.output_csv
        else:
            output_csv = output_dir / f"submission_{name.lower()}.csv"
        
        results = evaluate_dataset(
            backbone=backbone, device=device,
            data_dir=data_dir,
            dataset_name=name,
            k_values=args.k_values,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            crop_size=crop_size,
            output_csv=str(output_csv),
        )
        all_results[name] = results
    
    # Print final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Best k: {results['best_k']}")
        print(f"  Best val accuracy: {results['best_val_accuracy']*100:.2f}%")
        if 'submission_path' in results:
            print(f"  Submission: {results['submission_path']}")
    
    # Compute average across datasets
    if len(all_results) > 1:
        avg_acc = np.mean([r['best_val_accuracy'] for r in all_results.values()])
        print(f"\nAverage best val accuracy: {avg_acc*100:.2f}%")
    
    # Save results JSON if requested
    if args.output_json is not None:
        # Convert to serializable format
        serializable = {}
        for name, results in all_results.items():
            serializable[name] = {
                'best_k': results['best_k'],
                'best_val_accuracy': float(results['best_val_accuracy']),
                'k_results': {
                    str(k): {
                        'train_accuracy': float(v['train_accuracy']),
                        'val_accuracy': float(v['val_accuracy']),
                    }
                    for k, v in results['k_results'].items()
                },
            }
            if 'submission_path' in results:
                serializable[name]['submission_path'] = results['submission_path']
        
        with open(args.output_json, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # torch.serialization.safe_globals([__main__.include_in_wd_and_lars])
    # torch.serialization.safe_globals([__main__.exclude_from_wd_and_lars])
    main()