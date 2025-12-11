import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import torchvision.transforms as T
import torchvision.transforms.functional as TF

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

import torchvision.transforms as T

class EvalImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        image_list: List[str],
        labels: Optional[List[int]] = None,
        crop_size: int = 96,
        use_tta: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.use_tta = use_tta
        self.crop_size = crop_size
        
        # Standard Normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = T.Normalize(mean=mean, std=std)
        
        # Base resize: We need the image to be larger than the crop_size for TenCrop to work.
        # Standard practice is resize_dim = crop_size / 0.875 (e.g., 224 -> 256)
        resize_dim = int(crop_size / 0.875)

        if self.use_tta:
            # --- 10-CROP TTA ---
            self.transform = T.Compose([
                T.Resize(resize_dim),
                T.TenCrop(crop_size), # Returns tuple of 10 PIL images
                T.Lambda(lambda crops: torch.stack([
                    normalize(T.ToTensor()(crop)) for crop in crops
                ])) # Stacks into (10, 3, H, W)
            ])
        else:
            # --- SINGLE VIEW (Validation Standard) ---
            self.transform = T.Compose([
                T.Resize(resize_dim),
                T.CenterCrop(crop_size),
                T.ToTensor(),
                normalize
            ])

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        
        # Load PIL
        image = Image.open(img_path).convert('RGB')
        
        # Apply Transform (Returns Tensor)
        # If TTA: (10, 3, H, W)
        # If Not: (3, H, W)
        image_tensor = self.transform(image)
        
        if self.labels is not None:
            return image_tensor, self.labels[idx], img_name
        return image_tensor, img_name

def collate_fn_labeled(batch):
    images = torch.stack([item[0] for item in batch]) # If TTA: (B, 6, C, H, W)
    labels = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    return images, labels, filenames

def collate_fn_unlabeled(batch):
    images = torch.stack([item[0] for item in batch])
    filenames = [item[1] for item in batch]
    return images, filenames

# =============================================================================
#                          FEATURE EXTRACTION
# =============================================================================

def extract_features_multi_view(images, backbone, device):
    """
    Handle both single-view (B, C, H, W) and multi-view (B, V, C, H, W) inputs.
    """
    images = images.to(device)
    
    # Check if we have multi-view input
    if images.dim() == 5:
        B, V, C, H, W = images.shape
        # Flatten views into batch dimension: (B*V, C, H, W)
        images = images.view(B * V, C, H, W)
        
        features = backbone(images) # (B*V, D)
        
        # Normalize individual views first (Crucial for Barlow Twins)
        features = F.normalize(features, p=2.0, dim=1)
        
        # Reshape back to (B, V, D)
        features = features.view(B, V, -1)
        
        # Average pooling across views
        features = features.mean(dim=1) # (B, D)
        
        # Re-normalize the averaged representation
        features = F.normalize(features, p=2.0, dim=1)
        
    else:
        # Standard single view
        features = backbone(images)
        features = F.normalize(features, p=2.0, dim=1)
    
    return features.cpu().numpy()

def extract_features_from_dataloader(
    backbone, device,
    dataloader: DataLoader,
    split_name: str = 'train',
    has_labels: bool = True,
) -> Tuple[np.ndarray, Optional[List[int]], List[str]]:
    
    all_features = []
    all_labels = []
    all_filenames = []
    
    print(f"\nExtracting features from {split_name} set (TTA Enabled)...")
    
    for batch in tqdm(dataloader, desc=f"{split_name} features"):
        if has_labels:
            images, labels, filenames = batch
            all_labels.extend(labels)
        else:
            images, filenames = batch
        
        # Use the multi-view extraction logic
        features = extract_features_multi_view(images, backbone, device)
        all_features.append(features)
        all_filenames.extend(filenames)
    
    features = np.concatenate(all_features, axis=0)
    labels = all_labels if all_labels else None
    
    return features, labels, all_filenames

# =============================================================================
#                          AdaNPC ADAPTATION
# =============================================================================

def adanpc_refinement(
    train_feats: np.ndarray,
    train_labels: List[int],
    test_feats: np.ndarray,
    k: int = 50,
    threshold: float = 0.9, # Confidence threshold
    dataset_name: str = "dataset"
) -> np.ndarray:
    """
    Performs AdaNPC-style refinement.
    1. Train kNN on Support Set (Train+Val).
    2. Predict Test Set.
    3. Select confident test predictions.
    4. Add them to Support Set.
    5. Re-train kNN and re-predict.
    """
    print(f"\n{'='*40}")
    print(f"Running AdaNPC Adaptation for {dataset_name}")
    print(f"Initial Memory Size: {len(train_labels)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*40}")

    # Convert labels to numpy
    memory_feats = train_feats.copy()
    memory_labels = np.array(train_labels).copy()
    
    # 1. Initial Classifier
    knn = KNeighborsClassifier(
        n_neighbors=k, weights='distance', metric='cosine', n_jobs=-1
    )
    knn.fit(memory_feats, memory_labels)
    
    # 2. Get Soft Predictions
    print("  Step 1: Initial prediction...")
    # predict_proba returns (N_test, N_classes)
    probs = knn.predict_proba(test_feats)
    
    # 3. Find Confident Samples
    max_probs = np.max(probs, axis=1)
    pseudo_labels = np.argmax(probs, axis=1)
    
    confident_mask = max_probs >= threshold
    num_confident = np.sum(confident_mask)
    
    print(f"  Step 2: Found {num_confident} / {len(test_feats)} confident test samples ({(num_confident/len(test_feats))*100:.1f}%)")
    
    if num_confident > 0:
        # 4. Update Memory Bank
        print("  Step 3: Augmenting memory bank...")
        
        # Get confident features and their pseudo-labels
        conf_feats = test_feats[confident_mask]
        conf_labels = pseudo_labels[confident_mask]
        
        # Stack
        memory_feats = np.vstack([memory_feats, conf_feats])
        memory_labels = np.concatenate([memory_labels, conf_labels])
        
        print(f"  New Memory Size: {len(memory_labels)}")
        
        # 5. Re-fit and Re-predict
        print("  Step 4: Refinement prediction...")
        knn.fit(memory_feats, memory_labels)
        
        # We return the NEW predictions
        final_preds = knn.predict(test_feats)
    else:
        print("  No confident samples found. Returning initial predictions.")
        final_preds = pseudo_labels
        
    return final_preds

# =============================================================================
#                          MODEL & UTILS
# =============================================================================

class LayerNorm(nn.Module):
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
    backbone_name = config.get('backbone', 'convnextv2_tiny')
    if backbone_name == 'convnextv2_tiny':
        backbone = create_model('convnextv2_tiny', pretrained=False, num_classes=0)
        backbone.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=1),
            LayerNorm(96, eps=1e-6, data_format="channels_first")
        )
    elif backbone_name == 'wideresnet':
        backbone = create_model('wide_resnet50_2', pretrained=False, num_classes=0)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    backbone = backbone.to(device)
    backbone.eval()
    return backbone

def load_model_weights(backbone: nn.Module, checkpoint: Dict[str, Any]) -> None:
    backbone_keys = ['model_state', 'model_state_dict', 'backbone', 'backbone_state_dict']
    for key in backbone_keys:
        if key in checkpoint:
            backbone.load_state_dict(checkpoint[key])
            break
    for param in backbone.parameters():
        param.requires_grad = False

# =============================================================================
#                          EVALUATION LOGIC
# =============================================================================

def evaluate_dataset(
    backbone1, backbone2, device,
    data_dir: str,
    dataset_name: str,
    k_values: List[int],
    batch_size: int,
    num_workers: int,
    crop_size: int,
    output_csv: Optional[str] = None,
    adanpc_threshold: float = 0.9,
) -> Dict[str, Any]:
    
    data_dir = Path(data_dir)
    print(f"\nProcessing {dataset_name} with TTA + Ensemble + AdaNPC...")
    
    # Load Lists
    train_df = pd.read_csv(data_dir / 'train_labels.csv')
    val_df = pd.read_csv(data_dir / 'val_labels.csv')
    test_csv = data_dir / 'test_images.csv'
    has_test = test_csv.exists()
    
    if has_test:
        test_df = pd.read_csv(test_csv)
    
    # --- DATASETS WITH TTA ENABLED ---
    # We use TTA=True for all splits to get robust features
    train_dataset = EvalImageDataset(
        data_dir / 'train', train_df['filename'].tolist(), train_df['class_id'].tolist(), crop_size, use_tta=True
    )
    val_dataset = EvalImageDataset(
        data_dir / 'val', val_df['filename'].tolist(), val_df['class_id'].tolist(), crop_size, use_tta=True
    )
    
    # Note: TTA increases memory usage, so we might need smaller batch size if GPU OOM
    # If 10 views, effective batch size is batch_size * 10.
    eff_batch_size = max(1, batch_size // 10)
    
    train_loader = DataLoader(train_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_labeled)
    val_loader = DataLoader(val_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_labeled)

    # --- FEATURE EXTRACTION (Multi-View TTA) ---
    print("Extracting features (Backbone 1: ConvNeXtV2)...")
    tr_f1, tr_l, _ = extract_features_from_dataloader(backbone1, device, train_loader, 'train')
    val_f1, val_l, _ = extract_features_from_dataloader(backbone1, device, val_loader, 'val')

    print("Extracting features (Backbone 2: WideResNet)...")
    tr_f2, _, _ = extract_features_from_dataloader(backbone2, device, train_loader, 'train')
    val_f2, _, _ = extract_features_from_dataloader(backbone2, device, val_loader, 'val')
    
    # --- FEATURE ENSEMBLING (Concatenation) ---
    # Features are already L2 normalized individually in the extract function
    train_features = np.concatenate((tr_f1, tr_f2), axis=1)
    val_features = np.concatenate((val_f1, val_f2), axis=1)
    
    # Normalize the combined feature
    train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    val_features = val_features / np.linalg.norm(val_features, axis=1, keepdims=True)
    
    train_labels = tr_l
    val_labels = val_l

    print(f"\n{'='*60}")
    print("k-NN Evaluation")
    print('='*60)
    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")
    print(f"  Feature dim: {train_features.shape[1]}")
    print(f"  k values: {k_values}")

    # --- VALIDATION (Check best K) ---
    # Merge Train + Val for final training if performing AdaNPC, 
    # but first let's check K on validation split.
    print(f"\nChecking K on Validation Set...")
    best_k = 200
    best_acc = 0.0
    
    for k in k_values:
        if k > len(train_labels): continue
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='cosine', n_jobs=-1)
        knn.fit(train_features, train_labels)
        acc = knn.score(val_features, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_k = k
        print(f"k={k} | Val Acc: {acc*100:.2f}%")
    
    print(f"Selected Best k: {best_k}")

    results = {
        'best_k': best_k,
        'best_val_accuracy': best_acc
    }

    print("-" * 35)
    print(f"Best: k={results['best_k']} with val accuracy {results['best_val_accuracy']*100:.2f}%")

    print(f"\n{'='*60}")
    print(f"Generating Submission: {dataset_name}")
    print('='*60)
    
    # Generate predictions
    print("Generating predictions on test set...")

    # --- TEST PREDICTION + AdaNPC ---
    if has_test and output_csv:
        test_dataset = EvalImageDataset(
            data_dir / 'test', test_df['filename'].tolist(), None, crop_size, use_tta=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=eff_batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_unlabeled
        )
        
        # Extract Test Features
        test_f1, _, test_filenames = extract_features_from_dataloader(backbone1, device, test_loader, 'test', has_labels=False)
        test_f2, _, _ = extract_features_from_dataloader(backbone2, device, test_loader, 'test', has_labels=False)
        
        test_features = np.concatenate((test_f1, test_f2), axis=1)
        test_features = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
        
        # Prepare Memory Bank (Train + Val)
        full_memory_feats = np.concatenate([train_features, val_features], axis=0)
        full_memory_labels = np.concatenate([train_labels, val_labels], axis=0)
        
        # Run AdaNPC
        final_predictions = adanpc_refinement(
            train_feats=full_memory_feats,
            train_labels=full_memory_labels,
            test_feats=test_features,
            k=best_k,
            threshold=adanpc_threshold,
            dataset_name=dataset_name
        )
        
        # Save Submission
        submission_df = pd.DataFrame({
            'id': test_filenames,
            'class_id': final_predictions.astype(int)
        })
        submission_df.to_csv(output_csv, index=False)
        print(f"Submission saved to {output_csv}")
        results['submission_path'] = output_csv

        print(f"\nSubmission file created: {output_csv}")
        print(f"Total predictions: {len(submission_df)}")
        print(f"\nClass distribution in predictions:")
        print(submission_df['class_id'].value_counts().head(10))
        
        # Validate submission format
        print(f"\nValidating submission format...")
        assert list(submission_df.columns) == ['id', 'class_id'], "Invalid columns!"
        assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
        assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
        print("✓ Submission format is valid!")

    return results

# =============================================================================
#                          MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TTA Evaluation Script")
    parser.add_argument('--checkpoint1', type=str, required=True)
    parser.add_argument('--checkpoint2', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='dataset')
    
    # Dataset specific args
    parser.add_argument('--cub200_dir', type=str, default=None)
    parser.add_argument('--miniimagenet_dir', type=str, default=None)
    parser.add_argument('--sun397_dir', type=str, default=None)

    parser.add_argument('--backbone1', type=str, default='convnextv2_tiny')
    parser.add_argument('--backbone2', type=str, default='wideresnet')
    parser.add_argument('--img_size', type=int, default=96)
    parser.add_argument('--k_values', type=int, nargs='+', default=[10, 20, 50, 100, 200])
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--output_dir', type=str, default='./submissions')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--adanpc_thresh', type=float, default=0.9, help="Threshold for AdaNPC memory expansion")
    
    return parser.parse_args()

def print_model_summary(backbone1: nn.Module, backbone2: nn.Module) -> None:
    """Print model parameter summary."""
    print("\n" + "=" * 60)
    print("Model Summary (Frozen for Evaluation)")
    print("=" * 60)
    
    backbone1_params = sum(p.numel() for p in backbone1.parameters())
    backbone1_trainable = sum(p.numel() for p in backbone1.parameters() if p.requires_grad)
    print(f"Backbone1: {backbone1_params:,} params ({backbone1_trainable:,} trainable)")
    
    backbone2_params = sum(p.numel() for p in backbone2.parameters())
    backbone2_trainable = sum(p.numel() for p in backbone2.parameters() if p.requires_grad)
    print(f"Backbone2: {backbone2_params:,} params ({backbone2_trainable:,} trainable)")
    
    total = backbone1_params + backbone2_params
    
    print(f"Total: {total:,} params")
    print("=" * 60)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Models
    c1 = torch.load(args.checkpoint1, map_location=device, weights_only=False)
    c2 = torch.load(args.checkpoint2, map_location=device, weights_only=False)
    
    b1 = build_backbone({'backbone': args.backbone1}, device)
    b2 = build_backbone({'backbone': args.backbone2}, device)
    
    load_model_weights(b1, c1)
    load_model_weights(b2, c2)

    print_model_summary(b1, b2)
    
    # Collect Datasets
    datasets = {}
    if args.data_dir: datasets[args.dataset_name] = args.data_dir
    if args.cub200_dir: datasets['CUB200'] = args.cub200_dir
    if args.miniimagenet_dir: datasets['miniImageNet'] = args.miniimagenet_dir
    if args.sun397_dir: datasets['SUN397'] = args.sun397_dir
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, d_dir in datasets.items():
        if args.output_csv and len(datasets) == 1:
            out_path = args.output_csv
        else:
            out_path = str(output_dir / f"submission_{name.lower()}.csv")
            
        evaluate_dataset(
            b1, b2, device, d_dir, name, args.k_values, 
            args.batch_size, 4, args.img_size, out_path,
            adanpc_threshold=args.adanpc_thresh
        )

if __name__ == "__main__":
    main()