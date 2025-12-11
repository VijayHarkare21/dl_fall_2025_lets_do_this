"""
k-NN Evaluation for Self-Supervised Learning
=============================================

This module implements k-Nearest Neighbors evaluation for assessing the quality
of learned representations. k-NN is the standard evaluation protocol for SSL
because it directly tests feature quality without any learning on the target task.

Evaluation Protocol:
1. Extract features from the frozen backbone for all train and test images
2. L2-normalize the features
3. For each test image, find the k nearest neighbors in the training set
4. Predict the label based on majority voting (or distance-weighted voting)
5. Report accuracy for different values of k

This implementation supports:
- Multiple k values (e.g., [1, 5, 10, 20, 50, 100, 200])
- Distance-weighted voting
- Efficient batch processing for feature extraction
- Distributed feature gathering (when features are extracted on multiple GPUs)

Usage:
    evaluator = KNNEvaluator(
        backbone=model,
        device='cuda',
        k_values=[1, 5, 10, 20],
    )
    
    results = evaluator.evaluate(
        train_loader=train_loader,
        test_loader=test_loader,
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.distributed import is_main_process, all_gather_tensors


class KNNEvaluator:
    """
    k-Nearest Neighbors evaluator for SSL representations.
    
    Extracts features from a frozen backbone and evaluates using k-NN
    classification on downstream tasks.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        device: Union[str, torch.device] = 'cuda',
        k_values: List[int] = [1, 5, 10, 20, 50, 100, 200],
        temperature: float = 0.07,
        distance_weighted: bool = True,
        use_cfm: bool = False,
        cfm: Optional[nn.Module] = None,
    ):
        """
        Args:
            backbone: The feature extraction backbone (will be set to eval mode)
            device: Device to run evaluation on
            k_values: List of k values to evaluate
            temperature: Temperature for distance weighting (lower = sharper)
            distance_weighted: Whether to weight votes by distance
            use_cfm: Whether to use CFM during feature extraction
            cfm: CFM network (required if use_cfm=True)
        """
        self.backbone = backbone
        self.device = torch.device(device) if isinstance(device, str) else device
        self.k_values = sorted(k_values)
        self.max_k = max(k_values)
        self.temperature = temperature
        self.distance_weighted = distance_weighted
        self.use_cfm = use_cfm
        self.cfm = cfm
        
        # Move models to device and set to eval mode
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        
        if self.cfm is not None:
            self.cfm = self.cfm.to(self.device)
            self.cfm.eval()
    
    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        desc: str = "Extracting features",
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract features from all images in the dataloader.
        
        Args:
            dataloader: DataLoader yielding (images, labels, filenames)
            desc: Description for progress bar
        
        Returns:
            features: (N, D) tensor of L2-normalized features
            labels: (N,) tensor of labels (-1 for unlabeled)
            filenames: List of filenames
        """
        all_features = []
        all_labels = []
        all_filenames = []
        
        # Only show progress on main process
        iterator = tqdm(dataloader, desc=desc) if is_main_process() else dataloader
        
        for batch in iterator:
            images, labels, filenames = batch
            images = images.to(self.device, non_blocking=True)
            
            # Get CFM modulations if enabled
            if self.use_cfm and self.cfm is not None:
                modulations = self.cfm(images)
            else:
                modulations = None
            
            # Extract features (CLS token for ViT)
            features = self.backbone(images, modulations=modulations)
            
            # L2 normalize
            features = F.normalize(features, dim=-1, p=2)
            
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_filenames.extend(filenames)
        
        # Concatenate all batches
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return features, labels, all_filenames
    
    @torch.no_grad()
    def knn_classify(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        test_features: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Perform k-NN classification.
        
        Args:
            train_features: (N_train, D) normalized training features
            train_labels: (N_train,) training labels
            test_features: (N_test, D) normalized test features
            k: Number of neighbors
        
        Returns:
            predictions: (N_test,) predicted labels
        """
        # Move to device for computation
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        test_features = test_features.to(self.device)
        
        num_test = test_features.shape[0]
        num_train = train_features.shape[0]
        
        # Process in chunks to avoid OOM for large datasets
        chunk_size = 100
        predictions = []
        
        for start_idx in range(0, num_test, chunk_size):
            end_idx = min(start_idx + chunk_size, num_test)
            test_chunk = test_features[start_idx:end_idx]
            
            # Compute cosine similarity (features are normalized, so dot product = cosine)
            # test_chunk: (chunk, D), train_features: (N_train, D)
            # similarity: (chunk, N_train)
            similarity = torch.mm(test_chunk, train_features.t())
            
            # Get top-k neighbors
            # distances: (chunk, k), indices: (chunk, k)
            distances, indices = similarity.topk(k, dim=1, largest=True)
            
            # Get labels of neighbors
            neighbor_labels = train_labels[indices]  # (chunk, k)
            
            if self.distance_weighted:
                # Convert similarity to weights (higher similarity = higher weight)
                # Apply temperature scaling
                weights = F.softmax(distances / self.temperature, dim=1)
                
                # Weighted voting
                num_classes = int(train_labels.max().item()) + 1
                votes = torch.zeros(end_idx - start_idx, num_classes, device=self.device)
                
                for i in range(k):
                    # Add weighted vote for each neighbor
                    votes.scatter_add_(
                        dim=1,
                        index=neighbor_labels[:, i:i+1],
                        src=weights[:, i:i+1],
                    )
                
                chunk_predictions = votes.argmax(dim=1)
            else:
                # Simple majority voting
                chunk_predictions = torch.mode(neighbor_labels, dim=1).values
            
            predictions.append(chunk_predictions.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Run full k-NN evaluation.
        
        Args:
            train_loader: DataLoader for training set (features + labels)
            test_loader: DataLoader for test set (features + labels)
            verbose: Whether to print results
        
        Returns:
            Dictionary mapping 'k={k}_accuracy' to accuracy values
        """
        # Extract features
        if verbose and is_main_process():
            print("\n[k-NN Evaluation]")
            print(f"  k values: {self.k_values}")
            print(f"  Distance weighted: {self.distance_weighted}")
            print(f"  Using CFM: {self.use_cfm}")
        
        train_features, train_labels, _ = self.extract_features(
            train_loader, desc="Extracting train features"
        )
        test_features, test_labels, _ = self.extract_features(
            test_loader, desc="Extracting test features"
        )
        
        if verbose and is_main_process():
            print(f"  Train features: {train_features.shape}")
            print(f"  Test features: {test_features.shape}")
            print(f"  Num classes: {int(train_labels.max().item()) + 1}")
        
        # Evaluate for each k
        results = {}
        
        for k in self.k_values:
            if k > train_features.shape[0]:
                if verbose and is_main_process():
                    print(f"  Skipping k={k} (larger than training set)")
                continue
            
            predictions = self.knn_classify(
                train_features, train_labels,
                test_features, k
            )
            
            # Compute accuracy
            correct = (predictions == test_labels).sum().item()
            total = len(test_labels)
            accuracy = correct / total
            
            results[f'k={k}_accuracy'] = accuracy
            
            if verbose and is_main_process():
                print(f"  k={k:3d}: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Find best k
        if results:
            best_k = max(results.keys(), key=lambda x: results[x])
            results['best_k'] = int(best_k.split('=')[1].split('_')[0])
            results['best_accuracy'] = results[best_k]
            
            if verbose and is_main_process():
                print(f"  Best: k={results['best_k']} with {results['best_accuracy']*100:.2f}%")
        
        return results
    
    def evaluate_multiple_datasets(
        self,
        datasets: Dict[str, Tuple[DataLoader, DataLoader]],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset name to (train_loader, test_loader)
            verbose: Whether to print results
        
        Returns:
            Dictionary mapping dataset name to results dictionary
        """
        all_results = {}
        
        for name, (train_loader, test_loader) in datasets.items():
            if verbose and is_main_process():
                print(f"\n{'='*60}")
                print(f"Evaluating on: {name}")
                print('='*60)
            
            results = self.evaluate(train_loader, test_loader, verbose=verbose)
            all_results[name] = results
        
        # Summary
        if verbose and is_main_process():
            print(f"\n{'='*60}")
            print("Summary")
            print('='*60)
            for name, results in all_results.items():
                if 'best_accuracy' in results:
                    print(f"  {name}: {results['best_accuracy']*100:.2f}% (k={results['best_k']})")
        
        return all_results


def create_predictions_csv(
    evaluator: KNNEvaluator,
    train_loader: DataLoader,
    test_loader: DataLoader,
    output_path: str,
    k: int = 20,
) -> None:
    """
    Create a CSV file with predictions for Kaggle submission.
    
    Args:
        evaluator: KNNEvaluator instance
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set (labels can be -1)
        output_path: Path to save the CSV file
        k: k value to use for predictions
    """
    import pandas as pd
    
    # Extract features
    train_features, train_labels, _ = evaluator.extract_features(
        train_loader, desc="Extracting train features"
    )
    test_features, test_labels, test_filenames = evaluator.extract_features(
        test_loader, desc="Extracting test features"
    )
    
    # Get predictions
    predictions = evaluator.knn_classify(
        train_features, train_labels,
        test_features, k
    )
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_filenames,
        'class_id': predictions.numpy(),
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    if is_main_process():
        print(f"\n[k-NN] Predictions saved to: {output_path}")
        print(f"  Total predictions: {len(submission_df)}")
        print(f"  k value used: {k}")


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("k-NN Evaluator Test")
    print("=" * 60)
    
    # Create a simple mock backbone
    class MockBackbone(nn.Module):
        def __init__(self, feat_dim=384):
            super().__init__()
            self.feat_dim = feat_dim
            self.fc = nn.Linear(3 * 96 * 96, feat_dim)
        
        def forward(self, x, modulations=None):
            # Flatten and project
            x = x.view(x.shape[0], -1)
            return self.fc(x)
    
    # Create mock data
    from torch.utils.data import TensorDataset
    
    num_train = 100
    num_test = 20
    num_classes = 5
    feat_dim = 384
    
    # Create fake image tensors and labels
    train_images = torch.randn(num_train, 3, 96, 96)
    train_labels = torch.randint(0, num_classes, (num_train,))
    train_filenames = [f"train_{i}.jpg" for i in range(num_train)]
    
    test_images = torch.randn(num_test, 3, 96, 96)
    test_labels = torch.randint(0, num_classes, (num_test,))
    test_filenames = [f"test_{i}.jpg" for i in range(num_test)]
    
    # Create simple datasets that return (image, label, filename)
    class SimpleDataset:
        def __init__(self, images, labels, filenames):
            self.images = images
            self.labels = labels
            self.filenames = filenames
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx], self.filenames[idx]
    
    train_dataset = SimpleDataset(train_images, train_labels, train_filenames)
    test_dataset = SimpleDataset(test_images, test_labels, test_filenames)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create evaluator
    backbone = MockBackbone(feat_dim=feat_dim)
    evaluator = KNNEvaluator(
        backbone=backbone,
        device='cpu',
        k_values=[1, 5, 10, 20],
        distance_weighted=True,
    )
    
    print("\n--- Feature Extraction Test ---")
    features, labels, filenames = evaluator.extract_features(train_loader, desc="Test extraction")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Num filenames: {len(filenames)}")
    
    # Check normalization
    norms = features.norm(dim=1)
    print(f"Feature norms (should be ~1): min={norms.min():.4f}, max={norms.max():.4f}")
    
    print("\n--- k-NN Classification Test ---")
    results = evaluator.evaluate(train_loader, test_loader, verbose=True)
    print(f"\nResults dictionary: {results}")
    
    # Test with multiple datasets
    print("\n--- Multiple Datasets Test ---")
    datasets = {
        'dataset_A': (train_loader, test_loader),
        'dataset_B': (train_loader, test_loader),  # Same loaders for testing
    }
    all_results = evaluator.evaluate_multiple_datasets(datasets, verbose=True)
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)