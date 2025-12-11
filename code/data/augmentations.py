"""
Augmentation Pipelines for Self-Supervised Learning
====================================================

This module implements the augmentation strategies used in DINOv2 and MoCo v3.
The key technique is multi-crop: generating multiple views of the same image
at different scales to learn both local and global features.

Multi-Crop Strategy (DINOv2):
- 2 global crops (96x96): capture full image context
- N local crops (48x48): capture fine-grained details
- All crops from the same image should have similar representations

Augmentation Pipeline:
1. Spatial: RandomResizedCrop, HorizontalFlip
2. Color: ColorJitter, Grayscale, GaussianBlur, Solarization
3. Normalization: ImageNet mean/std (standard for vision models)

References:
- DINOv2: "DINOv2: Learning Robust Visual Features without Supervision"
- DINO: "Emerging Properties in Self-Supervised Vision Transformers"
- MoCo v3: "An Empirical Study of Training Self-Supervised Vision Transformers"
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageOps
import random
import math


# ImageNet normalization (standard for vision models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianBlur:
    """
    Apply Gaussian blur with random sigma.
    
    Used in SSL to create views that differ in sharpness,
    forcing the model to learn blur-invariant features.
    """
    
    def __init__(self, sigma_min=0.1, sigma_max=2.0, p=0.5):
        """
        Args:
            sigma_min: Minimum blur sigma
            sigma_max: Maximum blur sigma  
            p: Probability of applying blur
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """
    Apply solarization (invert pixels above threshold).
    
    Creates a dramatic color transformation that helps the model
    learn color-invariant features. Used primarily in global crops.
    """
    
    def __init__(self, threshold=128, p=0.2):
        """
        Args:
            threshold: Pixel value threshold for inversion
            p: Probability of applying solarization
        """
        self.threshold = threshold
        self.p = p
    
    def __call__(self, img):
        if random.random() < self.p:
            img = ImageOps.solarize(img, threshold=self.threshold)
        return img


class MultiCropAugmentation:
    """
    Multi-crop augmentation strategy for DINOv2.
    
    Generates multiple views of the same image:
    - Global crops: Large crops that see most of the image (for teacher in DINO)
    - Local crops: Small crops that see image details (for student in DINO)
    
    The key insight is that local crops should match the representation of
    global crops, forcing the model to learn that local regions belong to
    the same semantic concept as the full image.
    """
    
    def __init__(
        self,
        global_crop_size=96,
        local_crop_size=48,
        n_global_crops=2,
        n_local_crops=6,
        global_crop_scale=(0.4, 1.0),
        local_crop_scale=(0.05, 0.4),
    ):
        """
        Args:
            global_crop_size: Size of global crops (should match backbone input)
            local_crop_size: Size of local crops (typically half of global)
            n_global_crops: Number of global crops (typically 2)
            n_local_crops: Number of local crops (typically 6-8)
            global_crop_scale: Scale range for global crops (min, max fraction of image)
            local_crop_scale: Scale range for local crops
        """
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        
        # Normalization (applied to all crops)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        # Global crop augmentation (stronger augmentation)
        # These are fed to both student and teacher in DINO
        self.global_transform = T.Compose([
            T.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(sigma_min=0.1, sigma_max=2.0, p=1.0),  # Always blur for global crop 1
        ])
        
        # Second global crop has different blur/solarization probability
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(
                global_crop_size,
                scale=global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(sigma_min=0.1, sigma_max=2.0, p=0.1),
            Solarization(p=0.2),
        ])
        
        # Local crop augmentation (less aggressive, no solarization)
        # These are only fed to the student in DINO
        self.local_transform = T.Compose([
            T.RandomResizedCrop(
                local_crop_size,
                scale=local_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(sigma_min=0.1, sigma_max=2.0, p=0.5),
        ])
    
    def __call__(self, img):
        """
        Generate multiple crops from a single image.
        
        Args:
            img: PIL Image
        
        Returns:
            List of tensors: [global_1, global_2, local_1, ..., local_n]
            First n_global_crops are global, rest are local.
        """
        crops = []
        
        # Global crops
        crops.append(self.normalize(self.global_transform(img)))
        if self.n_global_crops > 1:
            crops.append(self.normalize(self.global_transform_2(img)))
        for _ in range(2, self.n_global_crops):
            crops.append(self.normalize(self.global_transform(img)))
        
        # Local crops
        for _ in range(self.n_local_crops):
            crops.append(self.normalize(self.local_transform(img)))
        
        return crops


class TwoViewAugmentation:
    """
    Simple two-view augmentation for MoCo v3.
    
    MoCo v3 uses two augmented views of the same image:
    - Query view: fed to the online encoder + predictor
    - Key view: fed to the momentum encoder
    
    Both views use the same augmentation pipeline (unlike DINO's multi-crop).
    """
    
    def __init__(
        self,
        crop_size=96,
        crop_scale=(0.2, 1.0),
    ):
        """
        Args:
            crop_size: Output crop size
            crop_scale: Scale range for random resized crop
        """
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        # MoCo v3 augmentation (similar strength for both views)
        self.transform = T.Compose([
            T.RandomResizedCrop(
                crop_size,
                scale=crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(sigma_min=0.1, sigma_max=2.0, p=0.5),
            Solarization(p=0.0),  # MoCo v3 typically doesn't use solarization
        ])
    
    def __call__(self, img):
        """
        Generate two augmented views.
        
        Args:
            img: PIL Image
        
        Returns:
            Tuple of (view1, view2), each a normalized tensor
        """
        view1 = self.normalize(self.transform(img))
        view2 = self.normalize(self.transform(img))
        return view1, view2


class EvalAugmentation:
    """
    Simple augmentation for evaluation (k-NN testing).
    
    During evaluation, we want deterministic features, so we use
    minimal augmentation: just resize and center crop.
    """
    
    def __init__(self, crop_size=96):
        """
        Args:
            crop_size: Output size (should match training crop size)
        """
        self.transform = T.Compose([
            T.Resize(crop_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    def __call__(self, img):
        """
        Apply evaluation transform.
        
        Args:
            img: PIL Image
        
        Returns:
            Normalized tensor
        """
        return self.transform(img)


def get_augmentation(
    aug_type='multicrop',
    global_crop_size=96,
    local_crop_size=48,
    n_global_crops=2,
    n_local_crops=6,
    **kwargs
):
    """
    Factory function to get augmentation pipeline.
    
    Args:
        aug_type: Type of augmentation ('multicrop', 'twoview', 'eval')
        global_crop_size: Size for global crops
        local_crop_size: Size for local crops
        n_global_crops: Number of global crops
        n_local_crops: Number of local crops
        **kwargs: Additional arguments passed to augmentation class
    
    Returns:
        Augmentation callable
    """
    if aug_type == 'multicrop':
        return MultiCropAugmentation(
            global_crop_size=global_crop_size,
            local_crop_size=local_crop_size,
            n_global_crops=n_global_crops,
            n_local_crops=n_local_crops,
            **kwargs
        )
    elif aug_type == 'twoview':
        return TwoViewAugmentation(
            crop_size=global_crop_size,
            **kwargs
        )
    elif aug_type == 'eval':
        return EvalAugmentation(crop_size=global_crop_size)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Augmentation Pipeline Test")
    print("=" * 60)
    
    # Create a dummy image (random RGB)
    img = Image.fromarray(
        (torch.rand(96, 96, 3).numpy() * 255).astype('uint8'),
        mode='RGB'
    )
    print(f"\nInput image size: {img.size}")
    
    # Test MultiCropAugmentation
    print("\n--- MultiCropAugmentation (DINOv2) ---")
    multicrop = MultiCropAugmentation(
        global_crop_size=96,
        local_crop_size=48,
        n_global_crops=2,
        n_local_crops=6,
    )
    crops = multicrop(img)
    print(f"Number of crops: {len(crops)}")
    print(f"Global crops: {multicrop.n_global_crops} x {crops[0].shape}")
    print(f"Local crops: {multicrop.n_local_crops} x {crops[2].shape}")
    
    # Verify shapes
    for i, crop in enumerate(crops):
        expected_size = 96 if i < 2 else 48
        assert crop.shape == (3, expected_size, expected_size), \
            f"Crop {i} has wrong shape: {crop.shape}"
    print("All crop shapes correct")
    
    # Test TwoViewAugmentation
    print("\n--- TwoViewAugmentation (MoCo v3) ---")
    twoview = TwoViewAugmentation(crop_size=96)
    view1, view2 = twoview(img)
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    assert view1.shape == view2.shape == (3, 96, 96)
    print("Both views have correct shape")
    
    # Test EvalAugmentation
    print("\n--- EvalAugmentation ---")
    eval_aug = EvalAugmentation(crop_size=96)
    eval_out = eval_aug(img)
    print(f"Output shape: {eval_out.shape}")
    assert eval_out.shape == (3, 96, 96)
    print("Eval output shape correct")
    
    # Test factory function
    print("\n--- Factory Function ---")
    aug1 = get_augmentation('multicrop', n_local_crops=4)
    aug2 = get_augmentation('twoview')
    aug3 = get_augmentation('eval')
    print("Factory function works for all types")
    
    # Verify normalization (should be roughly zero mean, unit variance)
    print("\n--- Normalization Check ---")
    print(f"Crop mean: {crops[0].mean():.4f} (should be ~0)")
    print(f"Crop std: {crops[0].std():.4f} (should be ~1)")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)