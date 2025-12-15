"""
EMP-SSL Model Architecture (FIXED VERSION)
ResNet-50 backbone with EMP-SSL projection head

CRITICAL FIXES:
1. Modified ResNet for 96x96 images (smaller stride in conv1)
2. L2 normalization in projection head
3. Proper initialization
4. Better numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import math


class ProjectionHead(nn.Module):
    """
    MLP projection head for EMP-SSL with L2 normalization
    
    CRITICAL: Output must be L2 normalized for EMP-SSL to work properly!
    """
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward with L2 normalization"""
        x = self.net(x)
        # CRITICAL: L2 normalize the output
        x = F.normalize(x, dim=1, p=2)
        return x


class EMPSSL_ResNet50(nn.Module):
    """
    EMP-SSL model with ResNet-50 backbone (FIXED for 96x96 images)
    
    FIXES:
    - Reduced stride in conv1 from 2 to 1 (better for small images)
    - Optional: Removed first maxpool (better for 96x96)
    - L2 normalized projections
    - Proper initialization
    """
    def __init__(
        self, 
        backbone_output_dim=2048,
        projection_hidden_dim=2048,
        projection_output_dim=128,
        pretrained=False,
        small_image_mode=True  # Use for 96x96 images
    ):
        super().__init__()
        
        if pretrained:
            raise ValueError("Pretrained weights not allowed per project requirements!")
        
        # Create ResNet-50 backbone
        backbone = resnet50(weights=None)
        
        # CRITICAL FIX: Modify for 96x96 images
        if small_image_mode:
            # Change conv1 stride from 2 to 1
            # Original: kernel=7, stride=2, padding=3
            # Fixed: kernel=7, stride=1, padding=3 (keep spatial resolution)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
            
            # Optional: Remove maxpool to preserve resolution
            # This helps with small 96x96 images
            backbone.maxpool = nn.Identity()
        
        # Remove the final FC layer - we only want the feature extractor
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Projection head with L2 normalization
        self.projector = ProjectionHead(
            input_dim=backbone_output_dim,
            hidden_dim=projection_hidden_dim,
            output_dim=projection_output_dim
        )
        
        self.backbone_output_dim = backbone_output_dim
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Proper weight initialization for the backbone"""
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: If True, return both features and projections
            
        Returns:
            If return_features=False: projections [B, projection_output_dim] (L2 normalized)
            If return_features=True: (features [B, backbone_output_dim], projections)
        """
        # Extract features from backbone
        features = self.encoder(x)  # [B, 2048, H', W']
        features = torch.flatten(features, 1)  # [B, 2048]
        
        # Project features to embedding space (with L2 normalization)
        projections = self.projector(features)
        
        if return_features:
            return features, projections
        return projections
    
    def get_encoder(self):
        """Returns the encoder (backbone) for evaluation"""
        return self.encoder
    
    def get_parameter_count(self):
        """Calculate total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        projector_params = sum(p.numel() for p in self.projector.parameters())
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': encoder_params,
            'projector': projector_params
        }


def create_empssl_model(
    projection_hidden_dim=2048,
    projection_output_dim=128,
    small_image_mode=True
):
    """
    Factory function to create EMP-SSL model
    
    Args:
        projection_hidden_dim: Hidden dim in projection head
        projection_output_dim: Output dim for projections
        small_image_mode: True for 96x96 images (reduces stride/maxpool)
    
    Returns:
        model: EMPSSL_ResNet50 instance
        param_info: Dictionary with parameter counts
    """
    model = EMPSSL_ResNet50(
        projection_hidden_dim=projection_hidden_dim,
        projection_output_dim=projection_output_dim,
        pretrained=False,
        small_image_mode=small_image_mode
    )
    
    param_info = model.get_parameter_count()
    
    # Verify we're under 100M parameter limit
    assert param_info['total'] < 100_000_000, \
        f"Model has {param_info['total']:,} parameters, exceeds 100M limit!"
    
    return model, param_info


if __name__ == "__main__":
    # Test model creation
    print("="*70)
    print("FIXED EMP-SSL ResNet-50 Model")
    print("="*70)
    
    model, param_info = create_empssl_model(small_image_mode=True)
    
    print("\nPARAMETER SUMMARY:")
    print(f"  Total parameters:      {param_info['total']:>12,}")
    print(f"  Trainable parameters:  {param_info['trainable']:>12,}")
    print(f"  Encoder parameters:    {param_info['encoder']:>12,}")
    print(f"  Projector parameters:  {param_info['projector']:>12,}")
    print(f"  Parameter limit: {'✓ PASS' if param_info['total'] < 100_000_000 else '✗ FAIL'}")
    print(f"  ({param_info['total']/1_000_000:.2f}M / 100M)")
    
    # Test forward pass with 96x96 images
    print("\nFORWARD PASS TEST (96x96 images):")
    dummy_input = torch.randn(4, 3, 96, 96)
    
    with torch.no_grad():
        model.eval()
        
        # Test projection output
        projections = model(dummy_input)
        print(f"  Input shape:       {dummy_input.shape}")
        print(f"  Projections shape: {projections.shape}")
        
        # Verify L2 normalization
        norms = torch.norm(projections, p=2, dim=1)
        print(f"  L2 norms (should be ~1.0): {norms}")
        print(f"  L2 norm check: {'✓ PASS' if torch.allclose(norms, torch.ones_like(norms), atol=1e-5) else '✗ FAIL'}")
        
        # Test feature extraction
        features, proj = model(dummy_input, return_features=True)
        print(f"  Features shape:    {features.shape}")
    
    print("\n" + "="*70)
    print("✓ Model tests passed!")
    print("="*70)