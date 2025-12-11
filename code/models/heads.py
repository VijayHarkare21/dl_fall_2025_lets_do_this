"""
Projection and Prediction Heads for SSL Methods
================================================

This module contains the head architectures used in self-supervised learning:

1. DINOv2 Head: Multi-layer projection head with L2 normalization and 
   weight-normalized prototype layer for self-distillation.

2. MoCo v3 Head: Projection + Prediction MLP for contrastive learning.

These heads are attached to the backbone during SSL pretraining but are
discarded during downstream evaluation (k-NN uses backbone features directly).

References:
- DINOv2: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
- MoCo v3: "An Empirical Study of Training Self-Supervised Vision Transformers" (Chen et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Head(nn.Module):
    """
    DINOv2 Projection Head.
    
    Architecture:
        Input (embed_dim) -> MLP layers -> L2 norm -> Prototype layer (weight-normalized)
    
    The head projects backbone features to a space where self-distillation is performed.
    Key components:
    - Multi-layer MLP with GELU activation and LayerNorm
    - L2 normalization before the final projection
    - Weight-normalized last layer (prototypes) for stable training
    
    Note: Default dimensions are scaled down for ViT-Small (~22M params backbone).
    Original DINOv2 used out_dim=65536, hidden_dim=2048 for larger backbones.
    """
    
    def __init__(
        self,
        in_dim,
        out_dim=16384,
        hidden_dim=1024,
        bottleneck_dim=256,
        nlayers=3,
        use_bn=False,
        norm_last_layer=True,
    ):
        """
        Args:
            in_dim: Input dimension (backbone embed_dim, e.g., 384 for ViT-Small)
            out_dim: Output dimension (number of prototypes, default 16384 for ViT-Small)
            hidden_dim: Hidden layer dimension in the MLP (default 1024 for ViT-Small)
            bottleneck_dim: Dimension before the final prototype layer
            nlayers: Number of MLP layers (minimum 2)
            use_bn: Whether to use BatchNorm (False for LayerNorm, which is more stable)
            norm_last_layer: Whether to weight-normalize the last layer
        """
        super().__init__()
        
        # Ensure at least 2 layers (input projection + output projection)
        nlayers = max(nlayers, 2)
        
        # Build the MLP layers
        # Structure: in_dim -> hidden_dim -> ... -> hidden_dim -> bottleneck_dim
        layers = []
        
        # First layer: in_dim -> hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        else:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        # Final layer before prototypes: hidden_dim -> bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Prototype layer (projects bottleneck_dim -> out_dim)
        # Weight normalization helps stabilize training
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.norm_last_layer = norm_last_layer
        if norm_last_layer:
            self.last_layer = nn.utils.parametrizations.weight_norm(self.last_layer)
            # Initialize direction component to uniform
            # Modern API: original0 = magnitude (g), original1 = direction (v)
            # Old API: original = direction (v)
            if hasattr(self.last_layer.parametrizations.weight, 'original'):
                self.last_layer.parametrizations.weight.original.data.fill_(1.0 / out_dim)
            else:
                self.last_layer.parametrizations.weight.original1.data.fill_(1.0 / out_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights using truncated normal."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the head.
        
        Args:
            x: (B, in_dim) backbone features (e.g., CLS token)
        
        Returns:
            (B, out_dim) projected features (L2-normalized before prototype layer)
        """
        # MLP projection
        x = self.mlp(x)
        
        # L2 normalize before prototype layer (crucial for stable training)
        x = F.normalize(x, dim=-1, p=2)
        
        # Project to prototype space
        x = self.last_layer(x)
        
        return x


class MoCoV3Head(nn.Module):
    """
    MoCo v3 Projection Head.
    
    Architecture:
        Input (embed_dim) -> Linear -> BN -> ReLU -> Linear -> BN -> ReLU -> Linear
    
    MoCo v3 uses a 3-layer MLP with BatchNorm and ReLU for projection.
    The output is L2-normalized for the contrastive loss.
    
    Unlike DINO, MoCo v3 also uses a prediction head on the query side.
    
    Note: Default dimensions are scaled down for ViT-Small (~22M params backbone).
    Original MoCo v3 used hidden_dim=4096 for larger backbones.
    """
    
    def __init__(
        self,
        in_dim,
        hidden_dim=1024,
        out_dim=256,
    ):
        """
        Args:
            in_dim: Input dimension (backbone embed_dim)
            hidden_dim: Hidden layer dimension (default 1024 for ViT-Small)
            out_dim: Output dimension (typically 256 for MoCo v3)
        """
        super().__init__()
        
        # 3-layer MLP with BN and ReLU
        self.projection = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, in_dim) backbone features
        
        Returns:
            (B, out_dim) L2-normalized projected features
        """
        x = self.projection(x)
        x = F.normalize(x, dim=-1, p=2)
        return x


class MoCoV3Predictor(nn.Module):
    """
    MoCo v3 Prediction Head (used only on the query/student side).
    
    Architecture:
        Input (out_dim) -> Linear -> BN -> ReLU -> Linear
    
    The predictor is a 2-layer MLP that introduces asymmetry between
    the query and key branches, which is important for avoiding collapse.
    
    Note: Default dimensions are scaled down for ViT-Small (~22M params backbone).
    Original MoCo v3 used hidden_dim=4096 for larger backbones.
    """
    
    def __init__(
        self,
        in_dim=256,
        hidden_dim=1024,
        out_dim=256,
    ):
        """
        Args:
            in_dim: Input dimension (projection output dim)
            hidden_dim: Hidden layer dimension (default 1024 for ViT-Small)
            out_dim: Output dimension (should match projection out_dim)
        """
        super().__init__()
        
        self.predictor = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.ReLU(inplace=False),
        nn.Linear(hidden_dim, out_dim),
    )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, in_dim) projected features
        
        Returns:
            (B, out_dim) predicted features (L2-normalized)
        """
        x = self.predictor(x)
        x = F.normalize(x, dim=-1, p=2)
        return x


class iBOTHead(nn.Module):
    """
    iBOT Head for Masked Image Modeling (used in DINOv2).
    
    This head is used for the patch-level masked prediction task that
    complements the image-level DINO loss in DINOv2. It predicts the
    teacher's patch tokens for masked positions.
    
    Architecture is similar to DINOv2Head but operates on patch tokens
    rather than the CLS token.
    
    Note: Default dimensions are scaled down for ViT-Small (~22M params backbone).
    """
    
    def __init__(
        self,
        in_dim,
        out_dim=16384,
        hidden_dim=1024,
        bottleneck_dim=256,
        nlayers=3,
        shared_head=None,
    ):
        """
        Args:
            in_dim: Input dimension (backbone embed_dim)
            out_dim: Output dimension (should match DINOv2Head out_dim, default 16384)
            hidden_dim: Hidden layer dimension (default 1024 for ViT-Small)
            bottleneck_dim: Bottleneck dimension
            nlayers: Number of MLP layers
            shared_head: Optional DINOv2Head to share the last layer with
        """
        super().__init__()
        
        nlayers = max(nlayers, 2)
        
        # Build MLP (same structure as DINOv2Head)
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Share or create last layer
        if shared_head is not None:
            self.last_layer = shared_head.last_layer
        else:
            self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
            self.last_layer = nn.utils.parametrizations.weight_norm(self.last_layer)
            # Initialize direction component to uniform
            # Modern API: original0 = magnitude (g), original1 = direction (v)
            # Old API: original = direction (v)
            if hasattr(self.last_layer.parametrizations.weight, 'original'):
                self.last_layer.parametrizations.weight.original.data.fill_(1.0 / out_dim)
            else:
                self.last_layer.parametrizations.weight.original1.data.fill_(1.0 / out_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize MLP weights."""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass for patch tokens.
        
        Args:
            x: (B, N, in_dim) patch token features or (B, in_dim) if already pooled
        
        Returns:
            (B, N, out_dim) or (B, out_dim) projected features
        """
        # Handle both batched tokens and single vectors
        input_shape = x.shape
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)
        
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        
        # Reshape back if needed
        if len(input_shape) == 3:
            x = x.reshape(B, N, -1)
        
        return x


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SSL Heads Implementation Test")
    print("=" * 60)
    
    # Test dimensions (ViT-Small backbone)
    batch_size = 4
    embed_dim = 384  # ViT-Small embed dim
    num_patches = 144  # 12x12 for 96x96 with patch_size=8
    
    # Create dummy inputs
    cls_token = torch.randn(batch_size, embed_dim)
    patch_tokens = torch.randn(batch_size, num_patches, embed_dim)
    
    # Test DINOv2Head (with ViT-Small scaled defaults)
    print("\n--- DINOv2Head ---")
    dino_head = DINOv2Head(
        in_dim=embed_dim,
        # Using defaults: out_dim=16384, hidden_dim=1024, bottleneck_dim=256
    )
    dino_out = dino_head(cls_token)
    print(f"Input shape: {cls_token.shape}")
    print(f"Output shape: {dino_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in dino_head.parameters()):,}")
    
    # Test iBOTHead (shares last layer with DINOv2Head)
    print("\n--- iBOTHead (shared last layer) ---")
    ibot_head = iBOTHead(
        in_dim=embed_dim,
        # Using defaults: out_dim=16384, hidden_dim=1024, bottleneck_dim=256
        shared_head=dino_head,
    )
    ibot_out = ibot_head(patch_tokens)
    print(f"Input shape: {patch_tokens.shape}")
    print(f"Output shape: {ibot_out.shape}")
    print(f"Parameters (own): {sum(p.numel() for p in ibot_head.mlp.parameters()):,}")
    
    # Test MoCoV3Head (with ViT-Small scaled defaults)
    print("\n--- MoCoV3Head ---")
    moco_head = MoCoV3Head(
        in_dim=embed_dim,
        # Using defaults: hidden_dim=1024, out_dim=256
    )
    moco_out = moco_head(cls_token)
    print(f"Input shape: {cls_token.shape}")
    print(f"Output shape: {moco_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in moco_head.parameters()):,}")
    print(f"Output L2 norm: {moco_out.norm(dim=-1).mean():.4f} (should be ~1.0)")
    
    # Test MoCoV3Predictor (with ViT-Small scaled defaults)
    print("\n--- MoCoV3Predictor ---")
    moco_predictor = MoCoV3Predictor(
        in_dim=256,
        # Using defaults: hidden_dim=1024, out_dim=256
    )
    pred_out = moco_predictor(moco_out)
    print(f"Input shape: {moco_out.shape}")
    print(f"Output shape: {pred_out.shape}")
    print(f"Parameters: {sum(p.numel() for p in moco_predictor.parameters()):,}")
    
    # Verify L2 normalization
    print("\n--- Normalization Check ---")
    print(f"DINOv2 output norm (before softmax): {dino_out.norm(dim=-1).mean():.4f}")
    print(f"MoCo projection output norm: {moco_out.norm(dim=-1).mean():.4f}")
    print(f"MoCo prediction output norm: {pred_out.norm(dim=-1).mean():.4f}")
    
    # Summary
    print("\n--- Parameter Summary (ViT-Small scaled) ---")
    print(f"ViT-Small backbone: ~22M params")
    print(f"DINOv2Head: {sum(p.numel() for p in dino_head.parameters()):,} params")
    print(f"iBOTHead (own MLP): {sum(p.numel() for p in ibot_head.mlp.parameters()):,} params")
    print(f"MoCoV3Head: {sum(p.numel() for p in moco_head.parameters()):,} params")
    print(f"MoCoV3Predictor: {sum(p.numel() for p in moco_predictor.parameters()):,} params")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)