"""
Vision Transformer (ViT) Implementation
========================================

A clean, modular implementation of ViT-Small optimized for 96x96 input resolution.
Designed for self-supervised learning experiments including DINOv2 with iBOT.

Architecture (ViT-Small for 96x96):
- Patch size: 8x8 (yields 12x12 = 144 patches)
- Embedding dim: 384
- Heads: 6
- Layers: 12
- MLP ratio: 4
- Approx params: ~22M

Key features for DINOv2:
- Supports variable input sizes (for multi-crop training)
- Positional embedding interpolation for different resolutions
- Returns both CLS token and patch tokens (for iBOT masked prediction)
- Includes optional mask token for masked image modeling

References:
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- "DINO" (Caron et al., 2021)
- "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PatchEmbed(nn.Module):
    """
    Convert image into patch embeddings.
    
    For 96x96 images with patch_size=8:
    - Number of patches: (96/8) * (96/8) = 144
    - Each patch: 8*8*3 = 192 pixels -> projected to embed_dim
    
    Supports variable input sizes as long as dimensions are divisible by patch_size.
    """
    
    def __init__(self, img_size=96, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 144 for 96x96 with patch=8
        
        # Linear projection of flattened patches (implemented as conv2d)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        
        # Allow variable input sizes, just check divisibility
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}x{W}) must be divisible by patch size ({self.patch_size})"
        
        # (B, embed_dim, H/patch, W/patch) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with optional attention dropout.
    """
    
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) where N = num_patches + 1 (for CLS token)
        Returns:
            (B, N, D) attended features
        """
        B, N, D = x.shape
        
        # Compute Q, K, V: (B, N, 3*D) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        
        # Attention: softmax(QK^T / sqrt(d_k)) * V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with LayerNorm, Attention, and MLP.
    
    Includes LayerScale for training stability (important for training ViT from scratch).
    LayerScale: learnable per-channel scaling initialized to small values.
    """
    
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=True, 
        drop=0., 
        attn_drop=0.,
        layer_scale_init=1e-4,  # LayerScale initialization (None to disable)
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads, 
            qkv_bias=qkv_bias, 
            attn_drop=attn_drop, 
            proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim, 
            hidden_features=int(dim * mlp_ratio), 
            drop=drop
        )
        
        # LayerScale parameters (for stable training from scratch)
        self.use_layer_scale = layer_scale_init is not None
        if self.use_layer_scale:
            self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(dim))
    
    def forward(self, x, modulation=None):
        """
        Args:
            x: (B, N, D) input features
            modulation: Optional dict with 'gamma' and 'beta' for CFM
                       Each should be (B, D) or (D,)
        Returns:
            (B, N, D) output features
        """
        # Attention block with residual
        if self.use_layer_scale:
            x = x + self.gamma1 * self.attn(self._maybe_modulate(self.norm1(x), modulation))
        else:
            x = x + self.attn(self._maybe_modulate(self.norm1(x), modulation))
        
        # MLP block with residual
        if self.use_layer_scale:
            x = x + self.gamma2 * self.mlp(self.norm2(x))
        else:
            x = x + self.mlp(self.norm2(x))
        
        return x
    
    def _maybe_modulate(self, x, modulation):
        """Apply CFM modulation if provided: gamma * x + beta"""
        if modulation is None:
            return x
        gamma = modulation.get('gamma', None)
        beta = modulation.get('beta', None)
        
        if gamma is not None:
            # gamma: (B, D) -> (B, 1, D) for broadcasting
            if gamma.dim() == 2:
                gamma = gamma.unsqueeze(1)
            x = x * gamma
        if beta is not None:
            if beta.dim() == 2:
                beta = beta.unsqueeze(1)
            x = x + beta
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for self-supervised learning.
    
    This implementation:
    - Uses CLS token for global representation
    - Includes LayerScale for stable training from scratch
    - Supports optional CFM (Conditional Feature Modulation) per block
    - Returns both CLS token and patch tokens for DINOv2+iBOT
    - Supports variable input sizes with positional embedding interpolation
    - Includes optional mask token for masked image modeling
    """
    
    def __init__(
        self,
        img_size=96,
        patch_size=8,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        layer_scale_init=1e-4,
        norm_layer=None,
    ):
        """
        Args:
            img_size: Input image size (96 for this project)
            patch_size: Patch size (8 recommended for 96x96)
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            qkv_bias: Whether to use bias in QKV projection
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            layer_scale_init: LayerScale init value (None to disable)
            norm_layer: Normalization layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        self.num_patches = self.patch_embed.num_patches
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Mask token for iBOT masked image modeling (learnable)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                layer_scale_init=layer_scale_init,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ViT paper recommendations."""
        # Position embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Apply to all modules
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _interpolate_pos_embed(self, num_patches: int, height: int, width: int) -> torch.Tensor:
        """
        Interpolate positional embeddings for different input sizes.
        
        This is crucial for DINOv2 multi-crop training where local crops
        have different sizes than global crops.
        
        Args:
            num_patches: Number of patches in current input
            height: Height of patch grid
            width: Width of patch grid
        
        Returns:
            Interpolated positional embeddings (1, num_patches + 1, embed_dim)
        """
        # Separate CLS token and patch embeddings
        cls_pos = self.pos_embed[:, :1, :]  # (1, 1, embed_dim)
        patch_pos = self.pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)
        
        # Original grid size
        orig_size = int(self.num_patches ** 0.5)  # e.g., 12 for 144 patches
        
        # If same size, no interpolation needed
        if height == orig_size and width == orig_size:
            return self.pos_embed
        
        # Reshape to 2D grid for interpolation
        dim = patch_pos.shape[-1]
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        
        # Interpolate to new size
        patch_pos = F.interpolate(
            patch_pos,
            size=(height, width),
            mode='bicubic',
            align_corners=False,
        )
        
        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, height * width, dim)
        
        # Concatenate CLS and interpolated patch embeddings
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    def forward_features(self, x, modulations=None, mask=None):
        """
        Extract features from input images.
        
        Args:
            x: (B, C, H, W) input images
            modulations: Optional list of dicts (one per block) for CFM
            mask: Optional (B, num_patches) boolean mask for iBOT
                  True indicates positions to mask
        
        Returns:
            cls_token: (B, embed_dim) global representation
            patch_tokens: (B, num_patches, embed_dim) patch representations
        """
        B, C, H, W = x.shape
        
        # Patch embedding: (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        num_patches = x.shape[1]
        
        # Calculate patch grid dimensions
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        
        # Apply mask if provided (for iBOT)
        if mask is not None:
            # Replace masked positions with mask token
            mask_tokens = self.mask_token.expand(B, num_patches, -1)
            # mask: (B, num_patches) -> (B, num_patches, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * (1 - mask_expanded) + mask_tokens * mask_expanded
        
        # Prepend CLS token: (B, num_patches + 1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings (with interpolation if needed)
        pos_embed = self._interpolate_pos_embed(num_patches, patch_h, patch_w)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            mod = modulations[i] if modulations is not None else None
            x = block(x, modulation=mod)
        
        # Final normalization
        x = self.norm(x)
        
        # Return CLS token and patch tokens separately
        return x[:, 0], x[:, 1:]
    
    def forward(self, x, modulations=None, return_all_tokens=False, mask=None):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input images
            modulations: Optional list of dicts for CFM
            return_all_tokens: If True, return (cls_token, patch_tokens)
                              If False, return only cls_token
            mask: Optional (B, num_patches) boolean mask for iBOT
        
        Returns:
            If return_all_tokens=False: (B, embed_dim) CLS token only
            If return_all_tokens=True: tuple of (cls_token, patch_tokens)
                - cls_token: (B, embed_dim)
                - patch_tokens: (B, num_patches, embed_dim)
        """
        cls_token, patch_tokens = self.forward_features(x, modulations, mask)
        
        if return_all_tokens:
            return cls_token, patch_tokens
        return cls_token
    
    def get_num_params(self, trainable_only=True):
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# =============================================================================
#                         MODEL FACTORY FUNCTIONS
# =============================================================================

def vit_small(img_size=96, patch_size=8, **kwargs):
    """
    ViT-Small configuration optimized for 96x96 images.
    
    Specs:
    - embed_dim: 384
    - depth: 12
    - heads: 6
    - ~22M parameters
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        **kwargs
    )


def vit_tiny(img_size=96, patch_size=8, **kwargs):
    """
    ViT-Tiny for quick testing and debugging.
    
    Specs:
    - embed_dim: 192
    - depth: 12
    - heads: 3
    - ~5.5M parameters
    """
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.,
        **kwargs
    )


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    # Quick test to verify implementation
    print("=" * 60)
    print("ViT Implementation Test")
    print("=" * 60)
    
    # Test ViT-Small
    model = vit_small(img_size=96, patch_size=8)
    print(f"\nViT-Small Configuration:")
    print(f"  Image size: {model.img_size}")
    print(f"  Patch size: {model.patch_size}")
    print(f"  Num patches: {model.num_patches}")
    print(f"  Embed dim: {model.embed_dim}")
    print(f"  Depth: {model.depth}")
    print(f"  Num heads: {model.num_heads}")
    print(f"  Total params: {model.get_num_params():,}")
    
    # Test forward pass with 96x96 (global crop)
    x_global = torch.randn(2, 3, 96, 96)
    
    # CLS only
    cls_out = model(x_global)
    print(f"\nForward pass (96x96, CLS only):")
    print(f"  Input shape: {x_global.shape}")
    print(f"  Output shape: {cls_out.shape}")
    
    # CLS + patch tokens
    cls_out, patch_out = model(x_global, return_all_tokens=True)
    print(f"\nForward pass (96x96, all tokens):")
    print(f"  CLS shape: {cls_out.shape}")
    print(f"  Patch tokens shape: {patch_out.shape}")
    
    # Test forward pass with 48x48 (local crop) - tests pos_embed interpolation
    x_local = torch.randn(2, 3, 48, 48)
    cls_out_local, patch_out_local = model(x_local, return_all_tokens=True)
    print(f"\nForward pass (48x48, all tokens):")
    print(f"  Input shape: {x_local.shape}")
    print(f"  CLS shape: {cls_out_local.shape}")
    print(f"  Patch tokens shape: {patch_out_local.shape}")
    print(f"  Expected patches: {(48//8)**2} = 36")
    
    # Test with mask (for iBOT)
    mask = torch.zeros(2, 144, dtype=torch.bool)
    mask[:, :30] = True  # Mask first 30 patches
    cls_out_masked, patch_out_masked = model(x_global, return_all_tokens=True, mask=mask)
    print(f"\nForward pass (with mask):")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Masked positions: {mask.sum().item()}")
    print(f"  CLS shape: {cls_out_masked.shape}")
    print(f"  Patch tokens shape: {patch_out_masked.shape}")
    
    # Test with CFM modulation
    dummy_modulations = [
        {'gamma': torch.ones(2, 384), 'beta': torch.zeros(2, 384)}
        for _ in range(model.depth)
    ]
    cls_out_cfm = model(x_global, modulations=dummy_modulations)
    print(f"\nForward pass (with CFM modulation):")
    print(f"  Output shape: {cls_out_cfm.shape}")
    
    # Test ViT-Tiny for comparison
    model_tiny = vit_tiny()
    print(f"\nViT-Tiny params: {model_tiny.get_num_params():,}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)