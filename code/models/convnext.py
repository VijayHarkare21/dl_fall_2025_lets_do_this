"""
ConvNeXt-V2 Implementation for Self-Supervised Learning
========================================================

This module implements the ConvNeXt-V2 architecture, a modernized ConvNet that
incorporates design principles from Vision Transformers while retaining the
inductive biases of convolutions (translation equivariance, locality).

ConvNeXt-V2 Key Features:
1. Global Response Normalization (GRN) - The main innovation in V2
2. Depthwise separable convolutions for efficiency
3. LayerNorm instead of BatchNorm (channel-last format)
4. GELU activation
5. Hierarchical 4-stage design with increasing channels

Architecture Overview:
- Stem: 4x4 conv with stride 4 (patchify, similar to ViT)
- Stage 1-4: ConvNeXt blocks with downsampling between stages
- Head: Global average pooling -> LayerNorm -> features

CFM Integration:
- Modulation is applied after LayerNorm in each ConvNeXt block
- FiLM-style: gamma * features + beta

Differences from ViT for SSL:
- No CLS token or patch tokens - outputs single global feature vector
- iBOT (masked patch prediction) is not directly applicable
- Works with DINO loss and KoLeo regularizer

References:
- ConvNeXt: "A ConvNet for the 2020s" (Liu et al., 2022)
- ConvNeXt-V2: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (Woo et al., 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from functools import partial


class GRN(nn.Module):
    """
    Global Response Normalization (GRN) Layer.
    
    GRN is the key innovation in ConvNeXt-V2. It performs global feature
    normalization that helps with feature diversity and prevents collapse
    in self-supervised learning. This is particularly useful for masked
    autoencoder pretraining but also benefits contrastive/distillation methods.
    
    The operation is:
        X = X * norm(X) / (mean(norm(X)) + eps) * gamma + beta
    
    Where norm is the L2 norm across spatial dimensions.
    
    This is applied in channel-last format: (B, H, W, C)
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Number of channels (feature dimension)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        # Learnable scale and bias parameters
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Global Response Normalization.
        
        Args:
            x: Input tensor of shape (B, H, W, C) in channel-last format
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute L2 norm across spatial dimensions (H, W)
        # Keep channel dimension for broadcasting
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        
        # Compute mean norm across channels
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)  # (B, 1, 1, C)
        
        # Apply normalization with learnable parameters
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm2d(nn.Module):
    """
    LayerNorm for 2D feature maps in channel-first format.
    
    Standard LayerNorm expects (B, *, C) but conv layers output (B, C, H, W).
    This wrapper handles the permutation automatically.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Normalized tensor of shape (B, C, H, W)
        """
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-V2 Block.
    
    The block structure is:
        Input (B, C, H, W)
          |
        Depthwise Conv 7x7
          |
        Permute to (B, H, W, C)
          |
        LayerNorm
          |
        [CFM Modulation Point]
          |
        Pointwise Conv (expand by 4x)
          |
        GELU
          |
        GRN (Global Response Normalization)
          |
        Pointwise Conv (project back)
          |
        Permute to (B, C, H, W)
          |
        + Residual (with optional DropPath)
          |
        Output (B, C, H, W)
    
    The CFM modulation is applied after LayerNorm, which allows the
    modulation network to adjust features based on input characteristics
    before the main transformation.
    """
    
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            dim: Number of input/output channels
            drop_path: Stochastic depth drop probability
            mlp_ratio: Expansion ratio for the MLP (typically 4x)
        """
        super().__init__()
        
        # Depthwise convolution (7x7 kernel, groups=dim for depthwise)
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        
        # LayerNorm in channel-last format
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise convolutions (implemented as Linear for efficiency)
        # First expands channels, second projects back
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
        
        # Stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        modulation: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional CFM modulation.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            modulation: Optional dict with 'gamma' and 'beta' tensors
                       Each of shape (B, C) for FiLM modulation
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        residual = x
        
        # Depthwise conv
        x = self.dwconv(x)
        
        # Permute to channel-last for LayerNorm and Linear layers
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        
        # LayerNorm
        x = self.norm(x)
        
        # Apply CFM modulation after LayerNorm (FiLM: gamma * x + beta)
        if modulation is not None:
            gamma = modulation.get('gamma')
            beta = modulation.get('beta')
            if gamma is not None:
                # gamma is (B, C), need to broadcast to (B, H, W, C)
                x = x * gamma.unsqueeze(1).unsqueeze(1)
            if beta is not None:
                x = x + beta.unsqueeze(1).unsqueeze(1)
        
        # MLP with GRN
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        
        # Permute back to channel-first
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Residual connection with drop path
        x = residual + self.drop_path(x)
        
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (DropPath) regularization.
    
    Randomly drops entire residual branches during training, which acts
    as a form of regularization similar to dropout but at the block level.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Create random tensor for batch dimension
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # Binarize
        
        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtStage(nn.Module):
    """
    A stage in ConvNeXt consisting of an optional downsampling layer
    followed by a sequence of ConvNeXt blocks.
    
    ConvNeXt-V2 has 4 stages with the following structure:
    - Stage 1: No downsampling (stem already reduces resolution)
    - Stage 2-4: 2x2 conv with stride 2 for downsampling, then blocks
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        drop_path_rates: List[float],
        downsample: bool = True,
    ):
        """
        Args:
            in_dim: Input channel dimension
            out_dim: Output channel dimension
            depth: Number of ConvNeXt blocks in this stage
            drop_path_rates: List of drop path rates for each block
            downsample: Whether to apply downsampling at the start
        """
        super().__init__()
        
        # Downsampling layer (applied between stages)
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            )
        else:
            # For first stage, just project channels if different
            if in_dim != out_dim:
                self.downsample = nn.Sequential(
                    LayerNorm2d(in_dim),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1),
                )
            else:
                self.downsample = nn.Identity()
        
        # ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_dim,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])
    
    def forward(
        self, 
        x: torch.Tensor, 
        modulations: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Forward pass through the stage.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            modulations: Optional list of modulation dicts, one per block
        
        Returns:
            Output tensor
        """
        x = self.downsample(x)
        
        for i, block in enumerate(self.blocks):
            mod = modulations[i] if modulations is not None else None
            x = block(x, modulation=mod)
        
        return x


class ConvNeXtV2(nn.Module):
    """
    ConvNeXt-V2 Model for Self-Supervised Learning.
    
    This implementation supports:
    - Multiple model sizes (tiny, small, base, large)
    - CFM (Conditional Feature Modulation) integration
    - Optional return of spatial features for advanced SSL methods
    
    Model Variants (for 96x96 input):
    - Tiny:  depths=[3,3,9,3],   dims=[96,192,384,768]    ~28M params
    - Small: depths=[3,3,27,3],  dims=[96,192,384,768]    ~50M params
    - Base:  depths=[3,3,27,3],  dims=[128,256,512,1024]  ~89M params
    
    Output Modes:
    - Default: Returns global feature vector (B, embed_dim)
    - return_spatial=True: Also returns spatial features before pooling
    """
    
    # Model configurations
    CONFIGS = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'dims': [96, 192, 384, 768],
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'dims': [96, 192, 384, 768],
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'dims': [128, 256, 512, 1024],
        },
    }
    
    def __init__(
        self,
        img_size: int = 96,
        in_chans: int = 3,
        num_classes: int = 0,  # 0 = no classification head (for SSL)
        depths: List[int] = [3, 3, 27, 3],
        dims: List[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
    ):
        """
        Args:
            img_size: Input image size (assumes square images)
            in_chans: Number of input channels (3 for RGB)
            num_classes: Number of classes for classification head
                        Set to 0 for SSL (no head, just features)
            depths: Number of blocks in each of the 4 stages
            dims: Channel dimensions for each of the 4 stages
            drop_path_rate: Maximum drop path rate (linearly increases)
            head_init_scale: Scale for classification head init (if used)
        """
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.num_stages = len(depths)
        self.embed_dim = dims[-1]  # Final feature dimension
        
        # Store number of blocks per stage for CFM modulation indexing
        self.blocks_per_stage = depths
        self.total_blocks = sum(depths)
        
        # Stem: Patchify with 4x4 conv, stride 4 (like ViT patch embedding)
        # For 96x96 input, this gives 24x24 feature map
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        
        # Compute drop path rates for each block (linearly increasing)
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        # Build stages
        self.stages = nn.ModuleList()
        cur_block = 0
        
        for i in range(self.num_stages):
            stage = ConvNeXtStage(
                in_dim=dims[i - 1] if i > 0 else dims[0],
                out_dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur_block:cur_block + depths[i]],
                downsample=(i > 0),  # Downsample for stages 2, 3, 4
            )
            self.stages.append(stage)
            cur_block += depths[i]
        
        # Final normalization
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        # Global average pooling (applied in forward)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head (optional, not used for SSL)
        if num_classes > 0:
            self.head = nn.Linear(dims[-1], num_classes)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
        else:
            self.head = nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _distribute_modulations(
        self, 
        modulations: Optional[List[Dict[str, torch.Tensor]]]
    ) -> List[Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Distribute flat list of modulations to each stage.
        
        CFM produces one modulation dict per block (total_blocks total).
        This method splits them into sublists for each stage.
        
        Args:
            modulations: List of modulation dicts, one per block
        
        Returns:
            List of lists, one sublist per stage
        """
        if modulations is None:
            return [None] * self.num_stages
        
        stage_modulations = []
        idx = 0
        for depth in self.depths:
            stage_modulations.append(modulations[idx:idx + depth])
            idx += depth
        
        return stage_modulations
    
    def forward_features(
        self,
        x: torch.Tensor,
        modulations: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from input images.
        
        Args:
            x: Input images of shape (B, C, H, W)
            modulations: Optional list of modulation dicts from CFM,
                        one per block (total: sum of depths)
        
        Returns:
            Tuple of:
            - features: Global feature vector (B, embed_dim)
            - spatial_features: Feature map before pooling (B, embed_dim, H', W')
        """
        # Stem
        x = self.stem(x)
        
        # Distribute modulations to stages
        stage_mods = self._distribute_modulations(modulations)
        
        # Process through stages
        for i, stage in enumerate(self.stages):
            x = stage(x, modulations=stage_mods[i])
        
        # Store spatial features before pooling
        spatial_features = x  # (B, embed_dim, H', W')
        
        # Global average pooling
        x = self.pool(x)  # (B, embed_dim, 1, 1)
        x = x.flatten(1)  # (B, embed_dim)
        
        # Final layer norm
        x = self.norm(x)
        
        return x, spatial_features
    
    def forward(
        self,
        x: torch.Tensor,
        modulations: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_all_tokens: bool = False,
        return_spatial: bool = False,
        **kwargs,  # Accept extra args for compatibility with ViT interface
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images of shape (B, C, H, W)
            modulations: Optional list of modulation dicts from CFM
            return_all_tokens: For ViT compatibility. If True with ConvNeXt,
                              returns (global_features, flattened_spatial)
                              to simulate (cls_token, patch_tokens)
            return_spatial: If True, return spatial features before pooling
        
        Returns:
            If return_all_tokens=False and return_spatial=False:
                Global feature vector (B, embed_dim)
            If return_all_tokens=True or return_spatial=True:
                Tuple of (global_features, spatial_features)
                - global_features: (B, embed_dim)
                - spatial_features: (B, num_spatial, embed_dim) if return_all_tokens
                                   (B, embed_dim, H', W') if return_spatial only
        
        Note:
            return_all_tokens is provided for API compatibility with ViT.
            When used with DINOv2, this allows ConvNeXt to work with the
            same trainer code, though iBOT should be disabled since ConvNeXt
            doesn't have true patch tokens.
        """
        features, spatial = self.forward_features(x, modulations)
        
        if return_all_tokens:
            # Flatten spatial features to simulate patch tokens
            # spatial: (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = spatial.shape
            spatial_flat = spatial.flatten(2).transpose(1, 2)  # (B, H*W, C)
            return features, spatial_flat
        
        if return_spatial:
            return features, spatial
        
        if self.num_classes > 0:
            return self.head(features)
        
        return features
    
    def get_num_layers(self) -> int:
        """Return total number of blocks (for CFM)."""
        return self.total_blocks
    
    @property
    def num_blocks(self) -> int:
        """Alias for total number of blocks."""
        return self.total_blocks
        
    @property
    def block_dims(self):
        """
        Return the channel dimension for each block.
        
        This is needed for CFM compatibility since ConvNeXt has different
        channel dimensions at each stage (unlike ViT which has uniform embed_dim).
        
        Returns:
            List[int]: Channel dimension for each block
        """
        dims_list = []
        for stage_idx in range(self.num_stages):
            stage_dim = self.dims[stage_idx]
            num_blocks = self.depths[stage_idx]
            dims_list.extend([stage_dim] * num_blocks)
        return dims_list


def convnext_tiny(img_size: int = 96, **kwargs) -> ConvNeXtV2:
    """
    ConvNeXt-V2 Tiny model.
    
    Approximately 28M parameters.
    For 96x96 input: 24x24 -> 12x12 -> 6x6 -> 3x3 feature maps
    """
    config = ConvNeXtV2.CONFIGS['tiny']
    return ConvNeXtV2(
        img_size=img_size,
        depths=config['depths'],
        dims=config['dims'],
        **kwargs
    )


def convnext_small(img_size: int = 96, **kwargs) -> ConvNeXtV2:
    """
    ConvNeXt-V2 Small model.
    
    Approximately 50M parameters.
    Same architecture as tiny but with more blocks in stage 3.
    """
    config = ConvNeXtV2.CONFIGS['small']
    return ConvNeXtV2(
        img_size=img_size,
        depths=config['depths'],
        dims=config['dims'],
        **kwargs
    )


def convnext_base(img_size: int = 96, **kwargs) -> ConvNeXtV2:
    """
    ConvNeXt-V2 Base model.
    
    Approximately 89M parameters.
    Wider channels than small variant.
    """
    config = ConvNeXtV2.CONFIGS['base']
    return ConvNeXtV2(
        img_size=img_size,
        depths=config['depths'],
        dims=config['dims'],
        **kwargs
    )


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ConvNeXt-V2 Model Test")
    print("=" * 60)
    
    # Test configuration
    batch_size = 4
    img_size = 96
    in_chans = 3
    
    # Test each model variant
    for variant_name, variant_fn in [
        ('tiny', convnext_tiny),
        ('small', convnext_small),
    ]:
        print(f"\n--- Testing ConvNeXt-V2 {variant_name.capitalize()} ---")
        
        model = variant_fn(img_size=img_size)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Embed dimension: {model.embed_dim}")
        print(f"Total blocks: {model.total_blocks}")
        print(f"Blocks per stage: {model.blocks_per_stage}")
        
        # Test forward pass without modulation
        print("\nTesting forward pass (no modulation)...")
        x = torch.randn(batch_size, in_chans, img_size, img_size)
        
        # Standard output
        out = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        assert out.shape == (batch_size, model.embed_dim), "Output shape mismatch"
        
        # With return_all_tokens (ViT compatibility mode)
        print("\nTesting return_all_tokens mode (ViT compatibility)...")
        features, spatial = model(x, return_all_tokens=True)
        print(f"  Global features shape: {features.shape}")
        print(f"  Spatial features shape: {spatial.shape}")
        
        # Test with modulation
        print("\nTesting forward pass (with modulation)...")
        modulations = []
        for stage_idx, depth in enumerate(model.depths):
            dim = model.dims[stage_idx]
            for _ in range(depth):
                modulations.append({
                    'gamma': torch.ones(batch_size, dim),
                    'beta': torch.zeros(batch_size, dim),
                })
        
        out_mod = model(x, modulations=modulations)
        print(f"  Output shape with modulation: {out_mod.shape}")
        
        # Verify modulation doesn't change output with identity modulation
        diff = (out - out_mod).abs().max()
        print(f"  Max diff with identity modulation: {diff.item():.6f}")
        
        # Test different input sizes (for multi-crop training)
        print("\nTesting different input sizes...")
        for test_size in [48, 96]:
            x_test = torch.randn(batch_size, in_chans, test_size, test_size)
            out_test = model(x_test)
            print(f"  Input {test_size}x{test_size} -> Output shape: {out_test.shape}")
    
    # Test gradient flow
    print("\n--- Testing Gradient Flow ---")
    model = convnext_tiny(img_size=96)
    x = torch.randn(batch_size, in_chans, img_size, img_size, requires_grad=True)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradient is None"
    print("Gradient flows correctly through the model")
    
    # Check gradient norms for each stage
    print("\nGradient norms by component:")
    print(f"  Stem: {sum(p.grad.norm().item() for p in model.stem.parameters() if p.grad is not None):.4f}")
    for i, stage in enumerate(model.stages):
        grad_norm = sum(p.grad.norm().item() for p in stage.parameters() if p.grad is not None)
        print(f"  Stage {i+1}: {grad_norm:.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed")
    print("=" * 60)