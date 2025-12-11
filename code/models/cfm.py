"""
Conditional Feature Modulation (CFM) Network
=============================================

CFM is a lightweight auxiliary network that learns to modulate the main encoder's
behavior based on input characteristics. This addresses the challenge of producing
"universal representations" that work across diverse test distributions (CUB-200,
SUN-397, miniImageNet).

The core insight is that different visual domains may require different feature
emphases:
- Fine-grained bird recognition (CUB-200): emphasize texture and local patterns
- Scene recognition (SUN-397): emphasize global structure and context
- General objects (miniImageNet): balance of local and global features

Rather than hoping a single set of features works well for all domains, CFM
adaptively adjusts feature processing based on the input.

Architecture:
1. Context Encoder: Lightweight CNN that produces a compact context vector from
   the input image (at reduced resolution for efficiency).

2. Modulation Predictors: Small MLPs (one per transformer block) that take the
   context vector and predict (gamma, beta) FiLM parameters for each layer.

The modulation is applied as: x_modulated = gamma * x_normalized + beta

References:
- FiLM: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018)
- Squeeze-and-Excitation: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextEncoder(nn.Module):
    """
    Lightweight CNN that extracts a global context vector from the input image.
    
    The context encoder operates on a downsampled version of the input (e.g., 48x48)
    to be computationally efficient. It produces a compact context vector that
    captures the overall visual characteristics of the input (color distribution,
    texture patterns, scene type, etc.).
    
    Architecture:
        4 conv blocks with increasing channels [32, 64, 128, 256]
        Each block: Conv -> GroupNorm -> ReLU -> MaxPool
        Global average pooling -> context vector
    
    Note: We use GroupNorm instead of BatchNorm to avoid in-place running statistics
    updates that cause issues with DDP when the module is called multiple times
    in a single forward pass (as in DINOv2 multi-crop training).
    """
    
    def __init__(
        self,
        input_size=48,
        in_channels=3,
        context_dim=256,
        channels=[32, 64, 128, 256],
    ):
        """
        Args:
            input_size: Input image size (will be resized to this)
            in_channels: Number of input channels (3 for RGB)
            context_dim: Output context vector dimension
            channels: Channel progression through conv blocks
        """
        super().__init__()
        
        self.input_size = input_size
        self.context_dim = context_dim
        
        # Build convolutional blocks
        blocks = []
        in_ch = in_channels
        
        for out_ch in channels:
            # Use GroupNorm instead of BatchNorm for DDP compatibility
            # num_groups = min(32, out_ch) ensures valid group count
            num_groups = min(32, out_ch)
            # Ensure out_ch is divisible by num_groups
            while out_ch % num_groups != 0:
                num_groups -= 1
            
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_ch),
                nn.ReLU(inplace=False),  # inplace=False for DDP safety
                nn.MaxPool2d(kernel_size=2, stride=2),
            ))
            in_ch = out_ch
        
        self.conv_blocks = nn.Sequential(*blocks)
        
        # Calculate spatial size after conv blocks
        # Each block halves spatial dimension: input_size -> input_size / (2^num_blocks)
        final_spatial = input_size // (2 ** len(channels))
        
        # Global average pooling + projection to context_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], context_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Extract context vector from input image.
        
        Args:
            x: (B, C, H, W) input images (will be resized if needed)
        
        Returns:
            (B, context_dim) context vector
        """
        # Resize to expected input size if needed
        if x.shape[2] != self.input_size or x.shape[3] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=False)
        
        # Extract features through conv blocks
        x = self.conv_blocks(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.flatten(1)
        
        # Project to context dimension
        x = self.fc(x)
        
        return x


class ModulationPredictor(nn.Module):
    """
    Predicts FiLM modulation parameters (gamma, beta) for a single transformer block.
    
    Takes the context vector and outputs gamma and beta parameters that will be
    used to modulate the normalized features in the corresponding transformer block.
    
    The modulation is applied as: x_mod = gamma * x_norm + beta
    
    Initialization is crucial: gamma is initialized near 1 and beta near 0 so that
    the initial modulation is close to identity, allowing gradual learning.
    """
    
    def __init__(
        self,
        context_dim=256,
        feature_dim=384,
        hidden_dim=128,
    ):
        """
        Args:
            context_dim: Input context vector dimension
            feature_dim: Output dimension (matches transformer embed_dim)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Small MLP: context -> hidden -> (gamma, beta)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Separate heads for gamma and beta (allows different initializations)
        self.gamma_head = nn.Linear(hidden_dim, feature_dim)
        self.beta_head = nn.Linear(hidden_dim, feature_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights.
        
        Critical: Initialize gamma output near 1 and beta output near 0
        so initial modulation is close to identity.
        """
        # MLP initialization
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        
        # Gamma head: initialize to output ~1 (identity scaling)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        
        # Beta head: initialize to output ~0 (no shift)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)
    
    def forward(self, context):
        """
        Predict modulation parameters from context.
        
        Args:
            context: (B, context_dim) context vector
        
        Returns:
            dict with 'gamma': (B, feature_dim) and 'beta': (B, feature_dim)
        """
        h = self.mlp(context)
        gamma = self.gamma_head(h)
        beta = self.beta_head(h)
        
        return {'gamma': gamma, 'beta': beta}


class CFMNetwork(nn.Module):
    """
    Complete Conditional Feature Modulation Network.
    
    Combines the ContextEncoder and multiple ModulationPredictors (one per
    transformer block) into a single module that can be used alongside
    the main backbone.
    
    Usage:
        cfm = CFMNetwork(num_blocks=12, feature_dim=384)
        backbone = vit_small()
        
        # Get modulations
        modulations = cfm(images)  # List of dicts, one per block
        
        # Apply modulations in backbone forward
        features = backbone(images, modulations=modulations)
    """
    
    def __init__(
        self,
        num_blocks=12,
        feature_dim=384,
        feature_dims=None,
        context_dim=256,
        input_size=48,
        hidden_dim=128,
    ):
        """
        Args:
            num_blocks: Number of transformer/conv blocks to modulate
            feature_dim: Single feature dimension for all blocks (ViT style)
            feature_dims: List of feature dimensions per block (ConvNeXt style)
                         If provided, overrides feature_dim and num_blocks
            context_dim: Context vector dimension
            input_size: Input size for context encoder (typically half of backbone input)
            hidden_dim: Hidden dimension for modulation predictors
        """
        super().__init__()
        
        # Handle per-block dimensions (ConvNeXt) vs uniform dimension (ViT)
        if feature_dims is not None:
            # ConvNeXt style: different dimensions per block
            self.feature_dims = feature_dims
            self.num_blocks = len(feature_dims)
            self.feature_dim = feature_dims[-1]  # Store last dim for reference
        else:
            # ViT style: same dimension for all blocks
            self.feature_dims = [feature_dim] * num_blocks
            self.num_blocks = num_blocks
            self.feature_dim = feature_dim
        
        self.context_dim = context_dim
        
        # Context encoder (shared across all blocks)
        self.context_encoder = ContextEncoder(
            input_size=input_size,
            context_dim=context_dim,
        )
        
        # One modulation predictor per block, with appropriate output dimension
        self.modulation_predictors = nn.ModuleList([
            ModulationPredictor(
                context_dim=context_dim,
                feature_dim=dim,
                hidden_dim=hidden_dim,
            )
            for dim in self.feature_dims
        ])
    
    def forward(self, x):
        """
        Compute modulation parameters for all transformer blocks.
        
        Args:
            x: (B, C, H, W) input images
        
        Returns:
            List of dicts, each containing 'gamma' and 'beta' tensors
            of shape (B, feature_dim), one dict per transformer block.
        """
        # Extract context vector
        context = self.context_encoder(x)
        
        # Predict modulations for each block
        modulations = [predictor(context) for predictor in self.modulation_predictors]
        
        return modulations
    
    def forward_with_context(self, x):
        """
        Compute modulations and also return the context vector.
        
        Useful for debugging/visualization or if context is needed elsewhere.
        
        Args:
            x: (B, C, H, W) input images
        
        Returns:
            modulations: List of dicts with 'gamma' and 'beta'
            context: (B, context_dim) context vector
        """
        context = self.context_encoder(x)
        modulations = [predictor(context) for predictor in self.modulation_predictors]
        return modulations, context
    
    def get_num_params(self, trainable_only=True):
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class CFMWrapper(nn.Module):
    """
    Wrapper that combines a backbone with CFM for easy end-to-end usage.
    
    This wrapper handles the coordination between the CFM network and
    the backbone, making it easy to switch between CFM-enabled and
    CFM-disabled modes.
    
    Usage:
        backbone = vit_small()
        wrapped = CFMWrapper(backbone, use_cfm=True)
        
        # Forward with CFM
        features = wrapped(images)  # CFM is applied internally
        
        # Can disable CFM at any time
        wrapped.use_cfm = False
        features = wrapped(images)  # No CFM applied
    """
    
    def __init__(
        self,
        backbone,
        use_cfm=True,
        cfm_context_dim=256,
        cfm_input_size=48,
        cfm_hidden_dim=128,
    ):
        """
        Args:
            backbone: ViT backbone model
            use_cfm: Whether to use CFM (can be toggled)
            cfm_context_dim: Context dimension for CFM
            cfm_input_size: Input size for CFM context encoder
            cfm_hidden_dim: Hidden dimension for modulation predictors
        """
        super().__init__()
        
        self.backbone = backbone
        self.use_cfm = use_cfm
        
        # Create CFM network if enabled
        if use_cfm:
            # Get number of blocks (ViT uses 'depth', ConvNeXt uses 'total_blocks')
            if hasattr(backbone, 'depth'):
                num_blocks = backbone.depth
            elif hasattr(backbone, 'total_blocks'):
                num_blocks = backbone.total_blocks
            else:
                num_blocks = backbone.num_blocks
            
            self.cfm = CFMNetwork(
                num_blocks=num_blocks,
                feature_dim=backbone.embed_dim,
                context_dim=cfm_context_dim,
                input_size=cfm_input_size,
                hidden_dim=cfm_hidden_dim,
            )
        else:
            self.cfm = None
    
    def forward(self, x, return_all_tokens=False):
        """
        Forward pass with optional CFM.
        
        Args:
            x: (B, C, H, W) input images
            return_all_tokens: If True, return all tokens; else just CLS
        
        Returns:
            Backbone output (with CFM modulation if enabled)
        """
        modulations = None
        if self.use_cfm and self.cfm is not None:
            modulations = self.cfm(x)
        
        return self.backbone(x, modulations=modulations, return_all_tokens=return_all_tokens)
    
    def get_num_params(self, trainable_only=True):
        """Count total parameters (backbone + CFM)."""
        total = self.backbone.get_num_params(trainable_only)
        if self.cfm is not None:
            total += self.cfm.get_num_params(trainable_only)
        return total


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CFM Network Implementation Test")
    print("=" * 60)
    
    # Test dimensions
    batch_size = 4
    img_size = 96
    embed_dim = 384
    num_blocks = 12
    
    # Create dummy input
    images = torch.randn(batch_size, 3, img_size, img_size)
    
    # Test ContextEncoder
    print("\n--- ContextEncoder ---")
    context_enc = ContextEncoder(input_size=48, context_dim=256)
    context = context_enc(images)
    print(f"Input shape: {images.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Parameters: {sum(p.numel() for p in context_enc.parameters()):,}")
    
    # Test ModulationPredictor
    print("\n--- ModulationPredictor ---")
    mod_pred = ModulationPredictor(context_dim=256, feature_dim=embed_dim)
    modulation = mod_pred(context)
    print(f"Context shape: {context.shape}")
    print(f"Gamma shape: {modulation['gamma'].shape}")
    print(f"Beta shape: {modulation['beta'].shape}")
    print(f"Gamma mean: {modulation['gamma'].mean():.4f} (should be ~1.0)")
    print(f"Beta mean: {modulation['beta'].mean():.4f} (should be ~0.0)")
    print(f"Parameters: {sum(p.numel() for p in mod_pred.parameters()):,}")
    
    # Test full CFMNetwork
    print("\n--- CFMNetwork ---")
    cfm = CFMNetwork(
        num_blocks=num_blocks,
        feature_dim=embed_dim,
        context_dim=256,
        input_size=48,
    )
    modulations = cfm(images)
    print(f"Input shape: {images.shape}")
    print(f"Number of modulation dicts: {len(modulations)}")
    print(f"Each modulation gamma shape: {modulations[0]['gamma'].shape}")
    print(f"Total CFM parameters: {cfm.get_num_params():,}")
    
    # Verify identity initialization
    print("\n--- Identity Initialization Check ---")
    gamma_means = [m['gamma'].mean().item() for m in modulations]
    beta_means = [m['beta'].mean().item() for m in modulations]
    print(f"Gamma means across blocks: min={min(gamma_means):.4f}, max={max(gamma_means):.4f}")
    print(f"Beta means across blocks: min={min(beta_means):.4f}, max={max(beta_means):.4f}")
    
    # Test with context return
    modulations2, context2 = cfm.forward_with_context(images)
    print(f"\nWith context return:")
    print(f"  Modulations: {len(modulations2)} dicts")
    print(f"  Context: {context2.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)