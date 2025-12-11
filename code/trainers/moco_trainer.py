"""
MoCo v3 Trainer
===============

This module implements the MoCo v3 self-supervised learning method with full
support for CFM (Conditional Feature Modulation).

MoCo v3 Key Components:
1. Query-Key Framework: Two views of same image should have similar representations
2. Momentum Encoder: Key encoder weights are EMA of query encoder
3. Projection + Prediction: Query branch has additional predictor head (asymmetry)
4. InfoNCE Loss: Contrastive loss treating same-image pairs as positives

Key Differences from DINO:
- Uses two views (not multi-crop)
- Contrastive loss (not distillation)
- No centering mechanism
- Predictor head on query side creates asymmetry

CFM Integration:
- CFM modulations are applied to both query and key encoders
- Same input -> same modulations for consistency
- CFM learns to adapt features based on input characteristics

References:
- MoCo v3: "An Empirical Study of Training Self-Supervised Vision Transformers"
- MoCo v2: "Improved Baselines with Momentum Contrastive Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.heads import MoCoV3Head, MoCoV3Predictor
from models.cfm import CFMNetwork
from trainers.base_trainer import BaseTrainer, EMAModel, cosine_scheduler
from utils.distributed import is_distributed, get_world_size, all_gather_tensors


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
    
    The loss treats representations of different augmentations of the same image
    as positive pairs, and representations of different images as negative pairs.
    
    For a batch of N images with 2 views each:
    - Each image has 1 positive pair (its other view)
    - Each image has 2*(N-1) negative pairs (other images' views)
    
    In distributed training, negatives are gathered from all GPUs for larger
    effective batch size.
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        Args:
            temperature: Softmax temperature (lower = sharper distribution)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query representations (B, D), L2-normalized
            key: Key representations (B, D), L2-normalized
        
        Returns:
            Scalar loss value
        """
        batch_size = query.shape[0]
        
        # Gather keys from all GPUs for more negatives
        if is_distributed():
            # All-gather keys (no gradient through gathered keys)
            key_all = all_gather_tensors(key.detach())
            
            # For query, we need gradients only for the local queries
            # Get the start index for this GPU's samples
            rank = torch.distributed.get_rank()
            start_idx = rank * batch_size
        else:
            key_all = key.detach()
            start_idx = 0
        
        # Compute similarity matrix: (B, N) where N = total keys across all GPUs
        # query: (B, D), key_all: (N, D) -> similarity: (B, N)
        similarity = torch.mm(query, key_all.t()) / self.temperature
        
        # Labels: each query's positive is at position (start_idx + i) in key_all
        labels = torch.arange(batch_size, device=query.device) + start_idx
        
        # Cross-entropy loss (softmax over all keys)
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class MoCoV3Trainer(BaseTrainer):
    """
    MoCo v3 Trainer with CFM support.
    
    Implements the full MoCo v3 training pipeline:
    - Query encoder: backbone + projection head + prediction head
    - Key encoder: momentum-updated backbone + projection head (no predictor)
    - InfoNCE contrastive loss
    - Optional CFM for adaptive feature modulation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MoCo v3 trainer.
        
        Config should include:
        - All base trainer config options
        - aug_type: Should be 'twoview' for MoCo v3
        """
        # Force two-view augmentation for MoCo v3
        config['aug_type'] = 'twoview'
        config['n_global_crops'] = 2
        config['n_local_crops'] = 0
        
        super().__init__(config)
    
    def _build_model(self):
        """Build MoCo v3-specific model components."""
        # Head configuration
        head_config = self.config.get('head', {})
        # print(f"DEBUG head_config: {head_config}")
        hidden_dim = head_config.get('hidden_dim', 1024)
        # out_dim = head_config.get('out_dim', 256)
        out_dim = 256
        pred_hidden_dim = head_config.get('pred_hidden_dim', 1024)
        # print(f"DEBUG: hidden_dim={hidden_dim}, out_dim={out_dim}, pred_hidden_dim={pred_hidden_dim}")
        
        
        # Build query encoder components
        self.query_head = MoCoV3Head(
            in_dim=self.embed_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        ).to(self.device)
        
        self.predictor = MoCoV3Predictor(
            in_dim=out_dim,
            hidden_dim=pred_hidden_dim,
            out_dim=out_dim,
        ).to(self.device)
        
        # Build key encoder (momentum-updated)
        self.key_backbone = EMAModel(
            self.backbone,
            momentum=self.config.get('ema_momentum', 0.99),
            device=self.device,
        )
        self.key_head = EMAModel(
            self.query_head,
            momentum=self.config.get('ema_momentum', 0.99),
            device=self.device,
        )
        
        # Build key CFM if using CFM
        if self.cfm is not None:
            self.key_cfm = EMAModel(
                self.cfm,
                momentum=self.config.get('ema_momentum', 0.99),
                device=self.device,
            )
        else:
            self.key_cfm = None
        
        # Build loss
        self.criterion = InfoNCELoss(
            temperature=self.config.get('temperature', 0.2),
        )
        
        # Wrap query components in DDP
        self.backbone = self._wrap_ddp(self.backbone)
        self.query_head = self._wrap_ddp(self.query_head)
        self.predictor = self._wrap_ddp(self.predictor)
        if self.cfm is not None:
            self.cfm = self._wrap_ddp(self.cfm)
            
        # Set static graph
        if is_distributed():
            self.backbone._set_static_graph()
            self.query_head._set_static_graph()
            self.predictor._set_static_graph()
            if self.cfm is not None:
                self.cfm._set_static_graph()
        
        # EMA momentum schedule
        self.ema_schedule = cosine_scheduler(
            base_value=self.config.get('ema_momentum', 0.99),
            final_value=1.0,
            epochs=self.config.get('epochs', 100),
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=0,
        )
        
        if self.is_main:
            print(f"[MoCo v3] Built query head: {sum(p.numel() for p in self.query_head.parameters()):,} params")
            print(f"[MoCo v3] Built predictor: {sum(p.numel() for p in self.predictor.parameters()):,} params")
            print(f"[MoCo v3] Output dim: {out_dim}")
            print(f"[MoCo v3] Temperature: {self.config.get('temperature', 0.2)}")
            print(f"[MoCo v3] Backbone type: {self.backbone_type}")
    
    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with separate handling for different components."""
        # Regularized parameters (with weight decay)
        regularized = []
        not_regularized = []
        
        # Backbone parameters
        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'gamma' in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        
        # Query head parameters
        for name, param in self.query_head.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'bn' in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        
        # Predictor parameters
        for name, param in self.predictor.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'bn' in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        
        # CFM parameters
        if self.cfm is not None:
            for name, param in self.cfm.named_parameters():
                if not param.requires_grad:
                    continue
                if 'bias' in name or 'norm' in name:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
        
        return [
            {'params': regularized},
            {'params': not_regularized, 'weight_decay': 0.0},
        ]
    
    def _set_train_mode(self):
        """Set query components to training mode."""
        self.query_head.train()
        self.predictor.train()
        # Key encoder is always in eval mode
        self.key_backbone.ema_model.eval()
        self.key_head.ema_model.eval()
        if self.key_cfm is not None:
            self.key_cfm.ema_model.eval()
    
    def _get_head_param_counts(self) -> tuple:
        """Get parameter counts for MoCo v3 heads (query head + predictor only)."""
        # Get actual modules (handle DDP wrapper)
        head_module = self.query_head.module if hasattr(self.query_head, 'module') else self.query_head
        pred_module = self.predictor.module if hasattr(self.predictor, 'module') else self.predictor
        
        total = sum(p.numel() for p in head_module.parameters())
        total += sum(p.numel() for p in pred_module.parameters())
        
        trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in pred_module.parameters() if p.requires_grad)
        
        return total, trainable
    
    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single MoCo v3 training step.
        
        Args:
            batch: Tuple of (view1, view2) tensors, each (B, C, H, W)
        
        Returns:
            Dictionary containing loss values
        """
        # Unpack views
        view1, view2 = batch
        view1 = view1.to(self.device, non_blocking=True)
        view2 = view2.to(self.device, non_blocking=True)
        
        # Update EMA momentum
        current_momentum = self.ema_schedule[self.global_step] if self.global_step < len(self.ema_schedule) else 1.0
        self.key_backbone.set_momentum(current_momentum)
        self.key_head.set_momentum(current_momentum)
        if self.key_cfm is not None:
            self.key_cfm.set_momentum(current_momentum)
        
        # Get current CFM weight from curriculum
        cfm_weight = self.current_cfm_weight
        
        # ============ Compute query representations ============
        # Query uses view1 through query encoder + predictor
        if self.cfm is not None and cfm_weight > 0:
            q_modulations = self.cfm(view1)
            if cfm_weight < 1.0:
                q_modulations = self._apply_cfm_weight_to_modulations(q_modulations, cfm_weight)
        else:
            q_modulations = None
        
        q_features = self.backbone(view1, modulations=q_modulations)
        q_proj = self.query_head(q_features)
        q_pred = self.predictor(q_proj)  # Additional predictor for asymmetry
        
        # ============ Compute key representations (no gradients) ============
        with torch.no_grad():
            if self.key_cfm is not None and cfm_weight > 0:
                k_modulations = self.key_cfm(view2)
                if cfm_weight < 1.0:
                    k_modulations = self._apply_cfm_weight_to_modulations(k_modulations, cfm_weight)
            else:
                k_modulations = None
            
            k_features = self.key_backbone(view2, modulations=k_modulations)
            k_proj = self.key_head(k_features)
        
        # ============ Compute loss ============
        # Query (with predictor) should match key
        loss = self.criterion(q_pred, k_proj)
        
        # Symmetrize: also compute loss with swapped views
        # Query uses view2, key uses view1
        if self.cfm is not None and cfm_weight > 0:
            q_modulations_2 = self.cfm(view2)
            if cfm_weight < 1.0:
                q_modulations_2 = self._apply_cfm_weight_to_modulations(q_modulations_2, cfm_weight)
        else:
            q_modulations_2 = None
        
        q_features_2 = self.backbone(view2, modulations=q_modulations_2)
        q_proj_2 = self.query_head(q_features_2)
        q_pred_2 = self.predictor(q_proj_2)
        
        with torch.no_grad():
            if self.key_cfm is not None and cfm_weight > 0:
                k_modulations_2 = self.key_cfm(view1)
                if cfm_weight < 1.0:
                    k_modulations_2 = self._apply_cfm_weight_to_modulations(k_modulations_2, cfm_weight)
            else:
                k_modulations_2 = None
            
            k_features_2 = self.key_backbone(view1, modulations=k_modulations_2)
            k_proj_2 = self.key_head(k_features_2)
        
        loss_2 = self.criterion(q_pred_2, k_proj_2)
        
        # Total loss is average of both directions
        total_loss = (loss + loss_2) / 2
        
        # ============ Backward pass ============
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        if max_grad_norm is not None:
            params = []
            if hasattr(self.backbone, 'module'):
                params.extend(self.backbone.module.parameters())
            else:
                params.extend(self.backbone.parameters())
            if hasattr(self.query_head, 'module'):
                params.extend(self.query_head.module.parameters())
            else:
                params.extend(self.query_head.parameters())
            if hasattr(self.predictor, 'module'):
                params.extend(self.predictor.module.parameters())
            else:
                params.extend(self.predictor.parameters())
            if self.cfm is not None and self._should_train_cfm():
                if hasattr(self.cfm, 'module'):
                    params.extend(self.cfm.module.parameters())
                else:
                    params.extend(self.cfm.parameters())
            
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        
        self.optimizer.step()
        
        # ============ Update key encoder with EMA ============
        self.key_backbone.update(self.backbone)
        self.key_head.update(self.query_head)
        if self.cfm is not None and self.key_cfm is not None and cfm_weight > 0:
            self.key_cfm.update(self.cfm)
        
        return {
            'loss': total_loss,
            'loss_view1': loss,
            'loss_view2': loss_2,
            'ema_momentum': current_momentum,
            'cfm_weight': cfm_weight,
        }
    
    def _apply_cfm_weight_to_modulations(
        self, 
        modulations: List[Dict[str, torch.Tensor]], 
        weight: float
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Apply curriculum weight to CFM modulations.
        
        Adjusts gamma toward 1 and beta toward 0 based on weight.
        When weight=0: gamma=1, beta=0 (identity)
        When weight=1: gamma and beta unchanged (full modulation)
        """
        adjusted = []
        for mod in modulations:
            adj_mod = {}
            if 'gamma' in mod:
                adj_mod['gamma'] = weight * mod['gamma'] + (1 - weight) * torch.ones_like(mod['gamma'])
            if 'beta' in mod:
                adj_mod['beta'] = weight * mod['beta']
            adjusted.append(adj_mod)
        return adjusted
    
    def _should_train_cfm(self) -> bool:
        """Check if CFM should receive gradients at the current step."""
        if not self.use_cfm or self.cfm is None:
            return False
        
        if self.cfm_curriculum is None:
            return True
        
        return self.cfm_curriculum.should_train_cfm(self.global_step)
    
    def _save_checkpoint_hook(self) -> Dict[str, Any]:
        """Save MoCo v3-specific state."""
        state = {
            'key_backbone': self.key_backbone.state_dict(),
            'key_head': self.key_head.state_dict(),
        }
        
        # Handle DDP wrapper
        if hasattr(self.query_head, 'module'):
            state['query_head'] = self.query_head.module.state_dict()
        else:
            state['query_head'] = self.query_head.state_dict()
        
        if hasattr(self.predictor, 'module'):
            state['predictor'] = self.predictor.module.state_dict()
        else:
            state['predictor'] = self.predictor.state_dict()
        
        if self.key_cfm is not None:
            state['key_cfm'] = self.key_cfm.state_dict()
        
        return state
    
    def _load_checkpoint_hook(self, checkpoint: Dict[str, Any]):
        """Load MoCo v3-specific state."""
        if 'key_backbone' in checkpoint:
            self.key_backbone.load_state_dict(checkpoint['key_backbone'])
        if 'key_head' in checkpoint:
            self.key_head.load_state_dict(checkpoint['key_head'])
        if 'query_head' in checkpoint:
            if hasattr(self.query_head, 'module'):
                self.query_head.module.load_state_dict(checkpoint['query_head'])
            else:
                self.query_head.load_state_dict(checkpoint['query_head'])
        if 'predictor' in checkpoint:
            if hasattr(self.predictor, 'module'):
                self.predictor.module.load_state_dict(checkpoint['predictor'])
            else:
                self.predictor.load_state_dict(checkpoint['predictor'])
        if self.key_cfm is not None and 'key_cfm' in checkpoint:
            self.key_cfm.load_state_dict(checkpoint['key_cfm'])


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MoCo v3 Trainer Components Test")
    print("=" * 60)
    
    # Test InfoNCELoss
    print("\n--- InfoNCELoss Test ---")
    
    batch_size = 8
    feat_dim = 256
    
    loss_fn = InfoNCELoss(temperature=0.2)
    
    # Simulate normalized outputs
    query = F.normalize(torch.randn(batch_size, feat_dim), dim=1)
    key = F.normalize(torch.randn(batch_size, feat_dim), dim=1)
    
    loss = loss_fn(query, key)
    print(f"Loss value (random): {loss.item():.4f}")
    
    # With matching query and key, loss should be lower
    key_matching = query.detach().clone()
    loss_matching = loss_fn(query, key_matching)
    print(f"Loss value (matching): {loss_matching.item():.4f}")
    assert loss_matching < loss, "Loss should be lower when query matches key"
    print("Loss is lower when query matches key")
    
    # Test that loss is differentiable
    query_raw = torch.randn(batch_size, feat_dim, requires_grad=True)
    query_grad = F.normalize(query_raw, dim=1)  # Non-leaf tensor
    key_grad = F.normalize(torch.randn(batch_size, feat_dim), dim=1)
    loss = loss_fn(query_grad, key_grad)
    loss.backward()
    # Check gradient on the leaf tensor (query_raw), not the normalized one
    assert query_raw.grad is not None, "Query should have gradients"
    print("Loss is differentiable through query")
    
    # Test symmetrization (as done in training)
    print("\n--- Symmetrization Test ---")
    q1 = F.normalize(torch.randn(batch_size, feat_dim), dim=1)
    k1 = F.normalize(torch.randn(batch_size, feat_dim), dim=1)
    
    loss1 = loss_fn(q1, k1)
    loss2 = loss_fn(k1.clone().requires_grad_(True), q1.detach())
    total = (loss1 + loss2) / 2
    print(f"Loss view1: {loss1.item():.4f}")
    print(f"Loss view2: {loss2.item():.4f}")
    print(f"Symmetric loss: {total.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nNote: Full trainer testing requires data. Use scripts/train.py with small data.")