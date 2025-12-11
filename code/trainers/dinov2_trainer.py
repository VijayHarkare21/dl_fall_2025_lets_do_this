"""
DINOv2 Trainer
==============

This module implements the full DINOv2 self-supervised learning method with:
1. DINO loss - CLS token self-distillation
2. iBOT loss - Masked patch token prediction
3. KoLeo regularizer - Uniform feature distribution on hypersphere
4. CFM support - Conditional Feature Modulation

DINOv2 Key Components:
1. Student-Teacher Framework: Student learns to match teacher's output distribution
2. EMA Teacher: Teacher weights are exponential moving average of student
3. Centering: Prevents collapse by centering teacher outputs (separate for CLS and patches)
4. Multi-Crop: Global crops for teacher, all crops for student
5. Temperature Sharpening: Different temperatures for student and teacher
6. Masked Image Modeling (iBOT): Predict teacher's patch tokens at masked positions

Loss Computation:
- DINO: Cross-entropy between student and teacher CLS tokens
- iBOT: Cross-entropy between student and teacher patch tokens at masked positions
- KoLeo: Encourages uniform distribution of features on hypersphere

References:
- DINOv2: "DINOv2: Learning Robust Visual Features without Supervision" (Oquab et al., 2023)
- DINO: "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al., 2021)
- iBOT: "iBOT: Image BERT Pre-Training with Online Tokenizer" (Zhou et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.heads import DINOv2Head, iBOTHead
from models.cfm import CFMNetwork
from trainers.base_trainer import BaseTrainer, EMAModel, cosine_scheduler
from utils.distributed import is_distributed, get_world_size, all_reduce_mean


class DINOLoss(nn.Module):
    """
    DINO loss for CLS token self-distillation.
    
    Computes the cross-entropy between student and teacher CLS token output distributions.
    Uses centering to prevent collapse.
    """
    
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        # Center buffer (running mean of teacher CLS outputs)
        self.register_buffer('center', torch.zeros(1, out_dim))
    
    def forward(
        self,
        student_cls_outputs: List[torch.Tensor],
        teacher_cls_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute DINO loss on CLS tokens.
        
        Args:
            student_cls_outputs: List of student CLS outputs for all crops
            teacher_cls_outputs: List of teacher CLS outputs for global crops only
        
        Returns:
            Scalar loss value
        """
        n_global = len(teacher_cls_outputs)
        
        # Apply temperature and softmax
        student_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_cls_outputs]
        teacher_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() 
                        for t in teacher_cls_outputs]
        
        # Compute cross-entropy loss
        total_loss = 0
        n_loss_terms = 0
        
        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_prob in enumerate(student_probs):
                # Skip if same crop
                if s_idx == t_idx:
                    continue
                
                loss = -torch.sum(t_prob * s_prob, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
        
        total_loss /= n_loss_terms
        
        # Update center
        self._update_center(teacher_cls_outputs)
        
        return total_loss
    
    @torch.no_grad()
    def _update_center(self, teacher_outputs: List[torch.Tensor]):
        """Update center using EMA."""
        teacher_cat = torch.cat(teacher_outputs, dim=0)
        batch_center = teacher_cat.mean(dim=0, keepdim=True)
        
        if is_distributed():
            batch_center = all_reduce_mean(batch_center)
        
        # self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))


class iBOTLoss(nn.Module):
    """
    iBOT loss for masked patch token prediction.
    
    Computes cross-entropy between student and teacher patch token outputs
    at masked positions only.
    """
    
    def __init__(
        self,
        out_dim: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        # Separate center for patch tokens
        self.register_buffer('center', torch.zeros(1, out_dim))
    
    def forward(
        self,
        student_patch_out: torch.Tensor,
        teacher_patch_out: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute iBOT loss on masked patch tokens.
        
        Args:
            student_patch_out: (B, N, out_dim) student patch predictions
            teacher_patch_out: (B, N, out_dim) teacher patch predictions
            mask: (B, N) boolean mask, True = masked positions
        
        Returns:
            Scalar loss value
        """
        # Get masked tokens only
        # mask: (B, N) -> select positions where mask is True
        B, N, D = student_patch_out.shape
        
        # Flatten and select masked positions
        student_flat = student_patch_out.reshape(B * N, D)
        teacher_flat = teacher_patch_out.reshape(B * N, D)
        mask_flat = mask.reshape(B * N)
        
        student_masked = student_flat[mask_flat]  # (num_masked, D)
        teacher_masked = teacher_flat[mask_flat]  # (num_masked, D)
        
        if student_masked.shape[0] == 0:
            return torch.tensor(0.0, device=student_patch_out.device)
        
        # Apply temperature and compute loss
        student_probs = F.log_softmax(student_masked / self.student_temp, dim=-1)
        teacher_probs = F.softmax((teacher_masked - self.center) / self.teacher_temp, dim=-1).detach()
        
        loss = -torch.sum(teacher_probs * student_probs, dim=-1).mean()
        
        # Update center with all teacher patch tokens (not just masked)
        self._update_center(teacher_patch_out)
        
        return loss
    
    @torch.no_grad()
    def _update_center(self, teacher_patch_out: torch.Tensor):
        """Update center using EMA of all patch tokens."""
        # teacher_patch_out: (B, N, D)
        batch_center = teacher_patch_out.mean(dim=[0, 1], keepdim=False).unsqueeze(0)  # (1, D)
        
        if is_distributed():
            batch_center = all_reduce_mean(batch_center)
        
        # self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        self.center.copy_(self.center * self.center_momentum + batch_center * (1 - self.center_momentum))


class KoLeoLoss(nn.Module):
    """
    KoLeo (Kozachenko-Leonenko) regularizer.
    
    Encourages uniform distribution of features on the hypersphere by
    maximizing the entropy estimated via k-nearest neighbor distances.
    In practice, we minimize the negative log of distances to nearest neighbors.
    
    This prevents representation collapse and improves feature diversity.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute KoLeo loss with safety checks.
        """
        # Ensure features are normalized
        features = F.normalize(features, dim=-1, p=2, eps=self.eps)
        
        n = features.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=features.device, dtype=features.dtype)
        
        # Compute pairwise cosine similarity
        dots = torch.mm(features, features.t())  # (B, B)
        
        # Create mask to exclude diagonal (self-similarity)
        mask = ~torch.eye(n, dtype=torch.bool, device=features.device)
        
        # Get off-diagonal similarities
        off_diag = dots[mask].view(n, n - 1)
        
        # Find max similarity (nearest neighbor) for each sample
        max_sim = off_diag.max(dim=1).values  # (B,)
        
        # Clamp similarity to avoid numerical issues
        # max_sim should be <= 1 for normalized vectors, but clamp to be safe
        max_sim = torch.clamp(max_sim, max=1.0 - self.eps)
        
        # Convert similarity to distance: dist = sqrt(2 - 2*cos) for normalized vectors
        # But we use (1 - cos) as a simpler proxy that's always positive
        dists = 1.0 - max_sim  # (B,)
        
        # Clamp distances to avoid log(0)
        dists = torch.clamp(dists, min=self.eps)
        
        # KoLeo loss: -mean(log(distance))
        loss = -torch.log(dists).mean()
        
        # Final safety clamp
        loss = torch.clamp(loss, min=-100, max=100)
        
        return loss


class DINOv2Trainer(BaseTrainer):
    """
    Full DINOv2 Trainer with iBOT and KoLeo.
    
    Implements:
    - DINO loss on CLS tokens (self-distillation)
    - iBOT loss on masked patch tokens (masked image modeling)
    - KoLeo regularizer (uniform feature distribution)
    - EMA teacher for stable targets
    - Multi-crop training (global + local crops)
    - Optional CFM for adaptive feature modulation
    """
    
    def _build_model(self):
        """Build DINOv2-specific model components."""
        # Head configuration
        head_config = self.config.get('head', {})
        out_dim = head_config.get('out_dim', 16384)
        hidden_dim = head_config.get('hidden_dim', 1024)
        bottleneck_dim = head_config.get('bottleneck_dim', 256)
        
        # Build student DINO head (for CLS token)
        self.student_head = DINOv2Head(
            in_dim=self.embed_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        ).to(self.device)
        
        # Build student iBOT head (for patch tokens) - shares last layer with DINO head
        # iBOT requires discrete patch tokens, which only ViT provides
        # ConvNeXt outputs spatial features that are not compatible with iBOT
        self.use_ibot = self.config.get('use_ibot', True)
        
        # Auto-disable iBOT for ConvNeXt backbone
        if self.use_ibot and self.backbone_type == 'convnext':
            if self.is_main:
                print("[DINOv2] WARNING: iBOT is not compatible with ConvNeXt backbone.")
                print("[DINOv2] iBOT requires discrete patch tokens (ViT only). Disabling iBOT.")
            self.use_ibot = False
        
        if self.use_ibot:
            self.student_ibot_head = iBOTHead(
                in_dim=self.embed_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                shared_head=self.student_head,  # Share prototype layer
            ).to(self.device)
        else:
            self.student_ibot_head = None
        
        # Build teacher (EMA of student)
        self.teacher_backbone = EMAModel(
            self.backbone,
            momentum=self.config.get('ema_momentum', 0.996),
            device=self.device,
        )
        self.teacher_head = EMAModel(
            self.student_head,
            momentum=self.config.get('ema_momentum', 0.996),
            device=self.device,
        )
        
        if self.use_ibot:
            self.teacher_ibot_head = EMAModel(
                self.student_ibot_head,
                momentum=self.config.get('ema_momentum', 0.996),
                device=self.device,
            )
        else:
            self.teacher_ibot_head = None
        
        # Build teacher CFM if using CFM
        if self.cfm is not None:
            self.teacher_cfm = EMAModel(
                self.cfm,
                momentum=self.config.get('ema_momentum', 0.996),
                device=self.device,
            )
        else:
            self.teacher_cfm = None
        
        # Build losses
        self.dino_loss = DINOLoss(
            out_dim=out_dim,
            student_temp=self.config.get('student_temp', 0.1),
            teacher_temp=self.config.get('teacher_temp', 0.04),
            center_momentum=self.config.get('center_momentum', 0.9),
        ).to(self.device)
        
        if self.use_ibot:
            self.ibot_loss = iBOTLoss(
                out_dim=out_dim,
                student_temp=self.config.get('student_temp', 0.1),
                teacher_temp=self.config.get('teacher_temp', 0.04),
                center_momentum=self.config.get('center_momentum', 0.9),
            ).to(self.device)
        
        # KoLeo regularizer
        self.use_koleo = self.config.get('use_koleo', True)
        if self.use_koleo:
            self.koleo_loss = KoLeoLoss()
        
        # Loss weights
        self.dino_loss_weight = self.config.get('dino_loss_weight', 1.0)
        self.ibot_loss_weight = self.config.get('ibot_loss_weight', 1.0)
        self.koleo_loss_weight = self.config.get('koleo_loss_weight', 0.1)
        
        # Masking configuration
        self.mask_ratio = self.config.get('mask_ratio', 0.3)
        
        # Wrap student components in DDP
        self.backbone = self._wrap_ddp(self.backbone)
        self.student_head = self._wrap_ddp(self.student_head)
        if self.use_ibot:
            self.student_ibot_head = self._wrap_ddp(self.student_ibot_head)
        if self.cfm is not None:
            self.cfm = self._wrap_ddp(self.cfm)
            
        # Set static graph for DDP - required because we call backbone multiple times
        # per iteration (once per crop in multi-crop training)
        if is_distributed():
            self.backbone._set_static_graph()
            self.student_head._set_static_graph()
            if self.use_ibot and self.student_ibot_head is not None:
                self.student_ibot_head._set_static_graph()
            if self.cfm is not None:
                self.cfm._set_static_graph()
        
        # EMA momentum schedule
        self.ema_schedule = cosine_scheduler(
            base_value=self.config.get('ema_momentum', 0.996),
            final_value=1.0,
            epochs=self.config.get('epochs', 100),
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=0,
        )
        
        # Teacher temperature schedule (warmup)
        teacher_temp_warmup = self.config.get('teacher_temp_warmup_epochs', 30)
        self.teacher_temp_schedule = cosine_scheduler(
            base_value=self.config.get('teacher_temp', 0.04),
            final_value=self.config.get('teacher_temp', 0.04),
            epochs=self.config.get('epochs', 100),
            steps_per_epoch=self.steps_per_epoch,
            warmup_epochs=teacher_temp_warmup,
            warmup_value=self.config.get('teacher_temp_warmup', 0.04),
        )
        
        if self.is_main:
            student_head_params = sum(p.numel() for p in self.student_head.parameters())
            print(f"[DINOv2] Built student DINO head: {student_head_params:,} params")
            if self.use_ibot:
                # iBOT head shares last layer, so count only MLP params
                # ibot_mlp_params = sum(p.numel() for p in self.student_ibot_head.module.mlp.parameters() 
                #                      if hasattr(self.student_ibot_head, 'module') 
                #                      else p.numel() for p in self.student_ibot_head.mlp.parameters())
                # iBOT head shares last layer, so count only MLP params
                ibot_module = self.student_ibot_head.module if hasattr(self.student_ibot_head, 'module') else self.student_ibot_head
                ibot_mlp_params = sum(p.numel() for p in ibot_module.mlp.parameters())
                print(f"[DINOv2] Built student iBOT head: {ibot_mlp_params:,} params (MLP only, shares prototypes)")
            print(f"[DINOv2] Output dim: {out_dim}")
            print(f"[DINOv2] Student temp: {self.config.get('student_temp', 0.1)}")
            print(f"[DINOv2] Teacher temp: {self.config.get('teacher_temp', 0.04)}")
            # print(f"[DINOv2] iBOT enabled: {self.use_ibot}, mask_ratio: {self.mask_ratio}")
            if self.backbone_type == 'convnext' and self.config.get('use_ibot', True):
                print(f"[DINOv2] iBOT enabled: {self.use_ibot} (auto-disabled for ConvNeXt)")
            else:
                print(f"[DINOv2] iBOT enabled: {self.use_ibot}, mask_ratio: {self.mask_ratio}")
            print(f"[DINOv2] KoLeo enabled: {self.use_koleo}, weight: {self.koleo_loss_weight}")
            print(f"[DINOv2] Loss weights - DINO: {self.dino_loss_weight}, iBOT: {self.ibot_loss_weight}")
    
    def _get_param_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups with separate handling for different components."""
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
        
        # DINO head parameters
        for name, param in self.student_head.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                not_regularized.append(param)
            else:
                regularized.append(param)
        
        # iBOT head parameters (only MLP, last layer is shared)
        if self.use_ibot and self.student_ibot_head is not None:
            ibot_module = self.student_ibot_head.module if hasattr(self.student_ibot_head, 'module') else self.student_ibot_head
            for name, param in ibot_module.mlp.named_parameters():
                if not param.requires_grad:
                    continue
                if 'bias' in name or 'norm' in name:
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
        """Set student components to training mode."""
        self.student_head.train()
        if self.use_ibot and self.student_ibot_head is not None:
            self.student_ibot_head.train()
        # Teacher is always in eval mode
        self.teacher_backbone.ema_model.eval()
        self.teacher_head.ema_model.eval()
        if self.teacher_ibot_head is not None:
            self.teacher_ibot_head.ema_model.eval()
        if self.teacher_cfm is not None:
            self.teacher_cfm.ema_model.eval()
    
    def _get_head_param_counts(self) -> tuple:
        """Get parameter counts for DINOv2 heads."""
        head_module = self.student_head.module if hasattr(self.student_head, 'module') else self.student_head
        total = sum(p.numel() for p in head_module.parameters())
        trainable = sum(p.numel() for p in head_module.parameters() if p.requires_grad)
        
        # Add iBOT MLP params (last layer is shared, don't count twice)
        if self.use_ibot and self.student_ibot_head is not None:
            ibot_module = self.student_ibot_head.module if hasattr(self.student_ibot_head, 'module') else self.student_ibot_head
            total += sum(p.numel() for p in ibot_module.mlp.parameters())
            trainable += sum(p.numel() for p in ibot_module.mlp.parameters() if p.requires_grad)
        
        return total, trainable
    
    def _generate_mask(self, batch_size: int, num_patches: int) -> torch.Tensor:
        """
        Generate random mask for iBOT.
        
        Args:
            batch_size: Batch size
            num_patches: Number of patches
        
        Returns:
            (B, num_patches) boolean mask, True = masked
        """
        num_masked = int(num_patches * self.mask_ratio)
        mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            masked_indices = torch.randperm(num_patches, device=self.device)[:num_masked]
            mask[i, masked_indices] = True
        
        return mask
    
    def _train_step(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single DINOv2 training step.
        
        Args:
            batch: List of crop tensors [global1, global2, local1, ..., localN]
        
        Returns:
            Dictionary containing loss values
        """
        # Move crops to device
        crops = [crop.to(self.device, non_blocking=True) for crop in batch]
        n_global = self.config.get('n_global_crops', 2)
        
        global_crops = crops[:n_global]
        all_crops = crops
        
        # Update EMA momentum
        current_momentum = self.ema_schedule[self.global_step] if self.global_step < len(self.ema_schedule) else 1.0
        self.teacher_backbone.set_momentum(current_momentum)
        self.teacher_head.set_momentum(current_momentum)
        if self.teacher_ibot_head is not None:
            self.teacher_ibot_head.set_momentum(current_momentum)
        if self.teacher_cfm is not None:
            self.teacher_cfm.set_momentum(current_momentum)
        
        # Update teacher temperature
        if self.global_step < len(self.teacher_temp_schedule):
            current_teacher_temp = self.teacher_temp_schedule[self.global_step]
            self.dino_loss.teacher_temp = current_teacher_temp
            if self.use_ibot:
                self.ibot_loss.teacher_temp = current_teacher_temp
        
        # Get current CFM weight from curriculum
        cfm_weight = self.current_cfm_weight
        
        # ============ Teacher forward (no gradients) ============
        teacher_cls_outputs = []
        teacher_patch_outputs = []
        
        with torch.no_grad():
            for crop in global_crops:
                # Get CFM modulations for teacher
                if self.teacher_cfm is not None and cfm_weight > 0:
                    modulations = self.teacher_cfm(crop)
                    if cfm_weight < 1.0:
                        modulations = self._apply_cfm_weight_to_modulations(modulations, cfm_weight)
                else:
                    modulations = None
                
                # Teacher backbone forward - get CLS and patch tokens
                cls_token, patch_tokens = self.teacher_backbone(
                    crop, modulations=modulations, return_all_tokens=True
                )
                
                # DINO head on CLS token
                cls_output = self.teacher_head(cls_token)
                teacher_cls_outputs.append(cls_output)
                
                # iBOT head on patch tokens (for masked prediction targets)
                if self.use_ibot:
                    patch_output = self.teacher_ibot_head(patch_tokens)
                    teacher_patch_outputs.append(patch_output)
        
        # ============ Student forward ============
        student_cls_outputs = []
        student_patch_outputs = []
        masks = []
        student_cls_features = []  # For KoLeo
        
        for i, crop in enumerate(all_crops):
            # Get CFM modulations for student
            if self.cfm is not None and cfm_weight > 0:
                modulations = self.cfm(crop)
                if cfm_weight < 1.0:
                    modulations = self._apply_cfm_weight_to_modulations(modulations, cfm_weight)
            else:
                modulations = None
            
            # Generate mask for global crops only (iBOT)
            # Note: iBOT is only used with ViT backbone (auto-disabled for ConvNeXt in _build_model)
            is_global = i < n_global
            if self.use_ibot and is_global:
                B = crop.shape[0]
                H, W = crop.shape[2], crop.shape[3]
                # Get patch size from backbone (ViT only, ConvNeXt has iBOT disabled)
                backbone_module = self.backbone.module if hasattr(self.backbone, 'module') else self.backbone
                patch_size = getattr(backbone_module, 'patch_size', None)
                if patch_size is not None:
                    num_patches = (H // patch_size) ** 2
                    mask = self._generate_mask(B, num_patches)
                else:
                    mask = None
            else:
                mask = None
            
            # Student backbone forward with optional mask
            # Note: mask parameter is only used by ViT for masked token replacement
            if self.backbone_type == 'vit':
                cls_token, patch_tokens = self.backbone(
                    crop, modulations=modulations, return_all_tokens=True, mask=mask
                )
            else:
                # ConvNeXt: no masking, returns (global_features, spatial_features)
                cls_token, patch_tokens = self.backbone(
                    crop, modulations=modulations, return_all_tokens=True
                )
            
            # Save CLS features for KoLeo (before projection)
            if is_global:
                student_cls_features.append(cls_token)
            
            # DINO head on CLS token
            cls_output = self.student_head(cls_token)
            student_cls_outputs.append(cls_output)
            
            # iBOT head on patch tokens (only for global crops)
            if self.use_ibot and is_global:
                patch_output = self.student_ibot_head(patch_tokens)
                student_patch_outputs.append(patch_output)
                masks.append(mask)
        
        # ============ Compute losses ============
        
        # DINO loss (CLS token distillation)
        loss_dino = self.dino_loss(student_cls_outputs, teacher_cls_outputs)
        total_loss = self.dino_loss_weight * loss_dino
        
        # iBOT loss (masked patch prediction)
        loss_ibot = torch.tensor(0.0, device=self.device)
        if self.use_ibot and len(student_patch_outputs) > 0:
            ibot_losses = []
            for s_patch, t_patch, mask in zip(student_patch_outputs, teacher_patch_outputs, masks):
                ibot_loss_term = self.ibot_loss(s_patch, t_patch, mask)
                ibot_losses.append(ibot_loss_term)
            loss_ibot = sum(ibot_losses) / len(ibot_losses)
            total_loss = total_loss + self.ibot_loss_weight * loss_ibot
        
        # KoLeo loss (uniform feature distribution)
        loss_koleo = torch.tensor(0.0, device=self.device)
        if self.use_koleo and len(student_cls_features) > 0:
            # Concatenate CLS features from global crops
            cls_features_cat = torch.cat(student_cls_features, dim=0)
            loss_koleo = self.koleo_loss(cls_features_cat)
            total_loss = total_loss + self.koleo_loss_weight * loss_koleo
        
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
            if hasattr(self.student_head, 'module'):
                params.extend(self.student_head.module.parameters())
            else:
                params.extend(self.student_head.parameters())
            if self.use_ibot and self.student_ibot_head is not None:
                ibot_module = self.student_ibot_head.module if hasattr(self.student_ibot_head, 'module') else self.student_ibot_head
                params.extend(ibot_module.mlp.parameters())
            if self.cfm is not None and self._should_train_cfm():
                if hasattr(self.cfm, 'module'):
                    params.extend(self.cfm.module.parameters())
                else:
                    params.extend(self.cfm.parameters())
            
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        
        self.optimizer.step()
        
        # ============ Update teacher with EMA ============
        self.teacher_backbone.update(self.backbone)
        self.teacher_head.update(self.student_head)
        if self.use_ibot and self.teacher_ibot_head is not None:
            self.teacher_ibot_head.update(self.student_ibot_head)
        if self.cfm is not None and self.teacher_cfm is not None and cfm_weight > 0:
            self.teacher_cfm.update(self.cfm)
        
        return {
            'loss': total_loss,
            'loss_dino': loss_dino,
            'loss_ibot': loss_ibot,
            'loss_koleo': loss_koleo,
            'ema_momentum': current_momentum,
            'teacher_temp': self.dino_loss.teacher_temp,
            'cfm_weight': cfm_weight,
        }
    
    def _apply_cfm_weight_to_modulations(
        self, 
        modulations: List[Dict[str, torch.Tensor]], 
        weight: float
    ) -> List[Dict[str, torch.Tensor]]:
        """Apply curriculum weight to CFM modulations."""
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
        """Save DINOv2-specific state."""
        state = {
            'teacher_backbone': self.teacher_backbone.state_dict(),
            'teacher_head': self.teacher_head.state_dict(),
            'dino_center': self.dino_loss.center,
        }
        
        # Handle DDP wrapper for student head
        if hasattr(self.student_head, 'module'):
            state['student_head'] = self.student_head.module.state_dict()
        else:
            state['student_head'] = self.student_head.state_dict()
        
        # iBOT state
        if self.use_ibot:
            if self.teacher_ibot_head is not None:
                state['teacher_ibot_head'] = self.teacher_ibot_head.state_dict()
            if self.student_ibot_head is not None:
                if hasattr(self.student_ibot_head, 'module'):
                    state['student_ibot_head'] = self.student_ibot_head.module.state_dict()
                else:
                    state['student_ibot_head'] = self.student_ibot_head.state_dict()
            state['ibot_center'] = self.ibot_loss.center
        
        # CFM state
        if self.teacher_cfm is not None:
            state['teacher_cfm'] = self.teacher_cfm.state_dict()
        
        return state
    
    def _load_checkpoint_hook(self, checkpoint: Dict[str, Any]):
        """Load DINOv2-specific state."""
        if 'teacher_backbone' in checkpoint:
            self.teacher_backbone.load_state_dict(checkpoint['teacher_backbone'])
        if 'teacher_head' in checkpoint:
            self.teacher_head.load_state_dict(checkpoint['teacher_head'])
        if 'dino_center' in checkpoint:
            self.dino_loss.center = checkpoint['dino_center']
        if 'student_head' in checkpoint:
            if hasattr(self.student_head, 'module'):
                self.student_head.module.load_state_dict(checkpoint['student_head'])
            else:
                self.student_head.load_state_dict(checkpoint['student_head'])
        
        # iBOT state
        if self.use_ibot:
            if 'teacher_ibot_head' in checkpoint and self.teacher_ibot_head is not None:
                self.teacher_ibot_head.load_state_dict(checkpoint['teacher_ibot_head'])
            if 'student_ibot_head' in checkpoint and self.student_ibot_head is not None:
                if hasattr(self.student_ibot_head, 'module'):
                    self.student_ibot_head.module.load_state_dict(checkpoint['student_ibot_head'])
                else:
                    self.student_ibot_head.load_state_dict(checkpoint['student_ibot_head'])
            if 'ibot_center' in checkpoint:
                self.ibot_loss.center = checkpoint['ibot_center']
        
        # CFM state
        if self.teacher_cfm is not None and 'teacher_cfm' in checkpoint:
            self.teacher_cfm.load_state_dict(checkpoint['teacher_cfm'])


# =============================================================================
#                         TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DINOv2 Trainer Components Test")
    print("=" * 60)
    
    # Test DINOLoss
    print("\n--- DINOLoss Test ---")
    batch_size = 4
    out_dim = 1024
    n_global = 2
    n_local = 4
    
    dino_loss_fn = DINOLoss(out_dim=out_dim, student_temp=0.1, teacher_temp=0.04)
    
    student_cls = [torch.randn(batch_size, out_dim) for _ in range(n_global + n_local)]
    teacher_cls = [torch.randn(batch_size, out_dim) for _ in range(n_global)]
    
    loss = dino_loss_fn(student_cls, teacher_cls)
    print(f"DINO Loss: {loss.item():.4f}")
    print(f"Center shape: {dino_loss_fn.center.shape}")
    
    # Test iBOTLoss
    print("\n--- iBOTLoss Test ---")
    num_patches = 144
    
    ibot_loss_fn = iBOTLoss(out_dim=out_dim, student_temp=0.1, teacher_temp=0.04)
    
    student_patch = torch.randn(batch_size, num_patches, out_dim)
    teacher_patch = torch.randn(batch_size, num_patches, out_dim)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)
    mask[:, :30] = True  # Mask first 30 patches
    
    loss = ibot_loss_fn(student_patch, teacher_patch, mask)
    print(f"iBOT Loss: {loss.item():.4f}")
    print(f"Center shape: {ibot_loss_fn.center.shape}")
    print(f"Masked positions: {mask.sum().item()}")
    
    # Test KoLeoLoss
    print("\n--- KoLeoLoss Test ---")
    koleo_loss_fn = KoLeoLoss()
    
    features = F.normalize(torch.randn(batch_size * 2, 384), dim=-1)
    loss = koleo_loss_fn(features)
    print(f"KoLeo Loss: {loss.item():.4f}")
    
    # Test that losses are differentiable
    print("\n--- Gradient Test ---")
    student_cls_grad = [torch.randn(batch_size, out_dim, requires_grad=True) 
                        for _ in range(n_global + n_local)]
    loss = dino_loss_fn(student_cls_grad, teacher_cls)
    loss.backward()
    assert all(s.grad is not None for s in student_cls_grad), "DINO loss should be differentiable"
    print("DINO loss is differentiable")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nNote: Full trainer testing requires data. Use scripts/train.py with small data.")