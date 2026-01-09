# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)
from training.energy_ood_loss import EnergyOODLossWithWarmup


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        eim_enabled: bool = True,
        eim_temperature: float = 1.0,
        eim_weight: float = 0.002,  # Max weight after warmup
        energy_warmup_epochs: int = 15,  # Epochs with energy disabled
        max_epochs: int = 50,  # Total training epochs
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )
        
        # Energy-Based OOD Loss WITH WARMUP (avoids conflict with OE)
        # Phase 1 (0-15 epochs): energy disabled, pure OE training
        # Phase 2 (15+ epochs): energy gradually enabled for refinement
        self.eim_enabled = eim_enabled  # Kept for compatibility
        if eim_enabled:
            self.energy_ood_loss = EnergyOODLossWithWarmup(
                temperature=eim_temperature,
                max_weight=eim_weight,
                warmup_epochs=energy_warmup_epochs,
                max_epochs=max_epochs,
                warmup_schedule="cosine",
            )

    def set_epoch(self, epoch: int):
        """Update current epoch for energy warmup scheduling."""
        if self.eim_enabled:
            self.energy_ood_loss.set_epoch(epoch)
    
    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ): 
        if self.logit_norm_enabled and class_queries_logits is not None:
            logits_noobj = class_queries_logits[..., :-1]                 # [B, Q, C]
            norm = logits_noobj.norm(p=2, dim=-1, keepdim=True)  # [B, Q, 1]
            class_queries_logits = class_queries_logits / (self.logit_norm_tau * (norm + self.logit_norm_eps))
            
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)
        
        # Add Energy-Based OOD Loss
        losses = {**loss_masks, **loss_classes}
        if self.eim_enabled and class_queries_logits is not None:
            energy_loss = self.energy_ood_loss(class_queries_logits)
            losses["eim"] = energy_loss  # Keep "eim" key for compatibility with logging

        return losses

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "eim" in loss_key:
                # EIM loss is already weighted in the EnhancedIsotropyLoss class
                weighted_loss = loss
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
