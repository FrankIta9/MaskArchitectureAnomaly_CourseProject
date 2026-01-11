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
import math
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
        energy_warmup_start_epoch: int = 0,  # Virtual starting epoch (for resume from weights)
        logit_norm_enabled: bool = False,
        logit_norm_tau: float = 0.04,
        logit_norm_eps: float = 1e-6,
        max_epochs: int = 50,  # Total training epochs
    ):
        nn.Module.__init__(self)
        self.logit_norm_enabled = logit_norm_enabled
        self.logit_norm_tau = logit_norm_tau
        self.logit_norm_eps = logit_norm_eps
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
                warmup_start_epoch=energy_warmup_start_epoch,  # Virtual starting epoch for resume
            )
        self._last_energy_stats = None  # Store energy stats for logging

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
        # =================================================================
        # STEP 1: Save original logits BEFORE normalization
        # =================================================================
        # Clone logits ONLY if both Energy Loss and Logit Norm are enabled
        # This avoids unnecessary overhead if one is disabled
        if (self.eim_enabled and 
            self.logit_norm_enabled and 
            class_queries_logits is not None):
            # Clone for Energy Loss (will be used AFTER normalization)
            # Clone WITHOUT detach to maintain gradients for Energy Loss
            # Use contiguous() to ensure proper memory layout for AMP compatibility
            class_queries_logits_original = class_queries_logits.clone().contiguous()
            
            # Safety check: ensure logits are finite and real (no NaN/Inf/complex)
            if not torch.isfinite(class_queries_logits_original).all():
                # If original logits contain NaN/Inf, replace with zeros to avoid crashes
                class_queries_logits_original = torch.where(
                    torch.isfinite(class_queries_logits_original),
                    class_queries_logits_original,
                    torch.zeros_like(class_queries_logits_original)
                )
            # Ensure logits are real (not complex) - should always be true, but safety check
            if class_queries_logits_original.is_complex():
                # Convert complex to real (take real part) - should not happen, but safety
                class_queries_logits_original = class_queries_logits_original.real
        else:
            # If Logit Norm disabled, use same logits (no clone needed)
            class_queries_logits_original = class_queries_logits

        # =================================================================
        # STEP 2: Apply Logit Normalization (modifies IN-PLACE)
        # =================================================================
        # Normalization improves calibration for matcher/standard losses
        if self.logit_norm_enabled and class_queries_logits is not None:
            # Force float32 for logit normalization computations (better numerical stability)
            # This prevents NaN/Inf issues similar to energy/logsumexp
            logits_f32 = class_queries_logits.float() if class_queries_logits.dtype != torch.float32 else class_queries_logits
            logits_noobj = logits_f32[..., :-1]                 # [B, Q, C]
            norm = logits_noobj.norm(p=2, dim=-1, keepdim=True)  # [B, Q, 1]
            class_queries_logits_normalized = logits_f32 / (self.logit_norm_tau * (norm + self.logit_norm_eps))
            # Convert back to original dtype if needed
            if class_queries_logits.dtype != torch.float32:
                class_queries_logits = class_queries_logits_normalized.to(class_queries_logits.dtype)
            else:
                class_queries_logits = class_queries_logits_normalized

        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,  # Use normalized logits
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)  # Use normalized logits
        
        # =================================================================
        # STEP 3: Energy Loss uses ORIGINAL logits (not normalized)
        # =================================================================
        # Energy Loss requires original scale for correct margin values (m_in=-25.0)
        # Logit normalization changes scale drastically, making margins invalid
        losses = {**loss_masks, **loss_classes}
        if self.eim_enabled and class_queries_logits_original is not None:
            # Safety check: ensure logits are finite before computing energy loss
            if torch.isfinite(class_queries_logits_original).all():
                energy_loss = self.energy_ood_loss(class_queries_logits_original)  # Use original logits
                # Ensure loss is finite (safety check)
                if torch.isfinite(energy_loss):
                    losses["eim"] = energy_loss  # Keep "eim" key for compatibility with logging
                else:
                    # If energy loss is NaN/Inf, use zero loss to avoid crashing
                    losses["eim"] = torch.tensor(0.0, device=class_queries_logits_original.device, dtype=class_queries_logits_original.dtype, requires_grad=True)
            else:
                # Logits contain NaN/Inf, use zero loss
                losses["eim"] = torch.tensor(0.0, device=class_queries_logits_original.device, dtype=class_queries_logits_original.dtype, requires_grad=True)
            
            # Compute and store energy statistics for logging
            # Statistics are computed on original logits (before normalization)
            # Wrap in try-except to avoid crashes from numerical issues
            # Only compute stats every N batches to reduce overhead
            try:
                with torch.no_grad():
                    # Check for NaN/Inf before computing stats
                    if torch.isfinite(class_queries_logits_original).all():
                        energy_stats = self.energy_ood_loss.get_energy_stats(class_queries_logits_original)
                        # Verify stats are valid (all finite numbers)
                        stats_valid = all(
                            isinstance(v, (int, float)) and 
                            not (math.isnan(v) or math.isinf(v))
                            for v in energy_stats.values() if isinstance(v, (int, float, str)) and v != "warmup" and v != "active"
                        )
                        if stats_valid:
                            self._last_energy_stats = energy_stats  # Store for logging in loss_total
                        else:
                            # Invalid stats, skip logging for this batch
                            self._last_energy_stats = None
                    else:
                        # Logits contain NaN/Inf, skip stats calculation
                        self._last_energy_stats = None
            except Exception:
                # If stats calculation fails for any reason, skip it but don't crash
                # This ensures training continues even if stats computation has issues
                self._last_energy_stats = None

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
        
        # Log Energy Loss statistics for monitoring
        if self.eim_enabled and self._last_energy_stats is not None:
            stats = self._last_energy_stats
            log_fn("energy/weight_current", stats["energy_weight_current"], sync_dist=True)
            log_fn("energy/weight_max", stats["energy_weight_max"], sync_dist=True)
            log_fn("energy/mean", stats["energy_mean"], sync_dist=True)
            log_fn("energy/std", stats["energy_std"], sync_dist=True)
            log_fn("energy/min", stats["energy_min"], sync_dist=True)
            log_fn("energy/max", stats["energy_max"], sync_dist=True)
            # Log warmup phase as a metric (0.0 = warmup, 1.0 = active)
            warmup_phase_value = 0.0 if stats["warmup_phase"] == "warmup" else 1.0
            log_fn("energy/warmup_phase", warmup_phase_value, sync_dist=True)
            
            # Debug: Check if energy values are reasonable (should be around m_in=-25.0 for ID)
            # Log warning if energy is too high (might indicate scale issues)
            if stats["energy_weight_current"] > 0.0:  # Only check when energy is active
                if stats["energy_mean"] > -10.0:  # Energy too high (should be around -25.0 for ID)
                    # This would indicate potential scale issues, but we log it for monitoring
                    pass  # Just log the values, don't raise error

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
