# ---------------------------------------------------------------
# Energy-Based Out-of-Distribution Detection Loss
# Based on: "Energy-based Out-of-distribution Detection" (NeurIPS 2020)
# Adapted for Mask2Former-style segmentation with Outlier Exposure
# WITH WARMUP SCHEDULER to avoid conflict with OE in early training
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class EnergyOODLoss(nn.Module):
    """
    Energy-Based Out-of-Distribution Detection Loss.
    
    This loss encourages the model to produce:
    - LOW energy scores for in-distribution (ID) classes
    - HIGH energy scores for out-of-distribution (OOD) samples
    
    Energy Score: E(x) = -log(sum(exp(logits)))
    
    For segmentation with Outlier Exposure:
    - ID queries (matched to ground truth): minimize energy
    - Unmatched queries (potentially OOD): can have higher energy
    
    This loss is computed on class logits and encourages better
    separation between ID and OOD predictions without conflicting
    with Outlier Exposure (unlike isotropy-based approaches).
    
    Args:
        temperature: Temperature scaling for energy computation (default: 1.0)
        weight: Weight for the energy regularization term (default: 0.05)
        m_in: Margin for in-distribution samples (default: -25.0)
        m_out: Margin for out-of-distribution samples (default: -7.0)
    
    References:
        Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        weight: float = 0.05,
        m_in: float = -25.0,
        m_out: float = -7.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.m_in = m_in  # Margin for ID samples (should have energy < m_in)
        self.m_out = m_out  # Margin for OOD samples (should have energy > m_out)
        
    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score for given logits.
        
        Energy: E(x) = -T * log(sum(exp(logits / T)))
        
        Args:
            logits: Class logits tensor (batch, num_queries, num_classes + 1)
            
        Returns:
            Energy scores (batch, num_queries)
        """
        # Safety check: ensure logits are real (not complex) and finite
        if logits.is_complex():
            # Convert complex to real (take real part) - should not happen
            logits = logits.real
        
        if not torch.isfinite(logits).all():
            # If logits contain NaN/Inf, return safe default energy values
            return torch.full((logits.shape[0], logits.shape[1]), -25.0, device=logits.device, dtype=logits.dtype)
        
        # Force float32 for energy/softmax/logsumexp computations (better numerical stability)
        # Scale logits by temperature (cast to float32 before computation)
        logits_f32 = logits.float() if logits.dtype != torch.float32 else logits
        scaled_logits = logits_f32 / self.temperature
        
        # Exclude "no object" class (last class) for ID energy
        # We only consider ID classes for energy computation
        id_logits = scaled_logits[..., :-1]  # Remove "no object" class
        
        # Safety check: ensure id_logits has valid shape and values
        if id_logits.numel() == 0:
            # Return zeros if no ID logits (shouldn't happen, but safety check)
            return torch.zeros(logits.shape[0], logits.shape[1], device=logits.device, dtype=logits.dtype)
        
        # Clamp logits to reasonable range to avoid numerical overflow/underflow
        # Large values in exp can cause Inf, which then becomes NaN in log
        # Very negative values can cause underflow, but logsumexp handles this
        id_logits = torch.clamp(id_logits, min=-50.0, max=50.0)
        
        # Ensure id_logits is real and finite before logsumexp
        if id_logits.is_complex():
            id_logits = id_logits.real
        if not torch.isfinite(id_logits).all():
            id_logits = torch.where(torch.isfinite(id_logits), id_logits, torch.zeros_like(id_logits))
        
        # Compute logsumexp for numerical stability (ensure float32)
        # Energy = -T * logsumexp(logits / T)
        # logsumexp is numerically stable and handles underflow/overflow
        energy = -self.temperature * torch.logsumexp(id_logits, dim=-1)
        # Convert back to original dtype if needed
        if logits.dtype != torch.float32:
            energy = energy.to(logits.dtype)
        
        # Ensure energy is real (not complex) - should always be true
        if energy.is_complex():
            energy = energy.real
        
        # Clamp energy to reasonable range (avoid NaN/Inf)
        energy = torch.clamp(energy, min=-100.0, max=100.0)
        
        # Replace any remaining NaN/Inf with finite values
        energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
        
        # Final safety: ensure energy is real float, not complex
        if not energy.dtype.is_floating_point:
            energy = energy.float()
        
        return energy
    
    def forward(
        self,
        class_logits: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Energy-based OOD regularization loss.
        
        This is a REGULARIZATION term that encourages:
        - Lower energy for predictions (assuming most are ID)
        - Energy-based separation for better OOD detection at inference
        
        For Outlier Exposure training:
        - The model sees both ID (Cityscapes) and OOD (COCO) samples
        - OE naturally creates high energy for OOD through standard losses
        - This loss adds gentle regularization to improve separation
        
        Args:
            class_logits: Class logits (batch_size, num_queries, num_classes + 1)
            target_labels: Optional target labels (not used in unsupervised OE)
            
        Returns:
            Energy regularization loss
        """
        # Safety check: ensure logits are finite
        if not torch.isfinite(class_logits).all():
            # If logits contain NaN/Inf, return zero loss to avoid crashing
            return torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        # Compute energy scores for all queries
        energy = self.compute_energy(class_logits)  # (batch, num_queries)
        
        # Safety check: ensure energy is finite before computing loss
        if not torch.isfinite(energy).all():
            # Replace NaN/Inf with safe values
            energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
        
        # Energy-based regularization:
        # Encourage low energy overall (ID-like behavior)
        # The standard Mask2Former loss + OE will naturally create
        # high energy for outliers, so we just regularize towards
        # reasonable energy values
        
        # Option 1: Simple energy regularization (encourage low energy)
        # This helps the model learn to produce low energy for ID samples
        # OE will naturally push energy higher for outliers via classification loss
        
        # Use hinge loss: max(0, energy - m_in)
        # Penalize if energy is too high (> m_in threshold)
        loss = F.relu(energy - self.m_in).mean()
        
        # Ensure loss is finite (safety check)
        if not torch.isfinite(loss):
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
        
        # Alternative: Could use MSE to target specific energy value
        # loss = F.mse_loss(energy, torch.full_like(energy, self.m_in))
        
        return loss * self.weight
    
    def get_energy_stats(self, class_logits: torch.Tensor) -> dict:
        """
        Get energy statistics for monitoring/debugging.
        
        Args:
            class_logits: Class logits (batch_size, num_queries, num_classes + 1)
            
        Returns:
            Dictionary with energy statistics
        """
        with torch.no_grad():
            energy = self.compute_energy(class_logits)
            
            # Ensure energy is finite before computing stats
            if not torch.isfinite(energy).all():
                energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
            
            # Compute statistics with safety checks
            energy_mean = energy.mean().item() if energy.numel() > 0 else -25.0
            energy_std = energy.std().item() if energy.numel() > 1 and torch.isfinite(energy.std()) else 0.0
            energy_min = energy.min().item() if energy.numel() > 0 and torch.isfinite(energy.min()) else -25.0
            energy_max = energy.max().item() if energy.numel() > 0 and torch.isfinite(energy.max()) else -25.0
            
            # Ensure all stats are finite (no NaN/Inf)
            energy_mean = energy_mean if isinstance(energy_mean, (int, float)) and (not math.isnan(energy_mean) and not math.isinf(energy_mean)) else -25.0
            energy_std = energy_std if isinstance(energy_std, (int, float)) and (not math.isnan(energy_std) and not math.isinf(energy_std)) else 0.0
            energy_min = energy_min if isinstance(energy_min, (int, float)) and (not math.isnan(energy_min) and not math.isinf(energy_min)) else -25.0
            energy_max = energy_max if isinstance(energy_max, (int, float)) and (not math.isnan(energy_max) and not math.isinf(energy_max)) else -25.0
            
            return {
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "energy_min": energy_min,
                "energy_max": energy_max,
            }


class EnergyOODLossWithWarmup(nn.Module):
    """
    Energy-Based OOD Loss with Warmup Scheduler.
    
    SOLVES CONFLICT WITH OUTLIER EXPOSURE:
    - Phase 1 (epochs 0-warmup_epochs): Energy weight = 0 (DISABLED)
      → Model learns via OE: COCO outliers → "no object" prediction
      → No conflict, stable convergence
    
    - Phase 2 (epochs warmup_epochs-max_epochs): Energy weight gradually increases
      → Model already knows outlier="no object", now refines energy separation
      → Cosine warmup: 0 → max_weight
    
    This approach follows best practices:
    1. Let OE teach outlier detection first (via standard losses)
    2. Then add energy regularization to refine confidence calibration
    3. Avoids early training instability from conflicting objectives
    
    Args:
        base_loss: EnergyOODLoss instance (with max weight configured)
        warmup_epochs: Number of epochs with energy disabled (default: 15)
        max_epochs: Total training epochs (for cosine schedule)
        warmup_schedule: "cosine" or "linear" (default: "cosine")
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        max_weight: float = 0.002,  # Conservative max weight
        m_in: float = -25.0,
        warmup_epochs: int = 15,
        max_epochs: int = 50,
        warmup_schedule: str = "cosine",
        warmup_start_epoch: int = 0,  # Virtual starting epoch for warmup (for resume from weights)
    ):
        super().__init__()
        self.base_loss = EnergyOODLoss(
            temperature=temperature,
            weight=max_weight,  # This is the MAX weight after warmup
            m_in=m_in,
        )
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_schedule = warmup_schedule
        self.max_weight = max_weight
        self.warmup_start_epoch = warmup_start_epoch  # Virtual starting epoch (e.g., 16 if resuming from epoch 16 weights)
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling."""
        self.current_epoch = epoch
        
    def get_current_weight(self) -> float:
        """
        Compute current energy weight based on warmup schedule.
        
        Uses warmup_start_epoch to account for virtual epoch offset when resuming
        from weights (e.g., if resuming from epoch 16, warmup_start_epoch=16).
        
        Returns:
            Current weight (0.0 during warmup, then gradually increases)
        """
        # Adjust current epoch by warmup_start_epoch offset
        # This allows skipping warmup when resuming from already-trained weights
        adjusted_epoch = self.current_epoch + self.warmup_start_epoch
        
        if adjusted_epoch < self.warmup_epochs:
            # Phase 1: Energy DISABLED (pure OE training)
            return 0.0
        
        # Phase 2: Gradual warmup
        progress = (adjusted_epoch - self.warmup_epochs) / (
            self.max_epochs - self.warmup_epochs
        )
        progress = min(1.0, max(0.0, progress))  # Clamp [0, 1]
        
        if self.warmup_schedule == "cosine":
            # Cosine warmup: smooth increase
            weight = self.max_weight * (1 - math.cos(progress * math.pi)) / 2
        else:  # linear
            weight = self.max_weight * progress
            
        return weight
        
    def forward(
        self,
        class_logits: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Energy loss with warmup.
        
        During warmup (epochs 0-warmup_epochs): returns 0.0
        After warmup: gradually increases energy loss weight
        """
        current_weight = self.get_current_weight()
        
        if current_weight == 0.0:
            # Warmup phase: return zero loss
            return torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        # Safety check: ensure logits are finite before computing loss
        if not torch.isfinite(class_logits).all():
            # If logits contain NaN/Inf, return zero loss to avoid crashing
            return torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        # Compute base energy loss
        energy_loss = self.base_loss(class_logits, target_labels)
        
        # Safety check: ensure loss is finite
        if not torch.isfinite(energy_loss):
            energy_loss = torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        # Scale by current warmup weight
        # base_loss already applies self.base_loss.weight, so we need to adjust
        scale_factor = current_weight / self.max_weight
        
        final_loss = energy_loss * scale_factor
        
        # Final safety check: ensure final loss is finite
        if not torch.isfinite(final_loss):
            final_loss = torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        return final_loss
    
    def get_energy_stats(self, class_logits: torch.Tensor) -> dict:
        """Get energy statistics (delegates to base loss)."""
        stats = self.base_loss.get_energy_stats(class_logits)
        stats["energy_weight_current"] = self.get_current_weight()
        stats["energy_weight_max"] = self.max_weight
        stats["warmup_phase"] = "warmup" if self.current_epoch < self.warmup_epochs else "active"
        return stats
