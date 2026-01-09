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
        # Scale logits by temperature
        scaled_logits = logits / self.temperature
        
        # Exclude "no object" class (last class) for ID energy
        # We only consider ID classes for energy computation
        id_logits = scaled_logits[..., :-1]  # Remove "no object" class
        
        # Compute logsumexp for numerical stability
        # Energy = -T * logsumexp(logits / T)
        energy = -self.temperature * torch.logsumexp(id_logits, dim=-1)
        
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
        # Compute energy scores for all queries
        energy = self.compute_energy(class_logits)  # (batch, num_queries)
        
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
            return {
                "energy_mean": energy.mean().item(),
                "energy_std": energy.std().item(),
                "energy_min": energy.min().item(),
                "energy_max": energy.max().item(),
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
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update current epoch for warmup scheduling."""
        self.current_epoch = epoch
        
    def get_current_weight(self) -> float:
        """
        Compute current energy weight based on warmup schedule.
        
        Returns:
            Current weight (0.0 during warmup, then gradually increases)
        """
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: Energy DISABLED (pure OE training)
            return 0.0
        
        # Phase 2: Gradual warmup
        progress = (self.current_epoch - self.warmup_epochs) / (
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
            return torch.tensor(0.0, device=class_logits.device)
        
        # Compute base energy loss
        energy_loss = self.base_loss(class_logits, target_labels)
        
        # Scale by current warmup weight
        # base_loss already applies self.base_loss.weight, so we need to adjust
        scale_factor = current_weight / self.max_weight
        
        return energy_loss * scale_factor
    
    def get_energy_stats(self, class_logits: torch.Tensor) -> dict:
        """Get energy statistics (delegates to base loss)."""
        stats = self.base_loss.get_energy_stats(class_logits)
        stats["energy_weight_current"] = self.get_current_weight()
        stats["energy_weight_max"] = self.max_weight
        stats["warmup_phase"] = "warmup" if self.current_epoch < self.warmup_epochs else "active"
        return stats
