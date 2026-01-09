# ---------------------------------------------------------------
# Energy-Based Out-of-Distribution Detection Loss
# Based on: "Energy-based Out-of-distribution Detection" (NeurIPS 2020)
# Adapted for Mask2Former-style segmentation with Outlier Exposure
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
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
