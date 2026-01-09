# ---------------------------------------------------------------
# Enhanced Isotropy Maximization Loss for Anomaly Segmentation
# Based on: "Enhanced Isotropy Maximization Loss for Out-of-Distribution Detection"
# This loss function improves OoD detection by encouraging isotropic representations
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EnhancedIsotropyLoss(nn.Module):
    """
    Enhanced Isotropy Maximization Loss for anomaly segmentation.
    
    This loss encourages the model to produce more isotropic (uniformly distributed)
    representations, which helps distinguish between in-distribution and 
    out-of-distribution samples.
    
    The loss is computed as:
    L_EIM = -log(sum(exp(logits / tau)) / (num_classes * max(exp(logits / tau))))
    where tau is a temperature parameter.
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        weight: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Args:
            temperature: Temperature parameter for logit scaling (default: 1.0)
            weight: Weight for the EIM loss component (default: 0.1)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight
        self.reduction = reduction
        
    def forward(
        self,
        class_logits: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Enhanced Isotropy Maximization Loss.
        
        Args:
            class_logits: Class logits tensor of shape (batch_size, num_queries, num_classes + 1)
            target_labels: Optional target labels for supervised component
            
        Returns:
            EIM loss value
        """
        # Scale logits by temperature
        scaled_logits = class_logits / self.temperature
        
        # Enhanced Isotropy Maximization Loss formula:
        # L_EIM = -log(sum(exp(logits/tau)) / (num_classes * max(exp(logits/tau))))
        # This encourages more uniform (isotropic) distributions
        
        # Compute exponential of scaled logits
        exp_logits = torch.exp(scaled_logits)
        
        # Compute maximum exponential value per query
        max_exp = exp_logits.max(dim=-1, keepdim=True)[0]
        
        # Compute sum of exponentials
        sum_exp = exp_logits.sum(dim=-1, keepdim=True)
        
        # Avoid division by zero
        max_exp = torch.clamp(max_exp, min=1e-8)
        
        # Compute isotropy score: sum(exp) / (num_classes * max(exp))
        # Higher score = more isotropic (uniform) distribution
        num_classes = class_logits.shape[-1]
        isotropy_score = sum_exp / (num_classes * max_exp)
        
        # Clamp to avoid numerical issues
        isotropy_score = torch.clamp(isotropy_score, min=1e-8, max=1.0)
        
        # Compute loss as negative log of isotropy score
        # We want to maximize isotropy, so minimize negative log
        loss = -torch.log(isotropy_score)
        
        # Apply weight
        loss = loss * self.weight
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
