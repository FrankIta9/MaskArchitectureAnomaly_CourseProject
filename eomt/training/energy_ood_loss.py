# ---------------------------------------------------------------
# Energy-Based Out-of-Distribution Detection Loss
# Based on: "Energy-based Out-of-distribution Detection" (NeurIPS 2020)
# Adapted for Mask2Former-style segmentation with Outlier Exposure
# WITH WARMUP SCHEDULER to avoid conflict with OE in early training
# 
# Task 3: Fixed to use margin-based loss (L_id = ReLU(E_id - m_in), L_ood = ReLU(m_out - E_ood))
# Task 6: Option A - Uses per-pixel semantic logits [B,C,H,W] aligned with inference
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
    - LOW energy scores for in-distribution (ID) pixels
    - HIGH energy scores for out-of-distribution (OOD) pixels
    
    Energy Score: E(x) = -T * log(sum(exp(logits / T)))
    
    Task 3: Margin-based loss:
    - L_id = ReLU(E_id - m_in) (ID must be below m_in)
    - L_ood = ReLU(m_out - E_ood) (OOD must be above m_out)
    - Total: L = L_id + L_ood
    
    Task 6: Uses per-pixel semantic logits [B,C,H,W] (Option A)
    - Aligns training (loss) with inference (anomaly map MSP/MaxLogit)
    
    Args:
        temperature: Temperature scaling for energy computation (default: 1.0)
        weight: Weight for the energy regularization term (default: 0.05)
        m_in: Margin for in-distribution samples (default: -25.0)
        m_out: Margin for out-of-distribution samples (default: -7.0)
        Note: m_out must be > m_in (OOD energy should be higher)
    
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
        
        # Validate that m_out > m_in
        if m_out <= m_in:
            raise ValueError(f"m_out ({m_out}) must be greater than m_in ({m_in}) for proper energy separation")
    
    def update_margins(self, m_in: float, m_out: float):
        """
        Update margins dynamically (Task 4: find real margins from distributions).
        
        Args:
            m_in: New margin for ID samples
            m_out: New margin for OOD samples
        """
        if m_out <= m_in:
            raise ValueError(f"m_out ({m_out}) must be greater than m_in ({m_in}) for proper energy separation")
        self.m_in = m_in
        self.m_out = m_out
    
    def compute_energy_from_pixel_logits(self, pixel_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score from per-pixel semantic logits (DEPRECATED - use compute_energy_from_pixel_scores).
        
        Energy: E(x) = -T * log(sum(exp(logits / T)))
        
        Args:
            pixel_logits: Per-pixel logits tensor [B, C, H, W]
            
        Returns:
            Energy scores [B, H, W]
        """
        # Safety check: ensure logits are real (not complex) and finite
        if pixel_logits.is_complex():
            pixel_logits = pixel_logits.real
        
        if not torch.isfinite(pixel_logits).all():
            # If logits contain NaN/Inf, return safe default energy values
            h, w = pixel_logits.shape[-2:]
            return torch.full((pixel_logits.shape[0], h, w), -25.0, device=pixel_logits.device, dtype=pixel_logits.dtype)
        
        # Force float32 for energy/softmax/logsumexp computations (better numerical stability)
        pixel_logits_f32 = pixel_logits.float() if pixel_logits.dtype != torch.float32 else pixel_logits
        scaled_logits = pixel_logits_f32 / self.temperature  # [B, C, H, W]
        
        # Clamp logits to reasonable range to avoid numerical overflow/underflow
        scaled_logits = torch.clamp(scaled_logits, min=-50.0, max=50.0)
        
        # Ensure logits are real and finite before logsumexp
        if scaled_logits.is_complex():
            scaled_logits = scaled_logits.real
        if not torch.isfinite(scaled_logits).all():
            scaled_logits = torch.where(torch.isfinite(scaled_logits), scaled_logits, torch.zeros_like(scaled_logits))
        
        # Compute logsumexp for numerical stability
        # Energy = -T * logsumexp(logits / T) along class dimension
        energy = -self.temperature * torch.logsumexp(scaled_logits, dim=1)  # [B, H, W]
        
        # Convert back to original dtype if needed
        if pixel_logits.dtype != torch.float32:
            energy = energy.to(pixel_logits.dtype)
        
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
    
    def compute_energy_from_pixel_scores(self, pixel_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score from per-pixel semantic scores (RAW logits, not probabilities).
        
        Energy: E(x) = -T * log(sum(exp(scores / T)))
        
        This version expects RAW logits (not probabilities from softmax), which is correct
        for energy computation as per the standard energy-based OOD detection formula.
        
        Args:
            pixel_scores: Per-pixel scores tensor [B, C, H, W] - RAW logits (not probabilities)
            
        Returns:
            Energy scores [B, H, W]
        """
        # Safety check: ensure scores are real (not complex) and finite
        if pixel_scores.is_complex():
            pixel_scores = pixel_scores.real
        
        if not torch.isfinite(pixel_scores).all():
            # If scores contain NaN/Inf, return safe default energy values
            h, w = pixel_scores.shape[-2:]
            return torch.full((pixel_scores.shape[0], h, w), -25.0, device=pixel_scores.device, dtype=pixel_scores.dtype)
        
        # Force float32 for energy/logsumexp computations (better numerical stability)
        pixel_scores_f32 = pixel_scores.float() if pixel_scores.dtype != torch.float32 else pixel_scores
        
        # Scale by temperature and clamp for numerical stability
        x = torch.clamp(pixel_scores_f32 / self.temperature, min=-50.0, max=50.0)  # [B, C, H, W]
        
        # Ensure scores are real and finite before logsumexp
        if x.is_complex():
            x = x.real
        if not torch.isfinite(x).all():
            x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        
        # Compute energy: E = -T * logsumexp(scores / T) along class dimension
        energy = -self.temperature * torch.logsumexp(x, dim=1)  # [B, H, W]
        
        # Convert back to original dtype if needed
        if pixel_scores.dtype != torch.float32:
            energy = energy.to(pixel_scores.dtype)
        
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
    
    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute energy score for given query-level logits (legacy method for compatibility).
        
        Energy: E(x) = -T * log(sum(exp(logits / T)))
        
        Args:
            logits: Class logits tensor (batch, num_queries, num_classes + 1)
            
        Returns:
            Energy scores (batch, num_queries)
        """
        # Safety check: ensure logits are real (not complex) and finite
        if logits.is_complex():
            logits = logits.real
        
        if not torch.isfinite(logits).all():
            return torch.full((logits.shape[0], logits.shape[1]), -25.0, device=logits.device, dtype=logits.dtype)
        
        logits_f32 = logits.float() if logits.dtype != torch.float32 else logits
        scaled_logits = logits_f32 / self.temperature
        
        # Exclude "no object" class (last class) for ID energy
        id_logits = scaled_logits[..., :-1]  # Remove "no object" class
        
        if id_logits.numel() == 0:
            return torch.zeros(logits.shape[0], logits.shape[1], device=logits.device, dtype=logits.dtype)
        
        id_logits = torch.clamp(id_logits, min=-50.0, max=50.0)
        
        if id_logits.is_complex():
            id_logits = id_logits.real
        if not torch.isfinite(id_logits).all():
            id_logits = torch.where(torch.isfinite(id_logits), id_logits, torch.zeros_like(id_logits))
        
        energy = -self.temperature * torch.logsumexp(id_logits, dim=-1)
        if logits.dtype != torch.float32:
            energy = energy.to(logits.dtype)
        
        if energy.is_complex():
            energy = energy.real
        
        energy = torch.clamp(energy, min=-100.0, max=100.0)
        energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
        
        if not energy.dtype.is_floating_point:
            energy = energy.float()
        
        return energy
    
    def forward(
        self,
        pixel_logits: Optional[torch.Tensor] = None,
        ood_mask: Optional[torch.Tensor] = None,
        class_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Energy-based OOD margin loss.
        
        Task 3: Margin-based loss:
        - L_id = ReLU(E_id - m_in) (penalize ID energy above m_in)
        - L_ood = ReLU(m_out - E_ood) (penalize OOD energy below m_out)
        - Total: L = L_id + L_ood
        
        Task 6: Uses per-pixel semantic logits [B,C,H,W] (Option A - preferred)
        - Aligns training with inference
        
        Args:
            pixel_logits: Per-pixel logits [B, C, H, W] (preferred - Task 6 Option A)
            ood_mask: OOD mask [B, H, W] with values: 0=ID, 1=OOD, 255=ignore (required if pixel_logits provided)
            class_logits: Query-level class logits [B, Q, C+1] (legacy - for backward compatibility)
            
        Returns:
            Energy regularization loss
        """
        # Task 6: Option A - Use per-pixel scores (RAW logits) if available (preferred)
        if pixel_logits is not None and ood_mask is not None:
            # Safety check: ensure scores are finite
            if not torch.isfinite(pixel_logits).all():
                return torch.tensor(0.0, device=pixel_logits.device, dtype=pixel_logits.dtype, requires_grad=True)
            
            # Compute energy map from per-pixel scores (RAW logits, not probabilities)
            energy_map = self.compute_energy_from_pixel_scores(pixel_logits)  # [B, H, W]
            
            # Safety check: ensure energy is finite
            if not torch.isfinite(energy_map).all():
                energy_map = torch.where(torch.isfinite(energy_map), energy_map, torch.full_like(energy_map, -25.0))
            
            # Task 3: Margin-based loss with ID/OOD separation
            batch_size = energy_map.shape[0]
            loss_id_list = []
            loss_ood_list = []
            
            for b in range(batch_size):
                energy_b = energy_map[b]  # [H, W]
                ood_mask_b = ood_mask[b]  # [H, W]
                
                # Filter out ignore pixels (255)
                valid_mask = ood_mask_b != 255
                if not valid_mask.any():
                    continue  # Skip if no valid pixels
                
                energy_valid = energy_b[valid_mask]  # [N]
                ood_mask_valid = ood_mask_b[valid_mask]  # [N]
                
                # Separate ID and OOD
                energy_id = energy_valid[ood_mask_valid == 0]  # ID pixels
                energy_ood = energy_valid[ood_mask_valid == 1]  # OOD pixels
                
                # Task 3: Margin-based loss
                # L_id = ReLU(E_id - m_in): penalize if ID energy > m_in
                if energy_id.numel() > 0:
                    loss_id = F.relu(energy_id - self.m_in).mean()
                    if torch.isfinite(loss_id):
                        loss_id_list.append(loss_id)
                
                # L_ood = ReLU(m_out - E_ood): penalize if OOD energy < m_out
                if energy_ood.numel() > 0:
                    loss_ood = F.relu(self.m_out - energy_ood).mean()
                    if torch.isfinite(loss_ood):
                        loss_ood_list.append(loss_ood)
            
            # Combine losses: L = L_id + L_ood
            if loss_id_list and loss_ood_list:
                loss = torch.stack(loss_id_list).mean() + torch.stack(loss_ood_list).mean()
            elif loss_id_list:
                loss = torch.stack(loss_id_list).mean()
            elif loss_ood_list:
                loss = torch.stack(loss_ood_list).mean()
            else:
                # No valid pixels, return zero loss
                device = pixel_logits.device
                dtype = pixel_logits.dtype
                loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
            
            # Ensure loss is finite (safety check)
            if not torch.isfinite(loss):
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            return loss * self.weight
        
        # Legacy: Use query-level logits (for backward compatibility)
        elif class_logits is not None:
            if not torch.isfinite(class_logits).all():
                return torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
            
            energy = self.compute_energy(class_logits)  # (batch, num_queries)
            
            if not torch.isfinite(energy).all():
                energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
            
            # Legacy: Simple energy regularization (encourage low energy)
            # Note: This doesn't properly separate ID/OOD, but kept for compatibility
            loss = F.relu(energy - self.m_in).mean()
            
            if not torch.isfinite(loss):
                loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)
            
            return loss * self.weight
        
        else:
            raise ValueError("Either (pixel_logits, ood_mask) or class_logits must be provided")
    
    def get_energy_stats(self, class_logits: torch.Tensor) -> dict:
        """
        Get energy statistics for monitoring/debugging (legacy method).
        
        Args:
            class_logits: Class logits (batch_size, num_queries, num_classes + 1)
            
        Returns:
            Dictionary with energy statistics
        """
        with torch.no_grad():
            energy = self.compute_energy(class_logits)
            
            if not torch.isfinite(energy).all():
                energy = torch.where(torch.isfinite(energy), energy, torch.full_like(energy, -25.0))
            
            energy_mean = energy.mean().item() if energy.numel() > 0 else -25.0
            energy_std = energy.std().item() if energy.numel() > 1 and torch.isfinite(energy.std()) else 0.0
            energy_min = energy.min().item() if energy.numel() > 0 and torch.isfinite(energy.min()) else -25.0
            energy_max = energy.max().item() if energy.numel() > 0 and torch.isfinite(energy.max()) else -25.0
            
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
    
    Task 5: Improved warmup schedule
    - Hard off for N epochs (5-10)
    - Then ramp (cosine or linear) to max_weight
    - Supports warmup_start_epoch for resume
    
    Args:
        base_loss: EnergyOODLoss instance (with max weight configured)
        warmup_epochs: Number of epochs with energy disabled (default: 15)
        max_epochs: Total training epochs (for cosine schedule)
        warmup_schedule: "cosine" or "linear" (default: "cosine")
        warmup_start_epoch: Virtual starting epoch for warmup (for resume from weights)
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        max_weight: float = 0.002,  # Conservative max weight
        m_in: float = -25.0,
        m_out: float = -7.0,
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
            m_out=m_out,
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
        
        Task 5: Improved warmup schedule
        - Hard off for warmup_epochs
        - Then ramp (cosine or linear) to max_weight
        - Uses warmup_start_epoch to account for virtual epoch offset when resuming
        
        Returns:
            Current weight (0.0 during warmup, then gradually increases)
        """
        # Adjust current epoch by warmup_start_epoch offset
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
        pixel_logits: Optional[torch.Tensor] = None,
        ood_mask: Optional[torch.Tensor] = None,
        class_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Energy loss with warmup.
        
        During warmup (epochs 0-warmup_epochs): returns 0.0
        After warmup: gradually increases energy loss weight
        
        Args:
            pixel_logits: Per-pixel logits [B, C, H, W] (Task 6 Option A - preferred)
            ood_mask: OOD mask [B, H, W] (required if pixel_logits provided)
            class_logits: Query-level class logits [B, Q, C+1] (legacy)
        """
        current_weight = self.get_current_weight()
        
        if current_weight == 0.0:
            # Warmup phase: return zero loss
            if pixel_logits is not None:
                device = pixel_logits.device
                dtype = pixel_logits.dtype
            elif class_logits is not None:
                device = class_logits.device
                dtype = class_logits.dtype
            else:
                device = torch.device("cpu")
                dtype = torch.float32
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # Safety check: ensure logits are finite before computing loss
        if pixel_logits is not None and not torch.isfinite(pixel_logits).all():
            return torch.tensor(0.0, device=pixel_logits.device, dtype=pixel_logits.dtype, requires_grad=True)
        if class_logits is not None and not torch.isfinite(class_logits).all():
            return torch.tensor(0.0, device=class_logits.device, dtype=class_logits.dtype, requires_grad=True)
        
        # Compute base energy loss
        energy_loss = self.base_loss(
            pixel_logits=pixel_logits,
            ood_mask=ood_mask,
            class_logits=class_logits,
        )
        
        # Safety check: ensure loss is finite
        if not torch.isfinite(energy_loss):
            device = energy_loss.device
            dtype = energy_loss.dtype
            energy_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        # Scale by current warmup weight
        # base_loss already applies self.base_loss.weight, so we need to adjust
        scale_factor = current_weight / self.max_weight
        
        final_loss = energy_loss * scale_factor
        
        # Final safety check: ensure final loss is finite
        if not torch.isfinite(final_loss):
            device = final_loss.device
            dtype = final_loss.dtype
            final_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        return final_loss
    
    def get_energy_stats(self, class_logits: torch.Tensor) -> dict:
        """Get energy statistics (delegates to base loss)."""
        stats = self.base_loss.get_energy_stats(class_logits)
        stats["energy_weight_current"] = self.get_current_weight()
        stats["energy_weight_max"] = self.max_weight
        stats["warmup_phase"] = "warmup" if self.current_epoch < self.warmup_epochs else "active"
        return stats
