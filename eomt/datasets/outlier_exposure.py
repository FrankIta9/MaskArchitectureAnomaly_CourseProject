# ---------------------------------------------------------------
# Outlier Exposure with Cut-Paste Augmentation
# Implements cut-paste augmentation using COCO objects on Cityscapes images
# Based on: "Cut-Paste: A Simple Data Augmentation Strategy for Outlier Detection"
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision.tv_tensors import Image, Mask
from typing import Optional, Dict, Any, Tuple
import random
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.v2.functional as F
from pathlib import Path
import json
import zipfile
from io import BytesIO

try:
    from pycocotools.coco import COCO
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False
    print("Warning: pycocotools not available. Install with: pip install pycocotools")

# Mapping COCO category names → Cityscapes trainId (for ID augmentation)
# Categories NOT in this mapping are treated as OOD (semseg=255, ood_mask=1)
# Cityscapes trainId: road 0, sidewalk 1, building 2, wall 3, fence 4, pole 5,
#                     traffic light 6, traffic sign 7, vegetation 8, terrain 9, sky 10,
#                     person 11, rider 12, car 13, truck 14, bus 15, train 16,
#                     motorcycle 17, bicycle 18
COCO_TO_CS_TRAINID = {
    "person": 11,
    "bicycle": 18,
    "car": 13,
    "motorcycle": 17,
    "bus": 15,
    "truck": 14,
    "train": 16,
    "traffic light": 6,
    "stop sign": 7,  # treat as traffic sign
}


class OutlierExposureTransform(nn.Module):
    """
    Outlier Exposure transformation using cut-paste augmentation.
    
    This transform randomly pastes objects from an outlier dataset (e.g., COCO)
    onto Cityscapes images to create synthetic anomaly examples.
    """
    
    def __init__(
        self,
        outlier_dataset: Optional[Any] = None,
        paste_probability: float = 0.5,  # DEPRECATED: use p_id_paste and p_ood_paste instead
        p_id_paste: float = 0.10,  # Probability of ID paste (mappable COCO → Cityscapes)
        p_ood_paste: float = 0.30,  # Probability of OOD paste (non-mappable COCO)
        min_objects: int = 1,
        max_objects: int = 3,
        min_scale: float = 0.1,
        max_scale: float = 0.3,
        blend_alpha: float = 1.0,  # 1.0 = dry paste (more stable), 0.8 for blending
        max_overlap_ratio: float = 0.02,  # Max overlap with existing GT (0.02 = 2%)
        # Multi-scale weighted distribution (for better matching with small anomalies)
        use_weighted_scale: bool = False,
        scale_ranges: Optional[list] = None,  # [(min1, max1), (min2, max2), ...]
        scale_weights: Optional[list] = None,  # [weight1, weight2, ...] (sum should be 1.0)
        # Perspective-aware placement (inspired by ClimaOoD)
        use_perspective_aware: bool = True,
        perspective_strength: float = 1.0,  # 0.0 = disabled, 1.0 = full effect
        # Drivable region constraints (inspired by ClimaOoD)
        use_drivable_regions: bool = True,
        drivable_class_ids: Optional[list] = None,  # [0, 1] for road, sidewalk in Cityscapes
        # P0 Fix: Y position range for paste (biased towards bottom = on-road)
        paste_y_range: Tuple[float, float] = (0.65, 0.98),  # (min_ratio, max_ratio) of image height
        # Min object size in pixels (to reduce resample attempts)
        min_obj_size_px: int = 30,  # Minimum object size in pixels (width or height)
    ):
        """
        Args:
            outlier_dataset: Dataset containing outlier objects (e.g., COCO)
            paste_probability: Probability of applying cut-paste (default: 0.5)
            min_objects: Minimum number of objects to paste (default: 1)
            max_objects: Maximum number of objects to paste (default: 3)
            min_scale: Minimum scale factor for pasted objects (default: 0.1)
            max_scale: Maximum scale factor for pasted objects (default: 0.3)
            blend_alpha: Alpha blending factor for pasted objects (default: 0.8)
            use_weighted_scale: If True, use weighted multi-scale distribution instead of uniform (default: False)
            scale_ranges: List of (min, max) scale ranges for each category (default: None)
            scale_weights: List of weights for each scale range (should sum to 1.0, default: None)
                          Example: scale_weights=[0.6, 0.3, 0.1] means 60% small, 30% medium, 10% large
            use_perspective_aware: If True, apply perspective-aware scaling (objects in lower Y = larger) (default: True)
            perspective_strength: Strength of perspective effect (0.0 = disabled, 1.0 = full effect) (default: 1.0)
            use_drivable_regions: If True, only place objects on drivable regions (road/sidewalk) (default: True)
            drivable_class_ids: List of train_id class IDs for drivable regions (default: [0, 1] for Cityscapes)
            paste_y_range: Tuple (min_ratio, max_ratio) of image height for Y position (default: (0.65, 0.98))
                           Ensures objects are placed in lower part of image (on-road proxy)
            min_obj_size_px: Minimum object size in pixels (width or height) to reduce resample attempts (default: 30)
        """
        super().__init__()
        self.outlier_dataset = outlier_dataset
        # Use separate probabilities for ID and OOD paste
        self.p_id_paste = p_id_paste if p_id_paste is not None else (paste_probability * 0.2)  # Fallback for backward compat
        self.p_ood_paste = p_ood_paste if p_ood_paste is not None else (paste_probability * 0.8)  # Fallback for backward compat
        self.paste_probability = paste_probability  # Keep for backward compatibility
        self.max_overlap_ratio = max_overlap_ratio
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blend_alpha = float(blend_alpha)
        self.use_weighted_scale = use_weighted_scale
        
        # Multi-scale weighted distribution
        if use_weighted_scale:
            if scale_ranges is None or scale_weights is None:
                raise ValueError("scale_ranges and scale_weights must be provided when use_weighted_scale=True")
            if len(scale_ranges) != len(scale_weights):
                raise ValueError("scale_ranges and scale_weights must have the same length")
            if abs(sum(scale_weights) - 1.0) > 1e-6:
                raise ValueError(f"scale_weights must sum to 1.0 (current sum: {sum(scale_weights)})")
            self.scale_ranges = scale_ranges
            self.scale_weights = scale_weights
        else:
            self.scale_ranges = None
            self.scale_weights = None
        
        # Perspective-aware placement (inspired by ClimaOoD)
        self.use_perspective_aware = use_perspective_aware
        self.perspective_strength = perspective_strength
        
        # Drivable region constraints (inspired by ClimaOoD)
        self.use_drivable_regions = use_drivable_regions
        self.drivable_class_ids = drivable_class_ids if drivable_class_ids is not None else [0, 1]  # Road=0, Sidewalk=1 in Cityscapes
        
        # P0 Fix: Y position range for paste (biased towards bottom = on-road)
        self.paste_y_range = paste_y_range
        
        # Min object size in pixels (to reduce resample attempts)
        self.min_obj_size_px = min_obj_size_px
        
        # Task 5: Placement counters for debugging (indicative, not critical)
        # Note: These are reset at epoch start (not thread-safe with multi-worker)
        self.drivable_placement_count = 0
        self.random_placement_count = 0
        
        # FIX 4: Debug counter for logging (first 2 batches only, ~4-6 samples)
        self._dbg_count = 0
        self._dbg_max_count = 6  # Limit to first ~2 batches (batch_size=2-3)
        self._resample_warn_logged = False  # Disabled: resample is silent (no logging)
    
    def _get_random_outlier_object(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get a random object from the outlier dataset.
        
        Returns:
            Tuple of (object_image, object_mask, category_name)
        """
        if self.outlier_dataset is None or len(self.outlier_dataset) == 0:
            # Return a dummy object if no outlier dataset is provided
            dummy_img = torch.zeros((3, 64, 64))
            dummy_mask = torch.ones((64, 64), dtype=torch.bool)
            return dummy_img, dummy_mask, "unknown"
            
        idx = random.randint(0, len(self.outlier_dataset) - 1)
        return self.outlier_dataset[idx]
    
    def _paste_object(
        self,
        img: torch.Tensor,
        target: Dict[str, Any],
        obj_img: torch.Tensor,
        obj_mask: torch.Tensor,
        position: Tuple[int, int],
        scale: float,
        category_name: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, bool]:
        """
        Paste an object onto the image at the given position.
        Implements hybrid strategy: ID augmentation (mappable) vs OOD augmentation (non-mappable).
        
        Args:
            img: Input image tensor
            target: Target dictionary with masks and labels
            obj_img: Object image to paste
            obj_mask: Object mask
            position: (x, y) position to paste
            scale: Scale factor for the object
            category_name: COCO category name (e.g., "person", "car", "chair")
            
        Returns:
            Modified image, target, binary paste_mask (H, W), and success flag (False if skipped due to overlap)
        """
        h, w = img.shape[-2:]
        obj_h, obj_w = obj_img.shape[-2:]
        
        # Initialize paste_mask as zeros (no paste yet)
        paste_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
        
        # Resize object based on scale
        new_h = int(obj_h * scale)
        new_w = int(obj_w * scale)
        new_h = min(new_h, h)
        new_w = min(new_w, w)
        
        if new_h > 0 and new_w > 0:
            obj_img_resized = F.resize(obj_img, (new_h, new_w), antialias=True)
            obj_mask_resized = F.resize(
                obj_mask.float().unsqueeze(0), 
                (new_h, new_w), 
                interpolation=F.InterpolationMode.NEAREST
            ).squeeze(0).bool()
            
            # Calculate paste region
            x, y = position
            x = max(0, min(x, w - new_w))
            y = max(0, min(y, h - new_h))
            x_end = x + new_w
            y_end = y + new_h
            
            # DEBUG: Assert shape coerenti (solo primi 200 batch)
            if not hasattr(self, '_debug_shape_count'):
                self._debug_shape_count = 0
            if self._debug_shape_count < 200:
                expected_shape = (y_end - y, x_end - x)
                assert obj_mask_resized.shape == expected_shape, \
                    f"obj_mask_resized shape mismatch: got {obj_mask_resized.shape}, expected {expected_shape}"
                self._debug_shape_count += 1
            
            # OVERLAP POLICY: Check overlap with existing GT before pasting
            # (Safety check, ma con FIX 1 diventa quasi superfluo)
            if "semseg" in target:
                semseg = target["semseg"]  # [H, W] LongTensor
                # Get object region in full image coordinates
                obj_region_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
                obj_region_mask[y:y_end, x:x_end] = obj_mask_resized
                
                # Check overlap with existing valid GT (semseg != 255)
                existing_gt = (semseg != 255)
                overlap = obj_region_mask & existing_gt
                overlap_ratio = overlap.sum().float() / max(1, obj_region_mask.sum().float())
                
                # Skip if overlap exceeds threshold (to avoid corrupting labels)
                # Con FIX 1 questo diventa quasi superfluo, ma mantienilo come safety
                if overlap_ratio > self.max_overlap_ratio:
                    return img, target, paste_mask, False  # Skip this object
            
            # Clone image and target for modification
            img_clone = img.clone()
            target = target.copy()  # Shallow copy to avoid modifying original
            if "semseg" in target:
                target["semseg"] = target["semseg"].clone()
            
            # Task 4: Light feathering only (kernel 3-5) - removed color matching, occlusion, shadow
            # Feathered alpha blending: apply light Gaussian blur to mask edges
            mask_float = obj_mask_resized.float()
            mask_blurred = mask_float.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            kernel_size = 3  # Light feathering (kernel 3-5, using 3 for subtlety)
            padding = kernel_size // 2
            
            # Check dimensions to ensure padding is valid
            h_mask, w_mask = mask_blurred.shape[-2:]
            if h_mask >= kernel_size and w_mask >= kernel_size:
                # Use avg_pool2d as a simple blur approximation (only if dimensions are large enough)
                mask_blurred = torch.nn.functional.avg_pool2d(
                    torch.nn.functional.pad(mask_blurred, (padding, padding, padding, padding), mode='reflect'),
                    kernel_size=kernel_size, stride=1
                )
            # If dimensions are too small, skip feathering (use original mask)
            mask_blurred = mask_blurred.squeeze(0).squeeze(0)  # [H, W]
            
            # Simple alpha blending with feathered mask (no color matching, no shadow, no occlusion)
            bg_patch = img_clone[:, y:y_end, x:x_end]  # [3, H, W]
            for c in range(3):
                blended = (
                    self.blend_alpha * obj_img_resized[c] * mask_blurred + 
                    (1 - self.blend_alpha * mask_blurred) * bg_patch[c]
                )
                img_clone[c, y:y_end, x:x_end] = blended
            
            # HYBRID STRATEGY: ID augmentation vs OOD augmentation
            # FIX 1: Scrittura SOLO su pixel già IGNORE (semseg==255) + valid_mask
            # FIX 2: Elimina doppio indexing usando patch locale
            # FIX 3: Verifica paste_mask sia solo oggetto (non bbox)
            
            if "semseg" in target:
                # Extract patch
                semseg_patch = target["semseg"][y:y_end, x:x_end].clone()  # [H_patch, W_patch]
                
                # FIX 1: Write mask = solo pixel IGNORE (255) + valid_mask se presente
                write_mask = obj_mask_resized & (semseg_patch == 255)
                
                # Opzionale: se hai valid_mask, ancora meglio
                if "valid_mask" in target:
                    vm_patch = target["valid_mask"][y:y_end, x:x_end].to(torch.bool)
                    write_mask = write_mask & vm_patch
                
                # FIX 2: Usa patch locale invece di doppio indexing
                if category_name in COCO_TO_CS_TRAINID:
                    # ID AUGMENTATION: Mappable category → supervised paste
                    cs_trainid = COCO_TO_CS_TRAINID[category_name]
                    # Scrivi solo dove semseg era 255 (IGNORE)
                    semseg_patch[write_mask] = cs_trainid
                    target["semseg"][y:y_end, x:x_end] = semseg_patch
                    # ood_mask stays 0 (not set to 1)
                else:
                    # OOD AUGMENTATION: Non-mappable category → outlier exposure
                    # Scrivi 255 solo dove era già 255 (di fatto non cambia, ma mantiene invariante)
                    semseg_patch[write_mask] = 255
                    target["semseg"][y:y_end, x:x_end] = semseg_patch
                    # ood_mask will be set to 1 in forward() based on paste_mask
            
            # FIX 3: paste_mask = SOLO dove oggetto è 1 (non bbox piena)
            # Assicurati che paste_mask sia esattamente obj_mask_resized alle coordinate corrette
            paste_mask[y:y_end, x:x_end] = paste_mask[y:y_end, x:x_end] | obj_mask_resized
            
            return img_clone, target, paste_mask, True  # Success
        
        return img, target, paste_mask, False  # Failed (invalid size)
    
    def forward(
        self,
        img: torch.Tensor,
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply outlier exposure transformation with perspective-aware placement and drivable region constraints.
        
        Args:
            img: Input image tensor
            target: Target dictionary with masks and labels
            
        Returns:
            Transformed image and target (with ood_mask added)
        """
        h, w = img.shape[-2:]
        
        # Initialize ood_mask: 0 = ID, 1 = OOD, 255 = ignore (opzionale)
        ood_mask = torch.zeros((h, w), dtype=torch.uint8, device=img.device)
        
        # HYBRID STRATEGY: Separate probabilities for ID paste (mappable) vs OOD paste (non-mappable)
        rand_id = random.random()
        rand_ood = random.random()
        do_id_paste = rand_id < self.p_id_paste
        do_ood_paste = rand_ood < self.p_ood_paste
        
        if not (do_id_paste or do_ood_paste):
            # No paste: all pixels are ID (0)
            target["ood_mask"] = ood_mask
            return img, target
        
        # FIX 4: Hard-min size check with resample (max 10 attempts)
        OOD_RATIO_MIN = 0.002
        MAX_RESAMPLE = 10
        
        num_objects = random.randint(self.min_objects, self.max_objects)
        
        # Get drivable mask once for all objects
        drivable_mask = self._get_drivable_mask(target, h, w)
        
        # Fix: Escludi pixel padding/letterbox e imponi y >= 0.7H
        # Questo evita OOD "nel vuoto" nero e oggetti "in cielo"
        y_min = int(0.7 * h)
        
        if drivable_mask is not None:
            # valid area: preferisci target["valid_mask"] se c'è
            if "valid_mask" in target:
                valid_area = target["valid_mask"]
            elif "semseg" in target:
                valid_area = (target["semseg"] != 255)
            else:
                valid_area = None

            if valid_area is not None:
                if valid_area.shape != drivable_mask.shape:
                    valid_area_rs = valid_area.to(torch.float32)[None, None, ...]
                    valid_area_rs = F.interpolate(valid_area_rs, size=drivable_mask.shape[-2:], mode="nearest")
                    valid_area = valid_area_rs[0, 0].to(torch.bool)

                drivable_mask = drivable_mask & valid_area

            # NEW: bottom constraint - y >= 0.7H
            drivable_mask[:y_min, :] = False

            if not drivable_mask.any():
                drivable_mask = None  # Nessuna posizione valida
        
        # Accumulate paste_mask from all pasted objects
        cumulative_paste_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
        
        # FIX 4: Resample loop if ood_ratio too small
        for resample_attempt in range(MAX_RESAMPLE):
            # Reset cumulative mask for resample
            if resample_attempt > 0:
                cumulative_paste_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
                img = img_original.clone()  # Restore original image for resample
            
            img_original = img.clone()  # Keep original image for resample
            
            for obj_idx in range(num_objects):
                # Decide paste type for this object (ID or OOD)
                # If both are enabled, alternate or use probability
                if do_id_paste and do_ood_paste:
                    # Alternate or random: 50% ID, 50% OOD
                    is_id_paste = (obj_idx % 2 == 0) if random.random() < 0.5 else random.random() < 0.5
                elif do_id_paste:
                    is_id_paste = True
                elif do_ood_paste:
                    is_id_paste = False
                else:
                    continue  # Should not happen, but safety check
                
                # Get random outlier object (with category_name)
                obj_img, obj_mask, category_name = self._get_random_outlier_object()
                obj_h_orig, obj_w_orig = obj_img.shape[-2:]
                
                # For ID paste: only use mappable categories
                # For OOD paste: only use non-mappable categories
                if is_id_paste:
                    if category_name not in COCO_TO_CS_TRAINID:
                        # Skip: this category is not mappable, try another object
                        max_retries = 10
                        retry_count = 0
                        while retry_count < max_retries:
                            obj_img, obj_mask, category_name = self._get_random_outlier_object()
                            if category_name in COCO_TO_CS_TRAINID:
                                break
                            retry_count += 1
                        if retry_count >= max_retries:
                            continue  # Skip this object if no mappable category found
                else:
                    if category_name in COCO_TO_CS_TRAINID:
                        # Skip: this category is mappable, try another object
                        max_retries = 10
                        retry_count = 0
                        while retry_count < max_retries:
                            obj_img, obj_mask, category_name = self._get_random_outlier_object()
                            if category_name not in COCO_TO_CS_TRAINID:
                                break
                            retry_count += 1
                        if retry_count >= max_retries:
                            continue  # Skip this object if no non-mappable category found
                
                # Select base scale using weighted distribution if enabled, otherwise uniform
                if self.use_weighted_scale:
                    base_scale = self._sample_weighted_scale()
                else:
                    base_scale = random.uniform(self.min_scale, self.max_scale)
                
                # P0 Fix: Ordine ottimizzato - se drivable_mask disponibile, scegli (x,y) PRIMA, poi calcola scale
                # Se drivable_mask non disponibile, usa ordine classico (y -> scale -> x)
                if drivable_mask is not None and drivable_mask.any():
                    # ORDINE NUOVO: prima scegli (x,y) drivable, poi calcola scale basato su y
                    # Usa dimensioni iniziali stimate per trovare posizione valida
                    obj_h_est = max(1, int(obj_h_orig * base_scale))
                    obj_w_est = max(1, int(obj_w_orig * base_scale))
                    obj_h_est = min(obj_h_est, h)
                    obj_w_est = min(obj_w_est, w)
                    
                    # Prova a trovare posizione drivable
                    position = self._sample_drivable_position(drivable_mask, obj_h_est, obj_w_est, h, w)
                    if position is not None:
                        x, y = position
                        self.drivable_placement_count += 1
                        
                        # Ora calcola scale basato su y scelto
                        scale = self._apply_perspective_aware_scale(base_scale, y, h)
                        scale = max(self.min_scale, min(self.max_scale * 2.0, scale))  # Allow up to 2.0x for perspective
                        
                        # Ricalcola dimensioni con scale finale
                        obj_h_scaled = max(1, int(obj_h_orig * scale))
                        obj_w_scaled = max(1, int(obj_w_orig * scale))
                        obj_h_scaled = min(obj_h_scaled, h)
                        obj_w_scaled = min(obj_w_scaled, w)
                        
                        # Fix 4: Validare che dopo ricalcolo dimensioni, posizione sia ancora valida
                        y_end = min(y + obj_h_scaled, h)
                        x_end = min(x + obj_w_scaled, w)
                        region = drivable_mask[y:y_end, x:x_end]
                        if region.numel() == 0 or (region.float().mean().item() < 0.6):
                            # Riprova con dimensioni finali
                            position2 = self._sample_drivable_position(drivable_mask, obj_h_scaled, obj_w_scaled, h, w)
                            if position2 is not None:
                                x, y = position2
                            else:
                                # Fallback safe: random solo in valid+bottom
                                y_min = int(0.7 * h)
                                x, y = self._fallback_safe_position(target, obj_h_scaled, obj_w_scaled, h, w, y_min)
                                if x is None:
                                    continue  # Skip this object
                        else:
                            # Verifica che l'oggetto ci stia ancora nella posizione scelta
                            if x + obj_w_scaled > w or y + obj_h_scaled > h:
                                # Se non ci sta, clamp x,y
                                x = min(x, max(0, w - obj_w_scaled))
                                y = min(y, max(0, h - obj_h_scaled))
                    else:
                        # Fallback safe: random solo in valid+bottom (non continue aggressivo)
                        y_min = int(0.7 * h)
                        x, y = self._fallback_safe_position(target, obj_h_est, obj_w_est, h, w, y_min)
                        if x is None:
                            continue  # Skip this object if no valid position
                else:
                    # ORDINE CLASSICO: y -> scale -> x (quando non c'è drivable_mask)
                    # Fix: Assicura che anche qui non finisca nel padding
                    y_min_ratio, y_max_ratio = self.paste_y_range
                    y_min = int(y_min_ratio * h)
                    y_max = int(y_max_ratio * h)
                    y = random.randint(y_min, y_max)
                    
                    scale = self._apply_perspective_aware_scale(base_scale, y, h)
                    scale = max(self.min_scale, min(self.max_scale * 2.0, scale))
                    
                    obj_h_scaled = max(1, int(obj_h_orig * scale))
                    obj_w_scaled = max(1, int(obj_w_orig * scale))
                    obj_h_scaled = min(obj_h_scaled, h)
                    obj_w_scaled = min(obj_w_scaled, w)
                    
                    # Fix: Usa fallback_safe_position anche qui per evitare padding
                    x, y = self._fallback_safe_position(target, obj_h_scaled, obj_w_scaled, h, w, y_min)
                    if x is None:
                        continue  # Skip this object if no valid position
                    self.random_placement_count += 1
                
                # Paste object and get paste_mask (with category_name for ID/OOD distinction)
                img, target, paste_mask, success = self._paste_object(
                    img, target, obj_img, obj_mask, (x, y), scale, category_name
                )
                
                # Skip if paste failed (e.g., overlap too high)
                if not success:
                    continue
                
                # LOG 2: Debug logging for first 2 batches only (silent after)
                if self._dbg_count < self._dbg_max_count:
                    ood_ratio_obj = paste_mask.float().mean().item() if paste_mask.numel() > 0 else 0.0
                    y_norm = float(y) / max(1.0, float(h - 1))
                    paste_type = "ID" if category_name in COCO_TO_CS_TRAINID else "OOD"
                    print(f"[OE Debug {self._dbg_count}] {paste_type} paste: cat={category_name}, y={y} (norm={y_norm:.3f}), base_scale={base_scale:.4f}, final_scale={scale:.4f}, obj_h={obj_h_scaled}, obj_w={obj_w_scaled}, ood_ratio={ood_ratio_obj:.6f}")
                    self._dbg_count += 1
                
                # Accumulate paste_mask (only for OOD, ID paste doesn't set ood_mask)
                # For OOD paste: paste_mask indicates OOD pixels
                # For ID paste: paste_mask is still tracked but ood_mask stays 0
                if category_name not in COCO_TO_CS_TRAINID:
                    # Only accumulate OOD paste_mask
                    cumulative_paste_mask = cumulative_paste_mask | paste_mask
            
            # FIX 4: Check ood_ratio and min object size after all objects pasted
            ood_ratio = cumulative_paste_mask.float().mean().item()
            
            # Config finale: cap "paste troppo grande" - ood_ratio_max ridotto per non distruggere ID
            # Silent resample: no logging to avoid performance impact
            OOD_RATIO_MAX = 0.03  # Ridotto da 0.05 a 0.03 per meno aggressività (0.02-0.03 range)
            if ood_ratio > OOD_RATIO_MAX:
                # Paste troppo grande, resample (silent)
                if resample_attempt < MAX_RESAMPLE - 1:
                    continue  # Resample
                else:
                    # After max attempts, accept silently
                    break
            
            # Check anche min object size (per ridurre resample)
            ood_pixels = cumulative_paste_mask.sum().item()
            min_obj_size_ok = ood_pixels >= (self.min_obj_size_px ** 2)  # Area minima
            
            if ood_ratio >= OOD_RATIO_MIN and min_obj_size_ok:
                # Accept this paste
                break
            else:
                # Silent resample: no logging to avoid performance impact
                # Restore original image for resample
                if resample_attempt < MAX_RESAMPLE - 1:
                    img = img_original
                # After max attempts, accept silently (no logging)
        
        # Task 1: Build ood_mask: 1 = OOD (non-mappable pasted), 0 = ID (rest)
        # Note: ID paste (mappable) does NOT set ood_mask to 1 (it stays 0)
        # Only OOD paste (non-mappable) sets ood_mask to 1
        ood_mask = cumulative_paste_mask.to(torch.uint8)  # 1 = OOD, 0 = ID
        
        # Note: semseg is already updated in _paste_object:
        # - ID paste: semseg set to Cityscapes trainId (not 255) in _paste_object
        # - OOD paste: semseg set to 255 (IGNORE) in _paste_object
        # No need to set IGNORE again here
        
        # Add ood_mask to target
        target["ood_mask"] = ood_mask
        
        # DEBUG: Assert/warn per invarianti (solo primi 200 batch)
        if not hasattr(self, '_debug_batch_count'):
            self._debug_batch_count = 0
        
        if self._debug_batch_count < 200:
            # 1. Semseg dtype assert
            if "semseg" in target:
                assert target["semseg"].dtype in (torch.int64, torch.long), \
                    f"semseg dtype must be int64/long, got {target['semseg'].dtype}"
            
            # 2. Invariant ID/OOD
            if "semseg" in target and "ood_mask" in target:
                semseg = target["semseg"]
                ood_mask_bool = (ood_mask == 1)
                
                # ID-paste invariant: ood_mask==1 NON deve avere semseg in trainIds mappati
                mappable_trainids = set(COCO_TO_CS_TRAINID.values())
                ood_semseg = semseg[ood_mask_bool]
                if ood_semseg.numel() > 0:
                    ood_has_mappable = (ood_semseg.unsqueeze(0) == torch.tensor(list(mappable_trainids), device=ood_semseg.device).unsqueeze(1)).any(dim=0).any()
                    if ood_has_mappable:
                        import warnings
                        warnings.warn(
                            f"⚠️ Invariant violation: ood_mask==1 has mappable trainIds! "
                            f"Unique values in OOD region: {torch.unique(ood_semseg).tolist()}"
                        )
                
                # OOD-paste invariant: ood_mask==1 deve avere semseg==255
                ood_not_ignore = (ood_mask_bool & (semseg != 255)).sum().item()
                if ood_not_ignore > 0:
                    import warnings
                    warnings.warn(
                        f"⚠️ Invariant violation: {ood_not_ignore} OOD pixels have semseg != 255!"
                    )
        
        self._debug_batch_count += 1
        
        return img, target
    
    def _sample_weighted_scale(self) -> float:
        """
        Sample scale from weighted multi-scale distribution.
        
        Returns:
            Scale value sampled from the weighted distribution
        """
        # Select scale range based on weights
        selected_range_idx = random.choices(
            range(len(self.scale_ranges)),
            weights=self.scale_weights,
            k=1
        )[0]
        
        # Sample uniformly within the selected range
        min_scale, max_scale = self.scale_ranges[selected_range_idx]
        scale = random.uniform(min_scale, max_scale)
        
        return scale
    
    def _apply_perspective_aware_scale(self, base_scale: float, y: int, h: int) -> float:
        """
        Apply perspective-aware scaling based on vertical position (inspired by ClimaOoD).
        
        Objects in lower Y positions (closer to camera) are scaled larger.
        Objects in higher Y positions (farther from camera) are scaled smaller.
        
        Args:
            base_scale: Base scale factor from weighted/uniform distribution
            y: Vertical position (pixel coordinate, 0 = top, h = bottom)
            h: Image height
            
        Returns:
            Adjusted scale factor considering perspective
        """
        if not self.use_perspective_aware or self.perspective_strength <= 0.0:
            return base_scale

        # t in [0,1]: 0=top (far), 1=bottom (near)
        t = float(y) / max(1.0, float(h - 1))

        # Fishyscapes-like: più aggressivo per oggetti più credibili
        min_factor = 0.60   # far -> smaller
        max_factor = 2.00   # near -> larger (più aggressivo)
        perspective_factor = min_factor + (max_factor - min_factor) * t

        # apply strength
        adjusted = 1.0 + (perspective_factor - 1.0) * self.perspective_strength
        return base_scale * adjusted
    
    def _get_drivable_mask(self, target: Dict[str, Any], h: int, w: int) -> Optional[torch.Tensor]:
        """
        Extract drivable region mask from target semantic masks.
        
        Args:
            target: Target dictionary with masks and labels
            h: Image height
            w: Image width
            
        Returns:
            Binary mask of drivable regions (road + sidewalk) or None if not available
        """
        if not self.use_drivable_regions:
            return None
        
        if "masks" not in target or "labels" not in target:
            return None
        
        masks = target["masks"]  # Shape: (num_classes, H, W)
        labels = target["labels"]  # Shape: (num_classes,)
        
        if masks.shape[0] == 0 or len(labels) == 0:
            return None
        
        # Safety check: ensure masks and labels have same length
        num_masks = min(masks.shape[0], len(labels))
        
        # Find masks for drivable classes (road=0, sidewalk=1)
        drivable_mask = torch.zeros((h, w), dtype=torch.bool, device=masks.device)
        
        # Fix: iterate over len(labels), not num_classes constant
        for i in range(len(labels)):
            if i >= num_masks:
                break  # Safety: don't access masks beyond available
            label_id = labels[i]
            if label_id.item() in self.drivable_class_ids:
                # Combine masks: road OR sidewalk
                mask_class = masks[i]  # Shape: (H, W) or (1, H, W) depending on how it's stored
                
                # Handle different mask shapes
                if mask_class.dim() == 3:
                    mask_class = mask_class.squeeze(0)  # Remove batch dimension if present
                
                if mask_class.shape == (h, w):
                    drivable_mask = drivable_mask | mask_class.bool()
                elif mask_class.numel() > 0:
                    # Handle resized masks
                    mask_resized = F.resize(
                        mask_class.unsqueeze(0).float() if mask_class.dim() == 2 else mask_class.float(),
                        (h, w),
                        interpolation=F.InterpolationMode.NEAREST
                    )
                    if mask_resized.dim() == 3:
                        mask_resized = mask_resized.squeeze(0)
                    drivable_mask = drivable_mask | mask_resized.bool()
        
        # Task 5: Erode drivable mask with smaller kernel (kernel_size=5) to avoid edge placements
        if drivable_mask.any():
            # Use max_pool2d with kernel_size=5 and stride=1 for erosion effect
            # Erosion: erode(mask) = ~dilate(~mask), or use max_pool2d on inverted mask
            mask_float = drivable_mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            # Erosion: min pool on inverted mask, then invert back
            inverted_mask = 1.0 - mask_float
            kernel_size = 5  # Task 5: Reduced from 9 to 5
            padding = kernel_size // 2
            eroded_inverted = torch.nn.functional.max_pool2d(
                inverted_mask, kernel_size=kernel_size, stride=1, padding=padding
            )
            eroded_mask = (1.0 - eroded_inverted).squeeze(0).squeeze(0).bool()
            drivable_mask = eroded_mask
        
        return drivable_mask if drivable_mask.any() else None
    
    def _sample_drivable_position(
        self, drivable_mask: torch.Tensor, obj_h: int, obj_w: int, h: int, w: int, max_attempts: int = 500
    ) -> Optional[Tuple[int, int]]:
        """
        Sample a random position within drivable regions that can fit the object.
        
        Args:
            drivable_mask: Binary mask of drivable regions (H, W)
            obj_h: Object height (after scaling)
            obj_w: Object width (after scaling)
            h: Image height
            w: Image width
            max_attempts: Maximum attempts to find valid position
            
        Returns:
            (x, y) position tuple or None if no valid position found
        """
        if drivable_mask is None or not drivable_mask.any():
            return None
        
        # Ensure object dimensions are valid
        obj_h = max(1, min(obj_h, h))
        obj_w = max(1, min(obj_w, w))
        
        # Pre-filter valid positions to ensure top-left fits (y<=h-obj_h, x<=w-obj_w)
        all_valid_positions = torch.nonzero(drivable_mask, as_tuple=False)  # Shape: (N, 2) with [y, x]
        
        # Config finale: y_low mixture - 70% a 0.70H, 30% a 0.55H
        if random.random() < 0.7:
            y_low = int(0.70 * h)  # 70% dei casi: molto bottom-biased
        else:
            y_low = int(0.55 * h)  # 30% dei casi: leggermente più alto ma sempre bottom
        
        max_y = h - obj_h
        max_x = w - obj_w
        
        # Filter positions where object fits (top-left corner constraint) AND y >= y_low
        valid_positions = []
        for pos in all_valid_positions:
            y, x = pos.tolist()
            # NEW: y must be in [y_low, max_y] so the object is bottom-biased
            if y < y_low:
                continue
            if y <= max_y and x <= max_x:
                valid_positions.append([y, x])
        
        # Fallback intelligente: se troppo restrittivo, rilassa y_low ma sempre bottom-ish
        if len(valid_positions) == 0:
            # fallback: relax y_low to a lower threshold but still bottom-ish
            y_low_relaxed = int(0.50 * h)
            for pos in all_valid_positions:
                y, x = pos.tolist()
                if y < y_low_relaxed:
                    continue
                if y <= max_y and x <= max_x:
                    valid_positions.append([y, x])
        
        if len(valid_positions) == 0:
            return None
        
        valid_positions = torch.tensor(valid_positions, device=drivable_mask.device, dtype=torch.long)
        
        # Try to find a position where the object fits
        for _ in range(max_attempts):
            # Randomly select a valid position
            idx = random.randint(0, len(valid_positions) - 1)
            y, x = valid_positions[idx].tolist()
            
            # Position already guaranteed to fit (pre-filtered), but verify bounds
            x_end = x + obj_w
            y_end = y + obj_h
            
            if x >= 0 and y >= 0 and x_end <= w and y_end <= h:
                # Config finale: drivable_percentage >= 0.5 (più permissivo)
                region_mask = drivable_mask[y:y_end, x:x_end]
                if region_mask.numel() > 0:
                    drivable_percentage = region_mask.sum().float() / region_mask.numel()
                    if drivable_percentage >= 0.5:  # Config finale: 0.5 (più permissivo)
                        return (x, y)
        
        # Fallback: return None, will use random position in forward
        return None
    
    def _fallback_safe_position(
        self, target: Dict[str, Any], obj_h: int, obj_w: int, h: int, w: int, y_min: int
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Fallback safe: random position solo in valid_area e bottom band (y >= y_min).
        Non piazza mai nel padding nero.
        
        Args:
            target: Target dictionary (deve contenere valid_mask o semseg)
            obj_h: Object height
            obj_w: Object width
            h: Image height
            w: Image width
            y_min: Minimum y position (bottom constraint)
        
        Returns:
            (x, y) position tuple or (None, None) if no valid position found
        """
        # valid area: preferisci target["valid_mask"] se c'è
        if "valid_mask" in target:
            valid_area = target["valid_mask"]
        elif "semseg" in target:
            valid_area = (target["semseg"] != 255)
        else:
            return (None, None)  # No valid area info, skip

        # bottom constraint
        valid_area = valid_area.clone()
        valid_area[:y_min, :] = False

        # Prova a campionare qualche punto valido
        ysxs = torch.nonzero(valid_area, as_tuple=False)
        if ysxs.numel() > 0:
            idx = random.randint(0, ysxs.shape[0] - 1)
            y, x = ysxs[idx].tolist()
            # clamp per farci stare l'oggetto
            x = min(x, max(0, w - obj_w))
            y = min(y, max(y_min, h - obj_h))  # Ensure y >= y_min
            return (x, y)
        else:
            return (None, None)  # No valid position
    
    def _reset_placement_counters(self):
        """Reset placement counters (called at start of each epoch for logging)"""
        self.drivable_placement_count = 0
        self.random_placement_count = 0
    
    def _get_placement_stats(self):
        """Get placement statistics for logging"""
        total = self.drivable_placement_count + self.random_placement_count
        if total > 0:
            drivable_pct = (self.drivable_placement_count / total) * 100
            random_pct = (self.random_placement_count / total) * 100
            return drivable_pct, random_pct
        return 0.0, 0.0


class COCOOutlierDataset:
    """
    Dataset for loading COCO objects for Outlier Exposure.
    
    Loads individual objects (with masks) from COCO dataset for cut-paste augmentation.
    Supports both directory-based and zip-based COCO datasets.
    """
    
    def __init__(
        self,
        coco_path: str,
        split: str = "train2017",
        min_area: int = 1000,
        max_area: Optional[int] = None,
        use_zip: bool = False,
    ):
        """
        Args:
            coco_path: Path to COCO dataset directory or zip file
            split: Dataset split ("train2017" or "val2017")
            min_area: Minimum object area in pixels (filters small objects)
            max_area: Maximum object area in pixels (filters very large objects)
            use_zip: If True, load from zip files instead of directories
        """
        if not COCO_AVAILABLE:
            raise ImportError(
                "pycocotools is required for COCOOutlierDataset. "
                "Install with: pip install pycocotools"
            )
        
        self.coco_path = Path(coco_path)
        self.split = split
        self.min_area = min_area
        self.max_area = max_area
        self.use_zip = use_zip
        
        # Load COCO annotations
        if use_zip:
            self._load_from_zip()
        else:
            self._load_from_directory()
        
        # Pre-process and cache valid objects
        self._prepare_valid_objects()
        
        print(f"COCOOutlierDataset: Loaded {len(self.valid_objects)} valid objects from {split}")
    
    def _load_from_directory(self):
        """Load COCO from directory structure."""
        ann_file = self.coco_path / "annotations" / f"instances_{self.split}.json"
        
        if not ann_file.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found: {ann_file}\n"
                f"Expected structure: {self.coco_path}/annotations/instances_{self.split}.json"
            )
        
        self.coco = COCO(str(ann_file))
        self.img_dir = self.coco_path / self.split
        
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"COCO image directory not found: {self.img_dir}\n"
                f"Expected: {self.coco_path}/{self.split}/"
            )
    
    def _load_from_zip(self):
        """Load COCO from zip files (compatible with eomt dataset structure)."""
        # Try to find zip files
        annotations_zip = self.coco_path / "annotations_trainval2017.zip"
        images_zip = self.coco_path / f"{self.split}.zip"
        
        if not annotations_zip.exists() or not images_zip.exists():
            raise FileNotFoundError(
                f"COCO zip files not found. Expected:\n"
                f"  - {annotations_zip}\n"
                f"  - {images_zip}"
            )
        
        # Load annotations from zip
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            ann_path = f"annotations/instances_{self.split}.json"
            if ann_path not in zip_ref.namelist():
                raise FileNotFoundError(f"Annotation file not found in zip: {ann_path}")
            
            with zip_ref.open(ann_path) as f:
                coco_data = json.load(f)
        
        # Create temporary COCO object
        # We'll need to handle images differently for zip
        self.coco_data = coco_data
        self.annotations_zip = annotations_zip
        self.images_zip = images_zip
        
        # Create a minimal COCO object for annotation access
        # Save annotations to temp file for COCO to load
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_data, f)
            temp_ann_file = f.name
        
        self.coco = COCO(temp_ann_file)
        self.temp_ann_file = temp_ann_file
    
    def _prepare_valid_objects(self):
        """Pre-process and cache valid objects for fast access."""
        self.valid_objects = []
        img_ids = self.coco.getImgIds()
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            img_info = self.coco.loadImgs(img_id)[0]
            
            for ann in anns:
                # Filter criteria
                area = ann['area']
                is_crowd = ann.get('iscrowd', 0)
                category_id = ann['category_id']
                
                if is_crowd:
                    continue
                
                if area < self.min_area:
                    continue
                
                if self.max_area is not None and area > self.max_area:
                    continue
                
                # Store object info (all categories allowed - hybrid strategy uses mapping)
                self.valid_objects.append({
                    'img_id': img_id,
                    'ann_id': ann['id'],
                    'img_info': img_info,
                    'ann': ann,
                    'bbox': ann['bbox'],  # [x, y, width, height]
                    'area': area,
                })
        
        if len(self.valid_objects) == 0:
            raise ValueError(
                f"No valid objects found in COCO {self.split}. "
                f"Try reducing min_area (current: {self.min_area})"
            )
        
        # Log category statistics
        from collections import Counter
        category_ids = [obj['ann']['category_id'] for obj in self.valid_objects]
        category_counts = Counter(category_ids)
        
        # Get category names
        cat_ids_unique = list(category_counts.keys())
        cats = self.coco.loadCats(cat_ids_unique)
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
        
        # Sort by count (descending)
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n📊 COCO Categories in valid_objects ({len(category_counts)} unique categories):")
        print(f"   Total objects: {len(self.valid_objects)}")
        print(f"   Top 20 categories by count:")
        for cat_id, count in sorted_categories[:20]:
            cat_name = cat_id_to_name.get(cat_id, f"unknown_{cat_id}")
            percentage = (count / len(self.valid_objects)) * 100
            print(f"     {cat_id:2d} ({cat_name:20s}): {count:5d} objects ({percentage:5.2f}%)")
        
        if len(sorted_categories) > 20:
            print(f"     ... and {len(sorted_categories) - 20} more categories")
        print()
    
    def _load_image_from_directory(self, img_info: dict) -> np.ndarray:
        """Load image from directory."""
        img_path = self.img_dir / img_info['file_name']
        img = PILImage.open(img_path).convert("RGB")
        return np.array(img)
    
    def _load_image_from_zip(self, img_info: dict) -> np.ndarray:
        """Load image from zip file."""
        with zipfile.ZipFile(self.images_zip, 'r') as zip_ref:
            img_path_in_zip = f"{self.split}/{img_info['file_name']}"
            if img_path_in_zip not in zip_ref.namelist():
                raise FileNotFoundError(f"Image not found in zip: {img_path_in_zip}")
            
            with zip_ref.open(img_path_in_zip) as f:
                img = PILImage.open(BytesIO(f.read())).convert("RGB")
                return np.array(img)
    
    def _load_image(self, img_info: dict) -> np.ndarray:
        """Load image (from directory or zip)."""
        if self.use_zip:
            return self._load_image_from_zip(img_info)
        else:
            return self._load_image_from_directory(img_info)
    
    def __len__(self):
        return len(self.valid_objects)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load a COCO object.
        
        Returns:
            img_tensor: Object image tensor (3, H, W) in range [0, 1]
            mask_tensor: Object mask tensor (H, W) of type bool
            category_name: COCO category name (e.g., "person", "car", "chair")
        """
        obj_info = self.valid_objects[idx]
        img_info = obj_info['img_info']
        ann = obj_info['ann']
        bbox = obj_info['bbox']
        
        # Get category name
        category_id = ann['category_id']
        cats = self.coco.loadCats([category_id])
        category_name = cats[0]['name'] if cats else "unknown"
        
        # Load full image
        img_array = self._load_image(img_info)
        img_h, img_w = img_array.shape[:2]
        
        # Load mask
        mask = self.coco.annToMask(ann)
        
        # Extract bounding box
        x, y, w, h = [int(v) for v in bbox]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Crop image and mask
        if w > 0 and h > 0:
            img_crop = img_array[y:y+h, x:x+w]
            mask_crop = mask[y:y+h, x:x+w]
            
            # Ensure mask is binary
            mask_crop = (mask_crop > 0.5).astype(np.uint8)
            
            # Convert to tensors
            # Image: PIL -> tensor, normalize to [0, 1]
            img_pil = PILImage.fromarray(img_crop)
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img_pil)  # Already in [0, 1]
            
            # Mask: numpy -> tensor (bool)
            mask_tensor = torch.from_numpy(mask_crop).bool()
            
            # Ensure minimum size
            if img_tensor.shape[1] < 10 or img_tensor.shape[2] < 10:
                # If too small, try another object
                if idx + 1 < len(self.valid_objects):
                    return self.__getitem__(idx + 1)
                else:
                    return self.__getitem__(0)
            
            return img_tensor, mask_tensor, category_name
        else:
            # Invalid bbox, try next object
            if idx + 1 < len(self.valid_objects):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)
    
    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, 'temp_ann_file') and hasattr(self, 'temp_ann_file'):
            import os
            try:
                os.unlink(self.temp_ann_file)
            except:
                pass
