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


class OutlierExposureTransform(nn.Module):
    """
    Outlier Exposure transformation using cut-paste augmentation.
    
    This transform randomly pastes objects from an outlier dataset (e.g., COCO)
    onto Cityscapes images to create synthetic anomaly examples.
    """
    
    def __init__(
        self,
        outlier_dataset: Optional[Any] = None,
        paste_probability: float = 0.5,
        min_objects: int = 1,
        max_objects: int = 3,
        min_scale: float = 0.1,
        max_scale: float = 0.3,
        blend_alpha: float = 0.8,
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
        """
        super().__init__()
        self.outlier_dataset = outlier_dataset
        self.paste_probability = paste_probability
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blend_alpha = blend_alpha
        
    def _get_random_outlier_object(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random object from the outlier dataset.
        
        Returns:
            Tuple of (object_image, object_mask)
        """
        if self.outlier_dataset is None or len(self.outlier_dataset) == 0:
            # Return a dummy object if no outlier dataset is provided
            # In practice, you should load COCO or another dataset
            dummy_img = torch.zeros((3, 64, 64))
            dummy_mask = torch.ones((64, 64), dtype=torch.bool)
            return dummy_img, dummy_mask
            
        idx = random.randint(0, len(self.outlier_dataset) - 1)
        # This is a placeholder - actual implementation depends on your outlier dataset format
        # You would load COCO images and masks here
        return self._load_coco_object(idx)
    
    def _load_coco_object(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a COCO object at the given index.
        This is a placeholder - implement based on your COCO dataset loader.
        """
        # TODO: Implement actual COCO object loading
        # For now, return a dummy object
        dummy_img = torch.rand((3, 64, 64))
        dummy_mask = torch.ones((64, 64), dtype=torch.bool)
        return dummy_img, dummy_mask
    
    def _paste_object(
        self,
        img: torch.Tensor,
        target: Dict[str, Any],
        obj_img: torch.Tensor,
        obj_mask: torch.Tensor,
        position: Tuple[int, int],
        scale: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Paste an object onto the image at the given position.
        
        Args:
            img: Input image tensor
            target: Target dictionary with masks and labels
            obj_img: Object image to paste
            obj_mask: Object mask
            position: (x, y) position to paste
            scale: Scale factor for the object
            
        Returns:
            Modified image and target
        """
        h, w = img.shape[-2:]
        obj_h, obj_w = obj_img.shape[-2:]
        
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
            
            # Create anomaly mask (will be used as "no object" class)
            anomaly_mask = torch.zeros((h, w), dtype=torch.bool, device=img.device)
            anomaly_mask[y:y+new_h, x:x+new_w] = obj_mask_resized
            
            # Blend object into image
            img_clone = img.clone()
            for c in range(3):
                img_clone[c, y:y+new_h, x:x+new_w] = (
                    self.blend_alpha * obj_img_resized[c] + 
                    (1 - self.blend_alpha) * img_clone[c, y:y+new_h, x:x+new_w]
                )
            
            # Add anomaly mask to target (as "no object" class = num_classes)
            num_classes = target["labels"].max().item() if len(target["labels"]) > 0 else 0
            anomaly_label = num_classes + 1  # Use "no object" class index
            
            # Append anomaly mask and label
            existing_masks = target["masks"]
            existing_labels = target["labels"]
            existing_is_crowd = target.get("is_crowd", torch.zeros_like(existing_labels))
            
            new_masks = torch.cat([existing_masks, anomaly_mask.unsqueeze(0)], dim=0)
            new_labels = torch.cat([existing_labels, torch.tensor([anomaly_label], device=existing_labels.device)])
            new_is_crowd = torch.cat([existing_is_crowd, torch.tensor([False], device=existing_is_crowd.device)])
            
            target = {
                "masks": new_masks,
                "labels": new_labels,
                "is_crowd": new_is_crowd,
            }
            
            return img_clone, target
        
        return img, target
    
    def forward(
        self,
        img: torch.Tensor,
        target: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply outlier exposure transformation.
        
        Args:
            img: Input image tensor
            target: Target dictionary
            
        Returns:
            Transformed image and target
        """
        if random.random() > self.paste_probability:
            return img, target
        
        num_objects = random.randint(self.min_objects, self.max_objects)
        h, w = img.shape[-2:]
        
        for _ in range(num_objects):
            # Get random outlier object
            obj_img, obj_mask = self._get_random_outlier_object()
            
            # Random position
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            
            # Random scale
            scale = random.uniform(self.min_scale, self.max_scale)
            
            # Paste object
            img, target = self._paste_object(
                img, target, obj_img, obj_mask, (x, y), scale
            )
        
        return img, target


class COCOOutlierDataset:
    """
    Simple wrapper for loading COCO objects for outlier exposure.
    This is a placeholder - implement based on your COCO dataset structure.
    """
    
    def __init__(self, coco_path: Optional[str] = None):
        """
        Args:
            coco_path: Path to COCO dataset
        """
        self.coco_path = coco_path
        self.objects = []
        # TODO: Load COCO dataset and extract object crops
        # For now, this is a placeholder
        
    def __len__(self):
        return len(self.objects) if self.objects else 1000  # Placeholder
    
    def __getitem__(self, idx):
        # TODO: Return (object_image, object_mask) from COCO
        # Placeholder implementation
        return None, None
