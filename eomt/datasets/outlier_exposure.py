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
        """
        super().__init__()
        self.outlier_dataset = outlier_dataset
        self.paste_probability = paste_probability
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.blend_alpha = blend_alpha
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
        
    def _get_random_outlier_object(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a random object from the outlier dataset.
        
        Returns:
            Tuple of (object_image, object_mask)
        """
        if self.outlier_dataset is None or len(self.outlier_dataset) == 0:
            # Return a dummy object if no outlier dataset is provided
            dummy_img = torch.zeros((3, 64, 64))
            dummy_mask = torch.ones((64, 64), dtype=torch.bool)
            return dummy_img, dummy_mask
            
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
            
            # Blend object into image using the object mask
            img_clone = img.clone()
            # Only blend where the mask is True
            mask_float = obj_mask_resized.float()
            for c in range(3):
                # Blend: alpha * obj + (1-alpha) * bg, but only where mask is True
                blended = (
                    self.blend_alpha * obj_img_resized[c] * mask_float + 
                    (1 - self.blend_alpha * mask_float) * img_clone[c, y:y+new_h, x:x+new_w]
                )
                img_clone[c, y:y+new_h, x:x+new_w] = blended
            
            # For Outlier Exposure: paste object visually but DON'T add to targets
            # The model should learn to predict "no object" for these regions naturally
            # This is the standard approach for OE in anomaly segmentation
            
            return img_clone, target
        
        return img, target
    
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
            Transformed image and target
        """
        if random.random() > self.paste_probability:
            return img, target
        
        num_objects = random.randint(self.min_objects, self.max_objects)
        h, w = img.shape[-2:]
        
        # Get drivable mask once for all objects
        drivable_mask = self._get_drivable_mask(target, h, w)
        
        for _ in range(num_objects):
            # Get random outlier object
            obj_img, obj_mask = self._get_random_outlier_object()
            obj_h_orig, obj_w_orig = obj_img.shape[-2:]
            
            # Select base scale using weighted distribution if enabled, otherwise uniform
            if self.use_weighted_scale:
                base_scale = self._sample_weighted_scale()
            else:
                base_scale = random.uniform(self.min_scale, self.max_scale)
            
            # First, sample an approximate Y position for perspective-aware scaling
            # If drivable regions are enabled, prefer lower Y (closer to camera, more drivable)
            if drivable_mask is not None and drivable_mask.any():
                # Sample Y from lower half more often (closer to camera)
                # 70% chance to sample from lower half, 30% from upper half
                if random.random() < 0.7:
                    y_approx = random.randint(h // 2, h - 1)  # Lower half
                else:
                    y_approx = random.randint(0, h // 2)  # Upper half
            else:
                y_approx = random.randint(0, h - 1)
            
            # Apply perspective-aware scaling based on approximate Y position
            scale = self._apply_perspective_aware_scale(base_scale, y_approx, h)
            
            # Clamp scale to valid range
            scale = max(self.min_scale, min(self.max_scale * 1.5, scale))  # Allow slightly larger for perspective
            
            # Calculate final object dimensions after scaling
            obj_h_scaled = max(1, int(obj_h_orig * scale))
            obj_w_scaled = max(1, int(obj_w_orig * scale))
            obj_h_scaled = min(obj_h_scaled, h)
            obj_w_scaled = min(obj_w_scaled, w)
            
            # Sample position (with drivable region constraint if enabled)
            position = None
            if drivable_mask is not None:
                # Try to sample from drivable regions using scaled dimensions
                position = self._sample_drivable_position(drivable_mask, obj_h_scaled, obj_w_scaled, h, w)
            
            # Fallback to random position if drivable sampling failed
            if position is None:
                x = random.randint(0, max(0, w - obj_w_scaled))
                y = random.randint(0, max(0, h - obj_h_scaled))
                
                # Re-apply perspective-aware scaling based on final Y position
                scale = self._apply_perspective_aware_scale(base_scale, y, h)
                scale = max(self.min_scale, min(self.max_scale * 1.5, scale))
            else:
                x, y = position
                # Re-apply perspective-aware scaling based on final Y position
                scale = self._apply_perspective_aware_scale(base_scale, y, h)
                scale = max(self.min_scale, min(self.max_scale * 1.5, scale))
            
            # Paste object
            img, target = self._paste_object(
                img, target, obj_img, obj_mask, (x, y), scale
            )
        
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
        
        # Perspective factor: objects in lower Y (higher y value, closer) should be larger
        # Formula inspired by ClimaOoD: hi = H/yi (inverse relationship)
        # Normalize Y to [0.5, 2.0] range for reasonable scaling
        # Y=0 (top, far) -> factor ~0.5 (smaller)
        # Y=h (bottom, close) -> factor ~2.0 (larger)
        normalized_y = (y + 1) / h  # +1 to avoid division by zero
        perspective_factor = (h / (normalized_y * h + 1))  # Inverse relationship
        
        # Normalize to reasonable range [0.7, 1.5] for perspective effect
        min_factor = 0.7
        max_factor = 1.5
        perspective_factor = min_factor + (max_factor - min_factor) * (
            (perspective_factor - 0.5) / 1.5  # Normalize from [0.5, 2.0] to [0.7, 1.5]
        )
        perspective_factor = max(min_factor, min(max_factor, perspective_factor))
        
        # Apply perspective strength (0.0 = no effect, 1.0 = full effect)
        adjusted_factor = 1.0 + (perspective_factor - 1.0) * self.perspective_strength
        
        return base_scale * adjusted_factor
    
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
        num_classes = min(masks.shape[0], len(labels))
        
        # Find masks for drivable classes (road=0, sidewalk=1)
        drivable_mask = torch.zeros((h, w), dtype=torch.bool, device=masks.device)
        
        for i in range(num_classes):
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
        
        return drivable_mask if drivable_mask.any() else None
    
    def _sample_drivable_position(
        self, drivable_mask: torch.Tensor, obj_h: int, obj_w: int, h: int, w: int, max_attempts: int = 100
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
        
        # Get all valid positions (where drivable_mask is True)
        valid_positions = torch.nonzero(drivable_mask, as_tuple=False)  # Shape: (N, 2) with [y, x]
        
        if len(valid_positions) == 0:
            return None
        
        # Try to find a position where the object fits
        for _ in range(max_attempts):
            # Randomly select a valid position
            idx = random.randint(0, len(valid_positions) - 1)
            y, x = valid_positions[idx].tolist()
            
            # Check if object fits at this position
            x_end = min(x + obj_w, w)
            y_end = min(y + obj_h, h)
            
            # Adjust position if object would go out of bounds
            if x_end > w:
                x = max(0, w - obj_w)
                x_end = w
            if y_end > h:
                y = max(0, h - obj_h)
                y_end = h
            
            if x >= 0 and y >= 0 and x_end <= w and y_end <= h:
                # Check if the object region is mostly drivable (at least 40%)
                region_mask = drivable_mask[y:y_end, x:x_end]
                if region_mask.numel() > 0 and region_mask.sum().float() >= (region_mask.numel() * 0.4):
                    return (x, y)
        
        # Fallback: return None, will use random position in forward
        return None


class COCOOutlierDataset:
    """
    Dataset for loading COCO objects for Outlier Exposure.
    
    Loads individual objects (with masks) from COCO dataset for cut-paste augmentation.
    Supports both directory-based and zip-based COCO datasets.
    """
    
    def __init__(
        self,
        coco_path: str,
        split: str = "val2017",
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
                
                if is_crowd:
                    continue
                
                if area < self.min_area:
                    continue
                
                if self.max_area is not None and area > self.max_area:
                    continue
                
                # Store object info
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a COCO object.
        
        Returns:
            img_tensor: Object image tensor (3, H, W) in range [0, 1]
            mask_tensor: Object mask tensor (H, W) of type bool
        """
        obj_info = self.valid_objects[idx]
        img_info = obj_info['img_info']
        ann = obj_info['ann']
        bbox = obj_info['bbox']
        
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
            
            return img_tensor, mask_tensor
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
