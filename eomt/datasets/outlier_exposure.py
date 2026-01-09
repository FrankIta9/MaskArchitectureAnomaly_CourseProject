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
