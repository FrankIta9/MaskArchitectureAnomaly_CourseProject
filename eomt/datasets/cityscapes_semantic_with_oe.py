# ---------------------------------------------------------------
# Cityscapes Semantic Dataset with Outlier Exposure Support
# Extends CityscapesSemantic to support COCO Outlier Exposure
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms
from datasets.outlier_exposure import OutlierExposureTransform, COCOOutlierDataset


class CityscapesSemanticWithOE(LightningDataModule):
    """
    Cityscapes Semantic Dataset with optional Outlier Exposure support.
    
    This extends CityscapesSemantic to optionally include COCO Outlier Exposure
    for anomaly segmentation training.
    """
    
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (1024, 1024),
        num_classes: int = 19,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        # Outlier Exposure parameters
        coco_path: Optional[str] = None,
        coco_split: str = "train2017",
        use_coco_zip: bool = False,
        paste_probability: float = 0.5,  # DEPRECATED: use p_id_paste and p_ood_paste instead
        p_id_paste: float = 0.10,  # Probability of ID paste (mappable COCO → Cityscapes)
        p_ood_paste: float = 0.30,  # Probability of OOD paste (non-mappable COCO)
        max_overlap_ratio: float = 0.02,  # Max overlap with existing GT (0.02 = 2%)
        min_objects: int = 1,
        max_objects: int = 3,
        min_scale: float = 0.1,
        max_scale: float = 0.3,
        coco_min_area: int = 1000,
        # Multi-scale weighted distribution (for better matching with small anomalies)
        use_weighted_scale: bool = False,
        scale_ranges: Optional[list] = None,  # [(min1, max1), (min2, max2), ...]
        scale_weights: Optional[list] = None,  # [weight1, weight2, ...] (should sum to 1.0)
        # Perspective-aware placement (inspired by ClimaOoD)
        use_perspective_aware: bool = True,
        perspective_strength: float = 1.0,  # 0.0 = disabled, 1.0 = full effect
        # Drivable region constraints (inspired by ClimaOoD)
        use_drivable_regions: bool = True,
        drivable_class_ids: Optional[list] = None,  # [0, 1] for road, sidewalk in Cityscapes
        # P0 Fix: Y position range and blending
        paste_y_range: Tuple[float, float] = (0.65, 0.98),
        blend_alpha: float = 1.0,  # 1.0 = dry paste (more stable), 0.8 for blending
        min_obj_size_px: int = 30,  # Minimum object size in pixels (to reduce resample)
    ) -> None:
        """
        Args:
            coco_path: Path to COCO dataset (directory or zip parent directory)
            coco_split: COCO split to use ("train2017" or "val2017")
            use_coco_zip: If True, load COCO from zip files
            paste_probability: Probability of applying cut-paste augmentation
            min_objects: Minimum number of objects to paste per image
            max_objects: Maximum number of objects to paste per image
            min_scale: Minimum scale factor for pasted objects
            max_scale: Maximum scale factor for pasted objects
            coco_min_area: Minimum object area in pixels for COCO objects
            use_weighted_scale: If True, use weighted multi-scale distribution
            scale_ranges: List of (min, max) scale ranges for weighted distribution
            scale_weights: List of weights for each scale range (should sum to 1.0)
            use_perspective_aware: If True, apply perspective-aware scaling (default: True)
            perspective_strength: Strength of perspective effect (0.0-1.0, default: 1.0)
            use_drivable_regions: If True, only place objects on drivable regions (default: True)
            drivable_class_ids: List of train_id class IDs for drivable regions (default: [0, 1])
            paste_y_range: Tuple (min_ratio, max_ratio) of image height for Y position (default: (0.65, 0.98))
            blend_alpha: Alpha blending factor for pasted objects (default: 1.0, dry paste)
        """
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # Store paste_y_range and blend_alpha for reference
        self.paste_y_range = paste_y_range
        self.blend_alpha = blend_alpha

        # Initialize Outlier Exposure if COCO path is provided
        outlier_exposure_transform = None
        if coco_path is not None:
            try:
                coco_dataset = COCOOutlierDataset(
                    coco_path=coco_path,
                    split=coco_split,
                    min_area=coco_min_area,
                    use_zip=use_coco_zip,
                    # No allowed_category_ids: hybrid strategy uses all categories
                )
                outlier_exposure_transform = OutlierExposureTransform(
                    outlier_dataset=coco_dataset,
                    paste_probability=paste_probability,  # For backward compatibility
                    p_id_paste=p_id_paste,
                    p_ood_paste=p_ood_paste,
                    max_overlap_ratio=max_overlap_ratio,
                    min_objects=min_objects,
                    max_objects=max_objects,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    use_weighted_scale=use_weighted_scale,
                    scale_ranges=scale_ranges,
                    scale_weights=scale_weights,
                    use_perspective_aware=use_perspective_aware,
                    perspective_strength=perspective_strength,
                    use_drivable_regions=use_drivable_regions,
                    drivable_class_ids=drivable_class_ids,
                    paste_y_range=paste_y_range,
                    blend_alpha=blend_alpha,
                    min_obj_size_px=min_obj_size_px,
                )
                print(f"Outlier Exposure enabled with {len(coco_dataset)} COCO objects")
            except Exception as e:
                print(f"Warning: Failed to load COCO dataset: {e}")
                print("Continuing without Outlier Exposure...")
                outlier_exposure_transform = None

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            outlier_exposure_transform=outlier_exposure_transform,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []
        
        # Extract semseg: target è tv_tensors.Mask con shape [H, W] o [1, H, W]
        # Gestisci entrambi i casi
        if target.dim() == 3:
            semseg_raw = target[0]  # [H, W]
        else:
            semseg_raw = target  # [H, W]
        
        # Crea semseg mappando label_id -> train_id
        # Ignore pixels (255) rimangono 255
        IGNORE = 255
        semseg = torch.full_like(semseg_raw, IGNORE, dtype=torch.long)
        
        for label_id in semseg_raw.unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            mask = (semseg_raw == label_id)
            masks.append(mask)
            labels.append(cls.train_id)
            
            # Mappa label_id -> train_id in semseg
            semseg[mask] = cls.train_id

        return masks, labels, [False for _ in range(len(masks))], semseg

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        cityscapes_dataset_kwargs = {
            "img_suffix": ".png",
            "target_suffix": ".png",
            "img_stem_suffix": "leftImg8bit",
            "target_stem_suffix": "gtFine_labelIds",
            "zip_path": Path(self.path, "leftImg8bit_trainvaltest.zip"),
            "target_zip_path": Path(self.path, "gtFine_trainvaltest.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.cityscapes_train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./leftImg8bit/train"),
            target_folder_path_in_zip=Path("./gtFine/train"),
            **cityscapes_dataset_kwargs,
        )
        self.cityscapes_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./leftImg8bit/val"),
            target_folder_path_in_zip=Path("./gtFine/val"),
            **cityscapes_dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
