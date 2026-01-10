# ---------------------------------------------------------------
# Cityscapes Semantic Dataset with Outlier Exposure Support
# Extends CityscapesSemantic to support COCO Outlier Exposure
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Optional
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
        coco_split: str = "val2017",
        use_coco_zip: bool = False,
        paste_probability: float = 0.5,
        min_objects: int = 1,
        max_objects: int = 3,
        min_scale: float = 0.1,
        max_scale: float = 0.3,
        coco_min_area: int = 1000,
        # Multi-scale weighted distribution (for better matching with small anomalies)
        use_weighted_scale: bool = False,
        scale_ranges: Optional[list] = None,  # [(min1, max1), (min2, max2), ...]
        scale_weights: Optional[list] = None,  # [weight1, weight2, ...] (should sum to 1.0)
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

        # Initialize Outlier Exposure if COCO path is provided
        outlier_exposure_transform = None
        if coco_path is not None:
            try:
                coco_dataset = COCOOutlierDataset(
                    coco_path=coco_path,
                    split=coco_split,
                    min_area=coco_min_area,
                    use_zip=use_coco_zip,
                )
                outlier_exposure_transform = OutlierExposureTransform(
                    outlier_dataset=coco_dataset,
                    paste_probability=paste_probability,
                    min_objects=min_objects,
                    max_objects=max_objects,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    use_weighted_scale=use_weighted_scale,
                    scale_ranges=scale_ranges,
                    scale_weights=scale_weights,
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

        for label_id in target[0].unique():
            cls = next((cls for cls in Cityscapes.classes if cls.id == label_id), None)

            if cls is None or cls.ignore_in_eval:
                continue

            masks.append(target[0] == label_id)
            labels.append(cls.train_id)

        return masks, labels, [False for _ in range(len(masks))]

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
