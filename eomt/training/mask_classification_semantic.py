# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import glob
import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule
from training.ood_validation_helper import compute_anomaly_map_msp, auprc, fpr_at_95_tpr


class MaskClassificationSemantic(LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        ignore_idx: int = 255,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
        # Energy OOD Loss with Warmup parameters
        energy_ood_enabled: bool = True,
        energy_ood_max_weight: float = 0.002,
        energy_warmup_epochs: int = 15,
        energy_warmup_start_epoch: int = 0,  # Virtual starting epoch for warmup (for resume from weights)
        max_epochs: int = 50,
        logit_norm_enabled: bool = False,
        logit_norm_tau: float = 0.04,
        logit_norm_eps: float = 1e-6,
        lr_decoder: Optional[float] = None,  # Optional: override LR for decoder/head/upscale
        lr_backbone: Optional[float] = None,  # Optional: LR for unfrozen backbone blocks
        unfreeze_last_n_blocks: int = 0,  # Number of last backbone blocks to unfreeze (0 = all frozen)
        # OOD Validation parameters
        ood_lostfound_path: Optional[str] = None,  # Path to FS_LostFound_full dataset
        ood_fsstatic_path: Optional[str] = None,  # Path to fs_static dataset
        ood_val_num_samples: int = 40,  # Number of images per OOD dataset to evaluate (30-50)
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
            lr_decoder=lr_decoder,
            lr_backbone=lr_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

        self.save_hyperparameters(ignore=["_class_path"])

        self.ignore_idx = ignore_idx
        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes = range(num_classes)

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
            eim_enabled=energy_ood_enabled,  # Energy OOD with warmup
            eim_temperature=1.0,
            eim_weight=energy_ood_max_weight,  # Max weight after warmup
            energy_warmup_epochs=energy_warmup_epochs,
            energy_warmup_start_epoch=energy_warmup_start_epoch,  # Virtual starting epoch for resume
            max_epochs=max_epochs,
            logit_norm_enabled=logit_norm_enabled,
            logit_norm_tau=logit_norm_tau,
            logit_norm_eps=logit_norm_eps,
        )

        self.init_metrics_semantic(ignore_idx, self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)
        
        # OOD Validation parameters
        self.ood_lostfound_path = ood_lostfound_path
        self.ood_fsstatic_path = ood_fsstatic_path
        self.ood_val_num_samples = ood_val_num_samples
    
    def on_train_epoch_start(self):
        """Update energy loss warmup scheduler with current epoch and log status."""
        self.criterion.set_epoch(self.current_epoch)
        
        # Log Energy Loss status at the start of each epoch
        if self.criterion.eim_enabled:
            current_weight = self.criterion.energy_ood_loss.get_current_weight()
            is_warmup = self.current_epoch < self.criterion.energy_ood_loss.warmup_epochs
            
            if is_warmup:
                status_msg = (
                    f"🔵 Epoch {self.current_epoch}: Energy Loss DISABLED (warmup phase) "
                    f"[{self.current_epoch}/{self.criterion.energy_ood_loss.warmup_epochs}]"
                )
            else:
                progress = ((self.current_epoch - self.criterion.energy_ood_loss.warmup_epochs) / 
                           (self.criterion.energy_ood_loss.max_epochs - self.criterion.energy_ood_loss.warmup_epochs)) * 100
                status_msg = (
                    f"🟢 Epoch {self.current_epoch}: Energy Loss ACTIVE "
                    f"[weight: {current_weight:.6f}/{self.criterion.energy_ood_loss.max_weight:.6f}, "
                    f"progress: {progress:.1f}%]"
                )
            
            # Log to console (rank_zero_info for distributed training)
            logging.info(status_msg)
            
            # Also log as metric for wandb/other loggers
            self.log("energy/epoch_weight", current_weight, sync_dist=False)
            self.log("energy/warmup_active", float(is_warmup), sync_dist=False)

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        img_sizes = [img.shape[-2:] for img in imgs]
        crops, origins = self.window_imgs_semantic(imgs)
        mask_logits_per_layer, class_logits_per_layer = self(crops)

        targets = self.to_per_pixel_targets_semantic(targets, self.ignore_idx)

        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            crop_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)
            logits = self.revert_window_logits_semantic(crop_logits, origins, img_sizes)

            self.update_metrics_semantic(logits, targets, i)

            if batch_idx == 0:
                self.plot_semantic(
                    imgs[0], targets[0], logits[0], log_prefix, i, batch_idx
                )

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_semantic("val")
        
        # OOD Validation: Lightweight validation on OOD datasets
        if (self.ood_lostfound_path or self.ood_fsstatic_path) and not self.trainer.sanity_checking:
            self._validate_ood_datasets()
    
    def _validate_ood_datasets(self):
        """
        Lightweight OOD validation on FS_LostFound and fs_static datasets.
        Evaluates 30-50 images per dataset and logs AUPRC and FPR95 metrics.
        Also logs aggregated metric for checkpointing/earlystopping.
        """
        # CRITICAL FIX #2: Save training state and restore it even on exception
        was_training = self.network.training
        try:
            # Import here to avoid circular imports
            from sklearn.metrics import average_precision_score, roc_curve
            
            device = next(self.network.parameters()).device
            
            # Setup image transform
            input_transform = Compose([
                Resize(self.img_size, Image.BILINEAR),
                ToTensor(),
            ])
            target_transform = Compose([
                Resize(self.img_size, Image.NEAREST),
            ])
            
            lostfound_auprc, fsstatic_auprc = None, None
            
            # Validate FS_LostFound
            if self.ood_lostfound_path:
                lostfound_auprc, lostfound_fpr95 = self._validate_single_ood_dataset(
                    self.ood_lostfound_path,
                    "FS_LostFound",
                    input_transform,
                    target_transform,
                    device,
                    image_ext="*.png",
                )
                if lostfound_auprc is not None:
                    self.log("metrics/ood_lostfound_auprc", lostfound_auprc, sync_dist=False)
                    self.log("metrics/ood_lostfound_fpr95", lostfound_fpr95, sync_dist=False)
            
            # Validate fs_static
            if self.ood_fsstatic_path:
                fsstatic_auprc, fsstatic_fpr95 = self._validate_single_ood_dataset(
                    self.ood_fsstatic_path,
                    "fs_static",
                    input_transform,
                    target_transform,
                    device,
                    image_ext="*.jpg",
                )
                if fsstatic_auprc is not None:
                    self.log("metrics/ood_fsstatic_auprc", fsstatic_auprc, sync_dist=False)
                    self.log("metrics/ood_fsstatic_fpr95", fsstatic_fpr95, sync_dist=False)
            
            # Task 1A: Compute and log aggregated metric with robust fallback handling
            # CRITICAL FIX: Always log ood_avg_auprc to avoid "monitored metric not found" error
            # If both are None, log -1.0 (safe value: no checkpoint saved, no false early stop, but no crash)
            if lostfound_auprc is not None and fsstatic_auprc is not None:
                # Both available: use average
                avg_auprc = 0.5 * (lostfound_auprc + fsstatic_auprc)
                self.log("metrics/ood_avg_auprc", avg_auprc, sync_dist=False)
            elif lostfound_auprc is not None:
                # Fallback: use only LostFound if fs_static missing
                self.log("metrics/ood_avg_auprc", lostfound_auprc, sync_dist=False)
            elif fsstatic_auprc is not None:
                # Fallback: use only fs_static if LostFound missing
                self.log("metrics/ood_avg_auprc", fsstatic_auprc, sync_dist=False)
            else:
                # CRITICAL FIX: Always log (even with safe value) to avoid Lightning crash
                # -1.0 is safe: no checkpoint saved, no false early stop, but no "monitored metric not found" error
                logging.warning("⚠️ Both OOD metrics are None, logging -1.0 as safe fallback for ood_avg_auprc")
                self.log("metrics/ood_avg_auprc", -1.0, sync_dist=False)
        except Exception as e:
            # Don't crash training if OOD validation fails
            logging.warning(f"⚠️ OOD validation failed: {e}")
        finally:
            # CRITICAL FIX #2: Always restore training mode (even on exception)
            if was_training:
                self.network.train()
    
    def _validate_single_ood_dataset(
        self,
        dataset_path: str,
        dataset_name: str,
        input_transform: Compose,
        target_transform: Compose,
        device: torch.device,
        image_ext: str = "*.png",
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Validate a single OOD dataset.
        
        Returns:
            (auprc, fpr95) tuple or (None, None) if validation fails
        """
        try:
            # Find image files
            images_dir = os.path.join(dataset_path, "images")
            if not os.path.exists(images_dir):
                logging.warning(f"⚠️ OOD dataset images directory not found: {images_dir}")
                return None, None
            
            image_files = glob.glob(os.path.join(images_dir, image_ext))
            if len(image_files) == 0:
                logging.warning(f"⚠️ No images found in {images_dir} with pattern {image_ext}")
                return None, None
            
            # Sample random images (30-50)
            num_samples = min(self.ood_val_num_samples, len(image_files))
            sampled_files = random.sample(image_files, num_samples)
            
            anomaly_scores_list = []
            ood_labels_list = []
            
            self.network.eval()
            with torch.no_grad():
                for img_path in sampled_files:
                    try:
                        # Load and preprocess image
                        img_pil = Image.open(img_path).convert("RGB")
                        img_tensor = input_transform(img_pil).float().to(device)  # [3, H, W] - NO batch dimension
                        
                        # Verification log (temporary)
                        logging.debug(f"OOD validation: img_tensor.shape before windowing = {img_tensor.shape} (expected [3, H, W])")

                        # Forward pass (use windowing for large images)
                        img_sizes = [img_tensor.shape[-2:]]
                        crops, origins = self.window_imgs_semantic([img_tensor])  # Expects [3, H, W], not [1, 3, H, W]
                        
                        # Verification log (temporary): check crops format
                        if len(crops) > 0:
                            logging.debug(f"OOD validation: crops[0].shape = {crops[0].shape} (expected [1, 3, h, w] or similar)")
                        
                        # window_imgs_semantic returns list of crops, self() expects list or tensor
                        # If crops need batch dimension, they should already have it from windowing
                        mask_logits_per_layer, class_logits_per_layer = self(crops)
                        
                        # Use last layer output
                        mask_logits = mask_logits_per_layer[-1]  # [B, Q, h, w]
                        class_logits = class_logits_per_layer[-1]  # [B, Q, C+1]
                        
                        # Interpolate to target size
                        mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear", align_corners=False)
                        
                        # Convert to per-pixel logits
                        per_pixel_logits = self.to_per_pixel_logits_semantic(mask_logits, class_logits)  # [B, C, H, W]
                        
                        # Revert windowing
                        logits = self.revert_window_logits_semantic(per_pixel_logits, origins, img_sizes)  # [B, C, H, W]
                        pixel_logits = logits[0]  # [C, H, W]
                        
                        # Compute anomaly map using MSP
                        anomaly_map = compute_anomaly_map_msp(pixel_logits, temperature=1.0)  # [H, W]
                        anomaly_scores = anomaly_map.detach().cpu().numpy()
                        
                        # Load ground truth
                        gt_path = img_path.replace("images", "labels_masks")
                        # Handle different extensions
                        if "fs_static" in gt_path:
                            gt_path = gt_path.replace(".jpg", ".png")
                        elif "LostAndFound" in gt_path or "FS_LostFound" in gt_path:
                            # Already .png
                            pass
                        
                        if not os.path.exists(gt_path):
                            continue
                        
                        gt_img = Image.open(gt_path)
                        gt_img = target_transform(gt_img)
                        ood_gts = np.array(gt_img)
                        
                        # Convert to binary labels (0=ID, 1=OOD, 255=ignore)
                        # Task 1B: Check if LostFound is already binary before remapping
                        if "LostAndFound" in gt_path or "FS_LostFound" in gt_path:
                            unique_vals = np.unique(ood_gts)
                            # Check if already binary (only contains 0, 1, 255)
                            if set(unique_vals).issubset({0, 1, 255}):
                                # Already binary: use directly (0=ID, 1=OOD, 255=ignore)
                                pass
                            else:
                                # Need remapping: LostFound format with multiple classes
                                # Task 1B: Log warning with unique values info for debugging
                                unique_sample = unique_vals[:10] if len(unique_vals) > 10 else unique_vals
                                logging.warning(
                                    f"⚠️ LostFound GT has non-binary format: unique values (first 10)={unique_sample.tolist()}, "
                                    f"min={unique_vals.min()}, max={unique_vals.max()}. Applying remapping."
                                )
                                ood_gts = np.where(ood_gts == 0, 255, ood_gts)  # 0 -> ignore
                                ood_gts = np.where(ood_gts == 1, 0, ood_gts)  # 1 -> ID
                                ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)  # 2-200 -> OOD
                        elif "fs_static" in gt_path:
                            # fs_static: 0=ID, 1=OOD (already binary)
                            pass
                        
                        # Resize anomaly scores to match GT if needed
                        if anomaly_scores.shape != ood_gts.shape:
                            # Use torch interpolation for resizing
                            anomaly_tensor = torch.from_numpy(anomaly_scores).unsqueeze(0).unsqueeze(0).float()
                            anomaly_tensor = F.interpolate(
                                anomaly_tensor,
                                size=ood_gts.shape,
                                mode="bilinear",
                                align_corners=False
                            )
                            anomaly_scores = anomaly_tensor.squeeze().numpy()
                        
                        # Filter out ignore pixels (255)
                        valid_mask = (ood_gts != 255)
                        if valid_mask.sum() > 0:
                            anomaly_scores_list.append(anomaly_scores[valid_mask])
                            ood_labels_list.append(ood_gts[valid_mask])
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to process {img_path}: {e}")
                        continue
            
            if len(anomaly_scores_list) == 0:
                logging.warning(f"⚠️ No valid samples processed for {dataset_name}")
                return None, None
            
            # Concatenate all scores and labels
            all_scores = np.concatenate(anomaly_scores_list)
            all_labels = np.concatenate(ood_labels_list).astype(np.int32)
            
            # Compute metrics
            try:
                auprc_value = auprc(all_scores, all_labels)
                fpr95_value = fpr_at_95_tpr(all_scores, all_labels)
                
                logging.info(f"📊 {dataset_name} OOD Validation: AUPRC={auprc_value*100:.2f}%, FPR95={fpr95_value*100:.2f}%")
                return auprc_value, fpr95_value
            except Exception as e:
                logging.warning(f"⚠️ Failed to compute metrics for {dataset_name}: {e}")
                return None, None
        except Exception as e:
            logging.warning(f"⚠️ OOD validation failed for {dataset_name}: {e}")
            return None, None

    def on_validation_end(self):
        self._on_eval_end_semantic("val")
