# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from:
# - the torchmetrics library by the PyTorch Lightning team
# - the Mask2Former repository by Facebook, Inc. and its affiliates
# All used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math
from typing import Optional, cast
import lightning
from lightning.fabric.utilities import rank_zero_info
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import PanopticQuality, MeanAveragePrecision
from torchmetrics.functional.detection._panoptic_quality_common import (
    _prepocess_inputs,
    _Color,
    _get_color_areas,
    _calculate_iou,
)
import wandb
from PIL import Image
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.v2.functional import pad
import logging

from training.two_stage_warmup_poly_schedule import TwoStageWarmupPolySchedule

bold_green = "\033[1;32m"
reset = "\033[0m"


class LightningModule(lightning.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        llrd_l2_enabled: bool,
        lr_mult: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int],
        ckpt_path=None,
        delta_weights=False,
        load_ckpt_class_head=True,
        lr_decoder: Optional[float] = None,  # Optional: override LR for decoder/head/upscale
        lr_backbone: Optional[float] = None,  # Optional: LR for unfrozen backbone blocks
        unfreeze_last_n_blocks: int = 0,  # Number of last backbone blocks to unfreeze (0 = all frozen)
    ):
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = warmup_steps
        self.llrd_l2_enabled = llrd_l2_enabled
        self.lr_decoder = lr_decoder if lr_decoder is not None else lr
        self.lr_backbone = lr_backbone if lr_backbone is not None else lr
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks

        self.strict_loading = False
        
        # Task 1: Counter for printing ood_mask statistics for first 3 batches
        self._ood_mask_print_count = 0
        
        # Task 4: Buffer for accumulating energy statistics to compute real margins
        # Accumulate energy values during first N steps (default: 5000)
        self._energy_margin_calculation_steps = 5000
        self._energy_id_buffer = []  # List to accumulate E_id values
        self._energy_ood_buffer = []  # List to accumulate E_ood values
        self._energy_margins_computed = False  # Flag to track if margins have been computed

        if delta_weights and ckpt_path:
            logging.info("Delta weights mode")
            self._zero_init_outside_encoder(skip_class_head=not load_ckpt_class_head)
            current_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
            if not load_ckpt_class_head:
                current_state_dict = {
                    k: v
                    for k, v in current_state_dict.items()
                    if "class_head" not in k and "class_predictor" not in k
                }
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
            incompatible_keys = self.load_state_dict(combined_state_dict, strict=False)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
        elif ckpt_path:
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            incompatible_keys = self.load_state_dict(ckpt, strict=False)
            
            # Check if class_head weights were loaded correctly
            if load_ckpt_class_head:
                # Check both in checkpoint and in model (after loading)
                class_head_keys_in_ckpt = [k for k in ckpt.keys() if "class_head" in k]
                class_head_keys_in_model = [k for k, _ in self.named_parameters() if "class_head" in k]
                
                if class_head_keys_in_ckpt:
                    logging.info(f"✅ Found class_head weights in checkpoint: {class_head_keys_in_ckpt}")
                    # Verify they were actually loaded (check if missing_keys contains class_head)
                    if incompatible_keys.missing_keys:
                        missing_class_head = [k for k in incompatible_keys.missing_keys if "class_head" in k]
                        if missing_class_head:
                            logging.warning(f"⚠️ class_head keys found in checkpoint but NOT loaded: {missing_class_head}")
                            logging.warning("   This means shapes/dimensions don't match - using random initialization!")
                            # Re-initialize more safely
                            with torch.no_grad():
                                if hasattr(self.network, 'class_head'):
                                    nn.init.xavier_uniform_(self.network.class_head.weight, gain=0.1)
                                    if self.network.class_head.bias is not None:
                                        nn.init.zeros_(self.network.class_head.bias)
                                    logging.info("✅ Re-initialized class_head with Xavier uniform (gain=0.1) for stability")
                        else:
                            logging.info(f"✅ class_head weights loaded successfully into model (other keys missing, but class_head OK)")
                    else:
                        # No missing keys - all weights loaded successfully
                        logging.info(f"✅ class_head weights loaded successfully (no missing keys)")
                        # Verify loaded weights are finite (safety check)
                        with torch.no_grad():
                            if hasattr(self.network, 'class_head'):
                                if not torch.isfinite(self.network.class_head.weight).all():
                                    num_invalid = (~torch.isfinite(self.network.class_head.weight)).sum().item()
                                    logging.warning(f"⚠️ class_head.weight contains {num_invalid} non-finite values after loading!")
                                    # Clean non-finite values
                                    self.network.class_head.weight.data = torch.where(
                                        torch.isfinite(self.network.class_head.weight),
                                        self.network.class_head.weight,
                                        torch.zeros_like(self.network.class_head.weight)
                                    )
                                    logging.info("✅ Cleaned non-finite values from class_head.weight")
                else:
                    logging.warning("⚠️ class_head weights NOT found in checkpoint - using random initialization!")
                    logging.warning(f"   This is OK if checkpoint doesn't contain class_head, but may cause numerical issues.")
                    logging.warning(f"   Consider using a checkpoint that includes class_head weights or set load_ckpt_class_head=False")
                    # Initialize more safely with smaller variance
                    with torch.no_grad():
                        if hasattr(self.network, 'class_head'):
                            nn.init.xavier_uniform_(self.network.class_head.weight, gain=0.1)  # Smaller gain for stability
                            if self.network.class_head.bias is not None:
                                nn.init.zeros_(self.network.class_head.bias)
                            logging.info("✅ Initialized class_head with Xavier uniform (gain=0.1) for numerical stability")
            
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
            
            # Final verification: ensure all class_head parameters are finite after loading
            with torch.no_grad():
                if hasattr(self.network, 'class_head'):
                    all_finite = torch.isfinite(self.network.class_head.weight).all()
                    if self.network.class_head.bias is not None:
                        all_finite = all_finite and torch.isfinite(self.network.class_head.bias).all()
                    if not all_finite:
                        logging.error("❌ class_head contains non-finite values after all checks! This will cause training issues!")
                        # Emergency fix: re-initialize completely
                        nn.init.xavier_uniform_(self.network.class_head.weight, gain=0.1)
                        if self.network.class_head.bias is not None:
                            nn.init.zeros_(self.network.class_head.bias)
                        logging.warning("✅ Emergency re-initialization of class_head completed")
                    else:
                        logging.info("✅ class_head parameters verified finite and ready for training")

        self.log = torch.compiler.disable(self.log)  # type: ignore

    def configure_optimizers(self):
        # ===================================================================
        # PARTIAL BACKBONE UNFREEZE FOR OUTLIER EXPOSURE FINE-TUNING
        # ===================================================================
        # Strategy: Unfreeze last N blocks (typically 2) for fine-tuning
        # Keep most of ViT frozen; unfreeze only last N transformer blocks
        # Use two optimizer param groups:
        # - decoder/head/upscale params: lr_decoder
        # - unfrozen backbone params: lr_backbone (typically lower)
        # ===================================================================
        if hasattr(self.network, 'encoder'):
            if self.unfreeze_last_n_blocks > 0:
                # CRITICAL FIX: Freeze ENTIRE backbone first (including patch_embed, pos_embed, etc.)
                if hasattr(self.network.encoder, 'backbone'):
                    # 1) FREEZE TUTTO IL BACKBONE (incluso patch_embed, pos_embed, ecc.)
                    for param in self.network.encoder.backbone.parameters():
                        param.requires_grad = False
                    
                    # 2) UNFREEZE SOLO GLI ULTIMI N BLOCKS + NORM
                    if hasattr(self.network.encoder.backbone, 'blocks'):
                        total_blocks = len(self.network.encoder.backbone.blocks)
                        unfrozen_params = 0
                        
                        # Unfreeze last N blocks
                        for i in range(total_blocks - self.unfreeze_last_n_blocks, total_blocks):
                            for param in self.network.encoder.backbone.blocks[i].parameters():
                                param.requires_grad = True
                                unfrozen_params += param.numel()
                        
                        # Also unfreeze final norm if present
                        if hasattr(self.network.encoder.backbone, 'norm'):
                            for param in self.network.encoder.backbone.norm.parameters():
                                param.requires_grad = True
                                unfrozen_params += param.numel()
                        
                        # Count frozen params for logging
                        frozen_params = sum(p.numel() for p in self.network.encoder.backbone.parameters() if not p.requires_grad)
                        
                        # Verification: check that patch_embed is frozen
                        patch_embed_frozen = next(self.network.encoder.backbone.patch_embed.proj.parameters()).requires_grad == False
                        print(f"🔒 Backbone PARTIALLY FROZEN: {frozen_params:,} params frozen, {unfrozen_params:,} params unfrozen (last {self.unfreeze_last_n_blocks} blocks + norm)")
                        print(f"✅ Training decoder + last {self.unfreeze_last_n_blocks} backbone blocks + norm")
                        print(f"✅ VERIFICATION: patch_embed.proj.weight.requires_grad = {not patch_embed_frozen} (should be False: {patch_embed_frozen})")
                else:
                    # Fallback: freeze all encoder params
                    frozen_params = 0
                    for param in self.network.encoder.parameters():
                        param.requires_grad = False
                        frozen_params += param.numel()
                    print(f"🔒 Backbone FROZEN (fallback): {frozen_params:,} params")
            else:
                # Full freeze: freeze entire backbone
                frozen_params = 0
                for param in self.network.encoder.parameters():
                    param.requires_grad = False
                    frozen_params += param.numel()
                print(f"🔒 Backbone FROZEN: {frozen_params:,} params")
                print("✅ Training only decoder for Outlier Exposure fine-tuning")
        
        # Configure optimizer with two param groups: decoder and unfrozen backbone
        # CRITICAL FIX: Group params into 2 lists instead of creating one param group per parameter
        decoder_params = []
        backbone_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
            
            # Check if parameter belongs to encoder.backbone.blocks (last N blocks)
            # Task 2B: Robust block index parsing - find "blocks" token and read next numeric token
            is_backbone_param = False
            if self.unfreeze_last_n_blocks > 0 and 'encoder.backbone.blocks' in name:
                # Robust parsing: split by '.' and find index of 'blocks', then check next token
                parts = name.split('.')
                try:
                    blocks_idx = parts.index('blocks')
                    if blocks_idx + 1 < len(parts) and parts[blocks_idx + 1].isdigit():
                        block_idx = int(parts[blocks_idx + 1])
                        total_blocks = len(self.network.encoder.backbone.blocks) if hasattr(self.network.encoder.backbone, 'blocks') else 0
                        if total_blocks > 0 and block_idx >= (total_blocks - self.unfreeze_last_n_blocks):
                            is_backbone_param = True
                except (ValueError, IndexError):
                    # If parsing fails, treat as decoder param (safer)
                    pass
            elif self.unfreeze_last_n_blocks > 0 and 'encoder.backbone.norm' in name:
                # Final norm is also unfrozen
                is_backbone_param = True
            
            if is_backbone_param:
                backbone_params.append(param)
            else:
                # Decoder/head/upscale params
                decoder_params.append(param)
        
        # Task 2B: Create only 2 param groups (not one per parameter)
        param_groups = []
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": self.lr_decoder})
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": self.lr_backbone})
        
        # Task 2B: Log param groups info for verification
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        decoder_numel = sum(p.numel() for p in decoder_params) if decoder_params else 0
        backbone_numel = sum(p.numel() for p in backbone_params) if backbone_params else 0
        
        print(f"🎯 Trainable params: {total_trainable:,}")
        print(f"   Decoder group: {len(decoder_params)} params, {decoder_numel:,} elements @ lr={self.lr_decoder}")
        if backbone_params:
            print(f"   Backbone group: {len(backbone_params)} params, {backbone_numel:,} elements @ lr={self.lr_backbone}")
            # Task 2B: Show sample backbone param names to verify they are blocks.10, blocks.11, etc.
            # Get sample backbone parameter names for logging (avoid tensor comparison)
            sample_backbone_names = []
            backbone_param_ids = {id(p) for p in backbone_params[:3]}  # Use id() for identity comparison
            for name, p in self.named_parameters():
                if id(p) in backbone_param_ids:
                    sample_backbone_names.append(name)
                if len(sample_backbone_names) >= 3:
                    break
            if sample_backbone_names:
                print(f"   Sample backbone params: {', '.join(sample_backbone_names[:3])}")
        else:
            print("   ⚠️ WARNING: Backbone param group is empty! Unfreeze not working.")
        
        optimizer = AdamW(param_groups, weight_decay=self.weight_decay)

        # Simplified scheduler: no backbone warmup needed (it's frozen)
        # Only decoder gets warmup + poly decay
        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=0,  # No backbone params since frozen
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs):
        # P0 - Problema #2: Normalizzazione gestita da EoMT.forward
        # EoMT.forward applica già: x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std
        # Quindi qui facciamo solo la conversione a float e divisione per 255.0
        x = imgs.float() / 255.0
        
        return self.network(x)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        
        # A1: OE Statistics - Quanto OE sta colpendo davvero
        batch_has_ood = False
        batch_n_ood_pixels = 0
        batch_total_pixels = 0
        batch_n_objects_pasted = 0  # Stima: numero di regioni connesse in ood_mask
        
        if targets and "ood_mask" in targets[0]:
            for target in targets:
                if "ood_mask" in target:
                    ood_mask = target["ood_mask"]  # [H, W]
                    h, w = ood_mask.shape
                    ood_pixels = (ood_mask == 1)
                    
                    if ood_pixels.any():
                        batch_has_ood = True
                        n_ood = ood_pixels.sum().item()
                        batch_n_ood_pixels += n_ood
                        batch_total_pixels += h * w
                        
                        # Stima numero oggetti pasted: conta regioni connesse (approssimativo)
                        # Usa connected components o semplicemente conta "blob" significativi
                        ood_mask_np = ood_mask.cpu().numpy()
                        try:
                            from scipy import ndimage
                            labeled, num_features = ndimage.label(ood_mask_np == 1)
                            batch_n_objects_pasted += num_features
                        except ImportError:
                            # Fallback se scipy non disponibile: stima basata su area media
                            if n_ood > 0:
                                # Assumendo area media per oggetto ~1000 pixel
                                batch_n_objects_pasted += max(1, n_ood // 1000)
                        except Exception:
                            # Fallback: stima basata su area media
                            if n_ood > 0:
                                # Assumendo area media per oggetto ~1000 pixel
                                batch_n_objects_pasted += max(1, n_ood // 1000)
        
        # Log A1: OE statistics
        paste_applied = 1.0 if batch_has_ood else 0.0
        self.log("dbg/oe/paste_applied", paste_applied, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        self.log("dbg/oe/n_ood_pixels", float(batch_n_ood_pixels), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        
        if batch_total_pixels > 0:
            ood_ratio = batch_n_ood_pixels / batch_total_pixels
            self.log("dbg/oe/ood_ratio", ood_ratio, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        
        if batch_n_objects_pasted > 0:
            self.log("dbg/oe/n_objects_pasted", float(batch_n_objects_pasted), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
        
        # P0 - Problema #2: Sanity check normalizzazione (solo primi 3 batch)
        if batch_idx < 3:
            # Forward pass per ottenere x dopo /255.0 (prima della normalizzazione in EoMT)
            x_pre = imgs.float() / 255.0
            
            # Log statistiche prima della normalizzazione
            logging.info(
                f"📊 P0 - Problema #2 - Batch {batch_idx}: Input dopo /255.0 (prima normalizzazione EoMT) - "
                f"x_min={x_pre.min().item():.4f}, "
                f"x_max={x_pre.max().item():.4f}, "
                f"x_mean={x_pre.mean().item():.4f}, "
                f"x_std={x_pre.std().item():.4f}"
            )
            
            # Log statistiche dopo normalizzazione (simulata usando pixel_mean/std di EoMT)
            if hasattr(self.network, 'encoder') and hasattr(self.network.encoder, 'pixel_mean'):
                pixel_mean = self.network.encoder.pixel_mean
                pixel_std = self.network.encoder.pixel_std
                x_normalized = (x_pre - pixel_mean) / pixel_std
                logging.info(
                    f"📊 P0 - Problema #2 - Batch {batch_idx}: Input dopo normalizzazione EoMT (simulata) - "
                    f"x_min={x_normalized.min().item():.4f}, "
                    f"x_max={x_normalized.max().item():.4f}, "
                    f"x_mean={x_normalized.mean().item():.4f}, "
                    f"x_std={x_normalized.std().item():.4f}"
                )
                # Log per canale (RGB)
                for c in range(3):
                    channel_mean = x_normalized[:, c].mean().item()
                    channel_std = x_normalized[:, c].std().item()
                    logging.info(
                        f"  Canale {c}: mean={channel_mean:.4f}, std={channel_std:.4f}"
                    )
        
        # Task 1: Print ood_mask for first 3 batches + overlay check
        if self._ood_mask_print_count < 3:
            # Check if any target has ood_mask
            for i, target in enumerate(targets):
                if "ood_mask" in target:
                    ood_mask = target["ood_mask"]
                    h, w = ood_mask.shape
                    
                    # Compute statistics
                    num_ood_pixels = (ood_mask == 1).sum().item()
                    num_id_pixels = (ood_mask == 0).sum().item()
                    num_ignore_pixels = (ood_mask == 255).sum().item()
                    total_pixels = h * w
                    ood_percentage = (num_ood_pixels / total_pixels) * 100
                    unique_values = torch.unique(ood_mask).tolist()
                    
                    logging.info(
                        f"📊 Task 1 - Batch {batch_idx}, Sample {i}: ood_mask - "
                        f"Shape: {h}x{w}, "
                        f"Unique values: {unique_values}, "
                        f"OOD pixels (1): {num_ood_pixels} ({ood_percentage:.2f}%), "
                        f"ID pixels (0): {num_id_pixels}, "
                        f"Ignore pixels (255): {num_ignore_pixels}"
                    )
                    
                    # PUNTO CRITICO #1: Overlay check - verify ood_mask alignment
                    # Create overlay visualization to verify ood_mask is aligned with image
                    try:
                        img_sample = imgs[i]  # [3, H, W] in range [0, 255]
                        # Convert image to numpy for visualization
                        img_np = img_sample.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, 3]
                        ood_mask_np = ood_mask.cpu().numpy()  # [H, W]
                        
                        # Create overlay: ood_mask in red over image
                        overlay = img_np.copy()
                        # Create red mask for OOD pixels (value == 1)
                        ood_region = (ood_mask_np == 1)
                        overlay[ood_region] = np.clip(
                            overlay[ood_region] * 0.5 + np.array([255, 0, 0]) * 0.5, 
                            0, 255
                        ).astype(np.uint8)
                        
                        # Log overlay as wandb image if logger is available
                        if hasattr(self, 'logger') and self.logger is not None:
                            try:
                                from PIL import Image as PILImage
                                overlay_pil = PILImage.fromarray(overlay)
                                # Log to wandb
                                self.logger.experiment.log({
                                    f"check/ood_mask_overlay_batch{batch_idx}_sample{i}": [wandb.Image(overlay_pil, caption=f"OOD mask overlay (red=OOD, batch {batch_idx}, sample {i})")]
                                })
                                logging.info(f"✅ Task 1: Logged ood_mask overlay for batch {batch_idx}, sample {i}")
                            except Exception as e:
                                logging.warning(f"⚠️ Task 1: Failed to log overlay: {e}")
                    except Exception as e:
                        logging.warning(f"⚠️ Task 1: Failed to create overlay check: {e}")
                    
                    self._ood_mask_print_count += 1
                    if self._ood_mask_print_count >= 3:
                        break

        mask_logits_per_block, class_logits_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits) in enumerate(
            list(zip(mask_logits_per_block, class_logits_per_block))
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses = {f"{key}{block_postfix}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        # Task 2: Log energy separated ID vs OOD
        # Use the last layer output for energy calculation
        if mask_logits_per_block and class_logits_per_block:
            mask_logits = mask_logits_per_block[-1]  # Last layer [B, Q, H, W]
            class_logits = class_logits_per_block[-1]  # Last layer [B, Q, C+1]
            
            # Check if any target has ood_mask (only for semantic segmentation with OE)
            if targets and "ood_mask" in targets[0]:
                # Interpolate mask_logits to img_size (same as in eval_step)
                mask_logits_interp = interpolate(mask_logits, self.img_size, mode="bilinear", align_corners=False)  # [B, Q, H, W]
                
                # Calculate pixel_scores (RAW logits) for energy computation
                # Use raw version (no softmax on class_logits) for correct energy formula
                pixel_logits = self.to_per_pixel_scores_semantic_raw(mask_logits_interp, class_logits)  # [B, C, H, W] - RAW logits
                
                # Calculate energy_map = -T*logsumexp(pixel_logits/T)
                # Use temperature T = 1.0 (same as energy loss)
                T = 1.0
                pixel_logits_scaled = pixel_logits / T  # [B, C, H, W]
                # Clamp for numerical stability
                pixel_logits_scaled = torch.clamp(pixel_logits_scaled, min=-50.0, max=50.0)
                energy_map = -T * torch.logsumexp(pixel_logits_scaled, dim=1)  # [B, H, W]
                
                # Process each sample in the batch
                for batch_idx_internal, target in enumerate(targets):
                    if "ood_mask" in target:
                        ood_mask = target["ood_mask"]  # [H, W]
                        energy = energy_map[batch_idx_internal]  # [H, W]
                        
                        # Filter energy by ood_mask: 0 = ID, 1 = OOD, 255 = ignore
                        # Exclude ignore pixels (255)
                        valid_mask = ood_mask != 255
                        energy_valid = energy[valid_mask]
                        ood_mask_valid = ood_mask[valid_mask]
                        
                        if energy_valid.numel() > 0:
                            # Separate energy for ID and OOD
                            energy_id = energy_valid[ood_mask_valid == 0]
                            energy_ood = energy_valid[ood_mask_valid == 1]
                            
                            # Log pixel counts (A1: OE Statistics)
                            n_id_pixels = energy_id.numel()
                            n_ood_pixels = energy_ood.numel()
                            self.log("dbg/n_id_pixels", float(n_id_pixels), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                            self.log("dbg/n_ood_pixels", float(n_ood_pixels), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                            self.log("dbg/oe/n_ood_pixels", float(n_ood_pixels), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)  # A1: Duplicato per coerenza
                            
                            # A3: Log mean energies (separazione ID vs OOD)
                            if energy_id.numel() > 0:
                                energy_id_mean = energy_id.mean().item()
                                energy_id_std = energy_id.std().item()
                                # Mantieni log legacy per compatibilità
                                self.log("dbg/energy_id_mean", energy_id_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                self.log("dbg/energy_id_std", energy_id_std, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                # A3: Log con prefisso dbg/energy/
                                self.log("dbg/energy/id_mean", energy_id_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                self.log("dbg/energy/id_std", energy_id_std, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                
                                # Log percentili ogni 200 step
                                current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else 0
                                if current_step % 200 == 0:
                                    energy_id_np = energy_id.detach().cpu().numpy()
                                    if len(energy_id_np) > 0:
                                        p10_id = float(np.percentile(energy_id_np, 10))
                                        p50_id = float(np.percentile(energy_id_np, 50))
                                        p90_id = float(np.percentile(energy_id_np, 90))
                                        self.log("dbg/energy_id_p10", p10_id, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy_id_p50", p50_id, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy_id_p90", p90_id, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                
                                # Task 4: Accumulate energy values for margin calculation
                                current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else len(self._energy_id_buffer)
                                if not self._energy_margins_computed and current_step < self._energy_margin_calculation_steps:
                                    # Accumulate individual values (sample to avoid memory issues)
                                    energy_id_values = energy_id.detach().cpu().numpy()
                                    # Sample up to 1000 values per batch to avoid memory explosion
                                    if len(energy_id_values) > 1000:
                                        indices = np.random.choice(len(energy_id_values), 1000, replace=False)
                                        energy_id_values = energy_id_values[indices]
                                    self._energy_id_buffer.extend(energy_id_values.tolist())
                            
                            if energy_ood.numel() > 0:
                                energy_ood_mean = energy_ood.mean().item()
                                energy_ood_std = energy_ood.std().item()
                                # Mantieni log legacy per compatibilità
                                self.log("dbg/energy_ood_mean", energy_ood_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                self.log("dbg/energy_ood_std", energy_ood_std, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                # A3: Log con prefisso dbg/energy/
                                self.log("dbg/energy/ood_mean", energy_ood_mean, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                self.log("dbg/energy/ood_std", energy_ood_std, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                
                                # Log percentili ogni 200 step
                                current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else 0
                                if current_step % 200 == 0:
                                    energy_ood_np = energy_ood.detach().cpu().numpy()
                                    if len(energy_ood_np) > 0:
                                        p10_ood = float(np.percentile(energy_ood_np, 10))
                                        p50_ood = float(np.percentile(energy_ood_np, 50))
                                        p90_ood = float(np.percentile(energy_ood_np, 90))
                                        self.log("dbg/energy_ood_p10", p10_ood, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy_ood_p50", p50_ood, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy_ood_p90", p90_ood, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
                                
                                # Task 4: Accumulate energy values for margin calculation
                                current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else len(self._energy_ood_buffer)
                                if not self._energy_margins_computed and current_step < self._energy_margin_calculation_steps:
                                    # Accumulate individual values (sample to avoid memory issues)
                                    energy_ood_values = energy_ood.detach().cpu().numpy()
                                    # Sample up to 1000 values per batch to avoid memory explosion
                                    if len(energy_ood_values) > 1000:
                                        indices = np.random.choice(len(energy_ood_values), 1000, replace=False)
                                        energy_ood_values = energy_ood_values[indices]
                                    self._energy_ood_buffer.extend(energy_ood_values.tolist())
                            
                            # A3: Log energy separation
                            if energy_id.numel() > 0 and energy_ood.numel() > 0:
                                energy_sep = energy_ood.mean() - energy_id.mean()
                                # Mantieni log legacy per compatibilità
                                self.log("dbg/energy_sep", energy_sep, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                # A3: Log con prefisso dbg/energy/
                                self.log("dbg/energy/sep", energy_sep, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                
                                # A2: Log energy loss components (loss_id, loss_ood, loss_total, weight, active)
                                # Use same formula as EnergyOODLoss: L_id = ReLU(E_id - m_in), L_ood = ReLU(m_out - E_ood)
                                if hasattr(self, 'criterion') and hasattr(self.criterion, 'energy_ood_loss'):
                                    try:
                                        base_loss = self.criterion.energy_ood_loss.base_loss if hasattr(self.criterion.energy_ood_loss, 'base_loss') else self.criterion.energy_ood_loss
                                        m_in = base_loss.m_in
                                        m_out = base_loss.m_out
                                        
                                        # Calculate loss components (RAW, prima di weight)
                                        loss_id = F.relu(energy_id - m_in).mean()
                                        loss_ood = F.relu(m_out - energy_ood).mean()
                                        loss_total_raw = loss_id + loss_ood
                                        
                                        # A2: Get current weight (dopo warmup)
                                        current_weight = 0.0
                                        if hasattr(self.criterion.energy_ood_loss, 'get_current_weight'):
                                            current_weight = self.criterion.energy_ood_loss.get_current_weight()
                                        
                                        # Apply weight if available
                                        if hasattr(base_loss, 'weight'):
                                            loss_id_weighted = loss_id * base_loss.weight
                                            loss_ood_weighted = loss_ood * base_loss.weight
                                            loss_total_weighted = loss_total_raw * base_loss.weight
                                        else:
                                            loss_id_weighted = loss_id
                                            loss_ood_weighted = loss_ood
                                            loss_total_weighted = loss_total_raw
                                        
                                        # Apply warmup weight if available
                                        if hasattr(self.criterion.energy_ood_loss, 'get_current_weight'):
                                            warmup_weight = self.criterion.energy_ood_loss.get_current_weight()
                                            if hasattr(self.criterion.energy_ood_loss, 'max_weight'):
                                                scale_factor = warmup_weight / self.criterion.energy_ood_loss.max_weight
                                                loss_id_weighted = loss_id_weighted * scale_factor
                                                loss_ood_weighted = loss_ood_weighted * scale_factor
                                                loss_total_weighted = loss_total_weighted * scale_factor
                                        
                                        # A2: Log energy loss statistics
                                        self.log("dbg/energy/weight", current_weight, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy/loss_raw", loss_total_raw.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("dbg/energy/loss_weighted", loss_total_weighted.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        
                                        # A2: active = 1 se loss_raw > 0 e n_ood_pixels > 0
                                        energy_active = 1.0 if (loss_total_raw.item() > 0.0 and n_ood_pixels > 0) else 0.0
                                        self.log("dbg/energy/active", energy_active, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        
                                        # Mantieni log legacy per compatibilità
                                        self.log("loss/energy_id", loss_id_weighted.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("loss/energy_ood", loss_ood_weighted.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                        self.log("loss/energy_total", loss_total_weighted.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=False)
                                    except Exception as e:
                                        # If calculation fails, skip logging but don't crash
                                        pass
                            
                            # Task 4: Compute margins after collecting enough statistics
                            current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else len(self._energy_id_buffer)
                            if (not self._energy_margins_computed and 
                                current_step >= self._energy_margin_calculation_steps and
                                len(self._energy_id_buffer) > 0 and 
                                len(self._energy_ood_buffer) > 0):
                                self._compute_and_update_margins()
                        
                        # Only process first sample in batch (avoid duplicate logs)
                        break

        total_loss = self.criterion.loss_total(losses_all_blocks, self.log)
        
        # Safety check: ensure total loss is finite before backprop
        # This prevents ComplexFloat errors in optimizer step
        if not torch.isfinite(total_loss):
            # If loss is NaN/Inf, return a small finite loss to avoid crash
            # This allows training to continue but skip this problematic batch
            total_loss = torch.tensor(1e-6, device=total_loss.device, dtype=total_loss.dtype, requires_grad=True)
            logging.warning(f"⚠️ Non-finite loss detected at batch {batch_idx}, using safe fallback loss")
        
        # Ensure loss is not complex (should never happen, but safety check)
        if total_loss.is_complex():
            total_loss = total_loss.real
            logging.warning(f"⚠️ Complex loss detected at batch {batch_idx}, using real part")
        
        return total_loss
    
    def _compute_and_update_margins(self):
        """
        Task 4: Compute real margins from accumulated energy distributions.
        
        Compute margins from percentiles:
        - m_in ~ P90(E_id) or mean_id + 1*std_id
        - m_out ~ P10(E_ood) or mean_ood - 1*std_ood
        """
        if len(self._energy_id_buffer) == 0 or len(self._energy_ood_buffer) == 0:
            logging.warning("⚠️ Task 4: Cannot compute margins - empty energy buffers")
            return
        
        # Convert to numpy arrays (numpy already imported at top of file)
        energy_id_array = np.array(self._energy_id_buffer)
        energy_ood_array = np.array(self._energy_ood_buffer)
        
        # Compute statistics
        energy_id_mean = float(np.mean(energy_id_array))
        energy_id_std = float(np.std(energy_id_array))
        energy_id_p90 = float(np.percentile(energy_id_array, 90))
        energy_id_p50 = float(np.percentile(energy_id_array, 50))
        energy_id_p10 = float(np.percentile(energy_id_array, 10))
        
        energy_ood_mean = float(np.mean(energy_ood_array))
        energy_ood_std = float(np.std(energy_ood_array))
        energy_ood_p90 = float(np.percentile(energy_ood_array, 90))
        energy_ood_p50 = float(np.percentile(energy_ood_array, 50))
        energy_ood_p10 = float(np.percentile(energy_ood_array, 10))
        
        # Log statistics
        current_step = self.trainer.global_step if hasattr(self, 'trainer') and self.trainer else 0
        logging.info(
            f"📊 Task 4 - Energy Statistics (step {current_step}):\n"
            f"  E_id: mean={energy_id_mean:.2f}, std={energy_id_std:.2f}, "
            f"P10={energy_id_p10:.2f}, P50={energy_id_p50:.2f}, P90={energy_id_p90:.2f}\n"
            f"  E_ood: mean={energy_ood_mean:.2f}, std={energy_ood_std:.2f}, "
            f"P10={energy_ood_p10:.2f}, P50={energy_ood_p50:.2f}, P90={energy_ood_p90:.2f}"
        )
        
        # Compute margins using percentiles (preferred) or mean+std (quick)
        # Option 1: Use percentiles (preferred)
        m_in_candidate = energy_id_p90  # P90 of ID energy
        m_out_candidate = energy_ood_p10  # P10 of OOD energy
        
        # Option 2: Use mean + 1*std (alternative)
        # m_in_candidate = energy_id_mean + energy_id_std
        # m_out_candidate = energy_ood_mean - energy_ood_std
        
        # Ensure m_out > m_in
        if m_out_candidate <= m_in_candidate:
            # If separation is not clear, use a small gap
            gap = max(1.0, (energy_ood_mean - energy_id_mean) * 0.1)
            m_out_candidate = m_in_candidate + gap
            logging.warning(
                f"⚠️ Task 4: Energy separation not clear (m_out <= m_in), "
                f"using gap={gap:.2f}. This indicates the model is not separating ID/OOD yet."
            )
        
        # Update margins in energy loss if available
        if hasattr(self, 'criterion') and hasattr(self.criterion, 'energy_ood_loss'):
            try:
                self.criterion.energy_ood_loss.base_loss.update_margins(m_in_candidate, m_out_candidate)
                logging.info(
                    f"✅ Task 4: Updated energy margins - m_in={m_in_candidate:.2f}, m_out={m_out_candidate:.2f}"
                )
                # Mark as computed
                self._energy_margins_computed = True
                # Clear buffers to save memory
                self._energy_id_buffer.clear()
                self._energy_ood_buffer.clear()
            except Exception as e:
                logging.warning(f"⚠️ Task 4: Failed to update margins: {e}")
        else:
            logging.warning("⚠️ Task 4: Cannot update margins - energy_ood_loss not available")
    
    def on_before_optimizer_step(self, optimizer):
        """
        Hook called before optimizer step to ensure parameters and gradients are real.
        This prevents ComplexFloat errors during optimizer step.
        """
        # Check all trainable parameters and gradients for complex values
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check if parameter is complex
            if param.is_complex():
                logging.warning(f"⚠️ Complex parameter detected: {name}, converting to real")
                # Convert complex parameter to real (take real part)
                param.data = param.data.real
            
            # Check if gradient exists and is complex
            if param.grad is not None:
                if param.grad.is_complex():
                    logging.warning(f"⚠️ Complex gradient detected: {name}, converting to real")
                    # Convert complex gradient to real (take real part)
                    param.grad = param.grad.real
                
                # Check if gradient contains NaN/Inf
                if not torch.isfinite(param.grad).all():
                    logging.warning(f"⚠️ Non-finite gradient detected: {name}, zeroing out")
                    # Zero out non-finite gradients
                    param.grad = torch.where(
                        torch.isfinite(param.grad),
                        param.grad,
                        torch.zeros_like(param.grad)
                    )
                
                # Additional safety: clamp gradient to reasonable range
                # More aggressive clamping for class_head (most sensitive to numerical issues)
                if "class_head" in name:
                    # Class head is sensitive - use tighter clamping
                    torch.clamp_(param.grad, min=-50.0, max=50.0)
                else:
                    # Other parameters can tolerate slightly larger gradients
                    torch.clamp_(param.grad, min=-100.0, max=100.0)
            
            # Check if parameter contains NaN/Inf
            if not torch.isfinite(param.data).all():
                logging.warning(f"⚠️ Non-finite parameter detected: {name}, replacing with zeros")
                # Replace non-finite parameters with zeros (shouldn't happen, but safety)
                param.data = torch.where(
                    torch.isfinite(param.data),
                    param.data,
                    torch.zeros_like(param.data)
                )

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        device = self.device
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if self.attn_mask_annealing_enabled:
            for i in range(self.network.num_blocks):
                self.network.attn_mask_probs[i] = self.mask_annealing(
                    self.attn_mask_annealing_start_steps[i],
                    self.global_step,
                    self.attn_mask_annealing_end_steps[i],
                )

            for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
                self.log(
                    f"attn_mask_prob_{i}",
                    attn_mask_prob,
                    on_step=True,
                )

    def init_metrics_semantic(self, ignore_idx, num_blocks):
        self.metrics = nn.ModuleList(
            [
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    validate_args=False,
                    ignore_index=ignore_idx,
                    average=None,
                )
                for _ in range(num_blocks)
            ]
        )

    def init_metrics_instance(self, num_blocks):
        self.metrics = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )

    def init_metrics_panoptic(self, thing_classes, stuff_classes, num_blocks):
        self.metrics = nn.ModuleList(
            [
                PanopticQuality(
                    thing_classes,
                    stuff_classes + [self.num_classes],
                    return_sq_and_rq=True,
                    return_per_class=True,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.compiler.disable
    def update_metrics_semantic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            self.metrics[block_idx].update(preds[i][None, ...], targets[i][None, ...])

    @torch.compiler.disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx,
    ):
        self.metrics[block_idx].update(preds, targets)

    @torch.compiler.disable
    def update_metrics_panoptic(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor],
        is_crowds: list[torch.Tensor],
        block_idx,
    ):
        for i in range(len(preds)):
            metric = self.metrics[block_idx]
            flatten_pred = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                preds[i][None, ...],
                metric.void_color,
                metric.allow_unknown_preds_category,
            )[0]
            flatten_target = _prepocess_inputs(
                metric.things,
                metric.stuffs,
                targets[i][None, ...],
                metric.void_color,
                True,
            )[0]

            pred_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_pred)
            )
            target_areas = cast(
                dict[_Color, torch.Tensor], _get_color_areas(flatten_target)
            )
            intersection_matrix = torch.transpose(
                torch.stack((flatten_pred, flatten_target), -1), -1, -2
            )
            intersection_areas = cast(
                dict[tuple[_Color, _Color], torch.Tensor],
                _get_color_areas(intersection_matrix),
            )

            pred_segment_matched = set()
            target_segment_matched = set()
            for pred_color, target_color in intersection_areas:
                if is_crowds[i][target_color[1]]:
                    continue
                if target_color == metric.void_color:
                    continue
                if pred_color[0] != target_color[0]:
                    continue
                iou = _calculate_iou(
                    pred_color,
                    target_color,
                    pred_areas,
                    target_areas,
                    intersection_areas,
                    metric.void_color,
                )
                continuous_id = metric.cat_id_to_continuous_id[target_color[0]]
                if iou > 0.5:
                    pred_segment_matched.add(pred_color)
                    target_segment_matched.add(target_color)
                    metric.iou_sum[continuous_id] += iou
                    metric.true_positives[continuous_id] += 1

            false_negative_colors = set(target_areas) - target_segment_matched
            false_positive_colors = set(pred_areas) - pred_segment_matched

            false_negative_colors.discard(metric.void_color)
            false_positive_colors.discard(metric.void_color)

            for target_color in list(false_negative_colors):
                void_target_area = intersection_areas.get(
                    (metric.void_color, target_color), 0
                )
                if void_target_area / target_areas[target_color] > 0.5:
                    false_negative_colors.discard(target_color)

            crowd_by_cat_id = {}
            for false_negative_color in false_negative_colors:
                if is_crowds[i][false_negative_color[1]]:
                    crowd_by_cat_id[false_negative_color[0]] = false_negative_color[1]
                    continue

                continuous_id = metric.cat_id_to_continuous_id[false_negative_color[0]]
                metric.false_negatives[continuous_id] += 1

            for pred_color in list(false_positive_colors):
                pred_void_crowd_area = intersection_areas.get(
                    (pred_color, metric.void_color), 0
                )

                if pred_color[0] in crowd_by_cat_id:
                    crowd_color = (pred_color[0], crowd_by_cat_id[pred_color[0]])
                    pred_void_crowd_area += intersection_areas.get(
                        (pred_color, crowd_color), 0
                    )

                if pred_void_crowd_area / pred_areas[pred_color] > 0.5:
                    false_positive_colors.discard(pred_color)

            for false_positive_color in false_positive_colors:
                continuous_id = metric.cat_id_to_continuous_id[false_positive_color[0]]
                metric.false_positives[continuous_id] += 1

    def block_postfix(self, block_idx):
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_semantic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            iou_per_class = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx, iou in enumerate(iou_per_class):
                    self.log(
                        f"metrics/{log_prefix}_iou_class_{class_idx}{block_postfix}",
                        iou,
                    )

            iou_all = float(iou_per_class.mean())
            self.log(
                f"metrics/{log_prefix}_iou_all{block_postfix}",
                iou_all,
            )

    def _on_eval_epoch_end_instance(self, log_prefix):
        for i, metric in enumerate(self.metrics):  # type: ignore
            results = metric.compute()
            metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(
                f"metrics/{log_prefix}_ap_all{block_postfix}",
                results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                results["map_75"],
            )

    def _on_eval_epoch_end_panoptic(self, log_prefix, log_per_class=False):
        for i, metric in enumerate(self.metrics):  # type: ignore
            result = metric.compute()[:-1]
            metric.reset()

            pq, sq, rq = result[:, 0], result[:, 1], result[:, 2]

            block_postfix = self.block_postfix(i)
            if log_per_class:
                for class_idx in range(len(pq)):
                    self.log(
                        f"metrics/{log_prefix}_pq_class_{class_idx}{block_postfix}",
                        pq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_sq_class_{class_idx}{block_postfix}",
                        sq[class_idx],
                    )
                    self.log(
                        f"metrics/{log_prefix}_rq_class_{class_idx}{block_postfix}",
                        rq[class_idx],
                    )

            self.log(
                f"metrics/{log_prefix}_pq_all{block_postfix}",
                pq.mean(),
            )
            self.log(f"metrics/{log_prefix}_sq_all{block_postfix}", sq.mean())
            self.log(f"metrics/{log_prefix}_rq_all{block_postfix}", rq.mean())

            num_things = len(metric.things)
            pq_things, sq_things, rq_things = (
                result[:num_things, 0],
                result[:num_things, 1],
                result[:num_things, 2],
            )
            pq_stuff, sq_stuff, rq_stuff = (
                result[num_things:, 0],
                result[num_things:, 1],
                result[num_things:, 2],
            )

            self.log(
                f"metrics/{log_prefix}_pq_things{block_postfix}",
                pq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_things{block_postfix}",
                sq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_things{block_postfix}",
                rq_things.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_pq_stuff{block_postfix}",
                pq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_sq_stuff{block_postfix}",
                sq_stuff.mean(),
            )
            self.log(
                f"metrics/{log_prefix}_rq_stuff{block_postfix}",
                rq_stuff.mean(),
            )

    def _on_eval_end_semantic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mIoU: {self.trainer.callback_metrics[f'metrics/{log_prefix}_iou_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_instance(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
                f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
                f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
                f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f}{reset}"
            )

    def _on_eval_end_panoptic(self, log_prefix):
        if not self.trainer.sanity_checking:
            rank_zero_info(
                f"{bold_green}PQ All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_all'] * 100:.1f} | "
                f"PQ Things: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_things'] * 100:.1f} | "
                f"PQ Stuff: {self.trainer.callback_metrics[f'metrics/{log_prefix}_pq_stuff'] * 100:.1f}{reset}"
            )

    @torch.compiler.disable
    def plot_semantic(
        self,
        img,
        target,
        logits,
        log_prefix,
        block_idx,
        batch_idx,
        cmap="tab20",
    ):
        fig, axes = plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)

        axes[0].imshow(img.cpu().numpy().transpose(1, 2, 0))
        axes[0].axis("off")

        target = target.cpu().numpy()
        unique_classes = np.unique(target)

        preds = torch.argmax(logits, dim=0).cpu().numpy()
        unique_classes = np.unique(np.concatenate((unique_classes, np.unique(preds))))

        num_classes = len(unique_classes)
        colors = plt.get_cmap(cmap, num_classes)(np.linspace(0, 1, num_classes))  # type: ignore

        if self.ignore_idx in unique_classes:
            colors[unique_classes == self.ignore_idx] = [0, 0, 0, 1]  # type: ignore

        custom_cmap = mcolors.ListedColormap(colors)  # type: ignore
        norm = mcolors.Normalize(0, num_classes - 1)

        axes[1].imshow(
            np.digitize(target, unique_classes) - 1,
            cmap=custom_cmap,
            norm=norm,
            interpolation="nearest",
        )
        axes[1].axis("off")

        if preds is not None:
            axes[2].imshow(
                np.digitize(preds, unique_classes, right=True),
                cmap=custom_cmap,
                norm=norm,
                interpolation="nearest",
            )
            axes[2].axis("off")

        patches = [
            Line2D([0], [0], color=colors[i], lw=4, label=str(unique_classes[i]))
            for i in range(num_classes)
        ]

        fig.legend(handles=patches, loc="upper left")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, facecolor="black")
        plt.close(fig)
        buf.seek(0)

        block_postfix = self.block_postfix(block_idx)
        name = f"{log_prefix}_pred_{batch_idx}{block_postfix}"
        self.trainer.logger.experiment.log({name: [wandb.Image(Image.open(buf))]})

    @torch.compiler.disable
    def scale_img_size_semantic(self, size: tuple[int, int]):
        factor = max(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def window_imgs_semantic(self, imgs):
        crops, origins = [], []

        for i in range(len(imgs)):
            img = imgs[i]
            new_h, new_w = self.scale_img_size_semantic(img.shape[-2:])
            # Fix PIL error: convert tensor to uint8 numpy array before Image.fromarray
            # PIL expects uint8 [0-255], not float32
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.dtype in (np.float32, np.float64):
                # Assume values are in [0, 1] range, convert to [0, 255]
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).clip(0, 255)
                img_np = img_np.astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode="RGB")
            resized_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(resized_img)).permute(2, 0, 1).to(img.device)
            )

            num_crops = math.ceil(max(resized_img.shape[-2:]) / min(self.img_size))
            overlap = num_crops * min(self.img_size) - max(resized_img.shape[-2:])
            overlap_per_crop = (overlap / (num_crops - 1)) if overlap > 0 else 0

            for j in range(num_crops):
                start = int(j * (min(self.img_size) - overlap_per_crop))
                end = start + min(self.img_size)
                if resized_img.shape[-2] > resized_img.shape[-1]:
                    crop = resized_img[:, start:end, :]
                else:
                    crop = resized_img[:, :, start:end]

                crops.append(crop)
                origins.append((i, start, end))

        return torch.stack(crops), origins

    def revert_window_logits_semantic(self, crop_logits, origins, img_sizes):
        logit_sums, logit_counts = [], []
        for size in img_sizes:
            h, w = self.scale_img_size_semantic(size)
            logit_sums.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )
            logit_counts.append(
                torch.zeros((crop_logits.shape[1], h, w), device=crop_logits.device)
            )

        for crop_i, (img_i, start, end) in enumerate(origins):
            if img_sizes[img_i][0] > img_sizes[img_i][1]:
                logit_sums[img_i][:, start:end, :] += crop_logits[crop_i]
                logit_counts[img_i][:, start:end, :] += 1
            else:
                logit_sums[img_i][:, :, start:end] += crop_logits[crop_i]
                logit_counts[img_i][:, :, start:end] += 1

        return [
            interpolate(
                (sums / counts)[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            for i, (sums, counts) in enumerate(zip(logit_sums, logit_counts))
        ]

    @staticmethod
    def to_per_pixel_logits_semantic(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),
            class_logits.softmax(dim=-1)[..., :-1],
        )
    
    @staticmethod
    def to_per_pixel_scores_semantic_raw(
        mask_logits: torch.Tensor, class_logits: torch.Tensor
    ):
        """
        Convert query-level logits to per-pixel semantic scores using RAW class logits.
        
        This is used for energy computation, which requires raw logits (not probabilities).
        
        Args:
            mask_logits: [B, Q, H, W] - mask predictions per query
            class_logits: [B, Q, C+1] - class predictions per query (C classes + 1 no-object)
        
        Returns:
            pixel_scores: [B, C, H, W] - per-pixel scores for C semantic classes (raw logits, not probabilities)
        """
        # mask: sigmoid ok (probabilità di presenza mask)
        # class: RAW logits, exclude no-object
        return torch.einsum(
            "bqhw, bqc -> bchw",
            mask_logits.sigmoid(),  # [B, Q, H, W] -> probabilità mask
            class_logits[..., :-1],  # [B, Q, C] -> RAW logits (no softmax, exclude no-object)
        )

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_semantic(
        targets: list[dict],
        ignore_idx,
    ):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = torch.full(
                target["masks"].shape[-2:],
                ignore_idx,
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[mask] = target["labels"][i]

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def scale_img_size_instance_panoptic(self, size: tuple[int, int]):
        factor = min(
            self.img_size[0] / size[0],
            self.img_size[1] / size[1],
        )

        return [round(s * factor) for s in size]

    @torch.compiler.disable
    def resize_and_pad_imgs_instance_panoptic(self, imgs):
        transformed_imgs = []

        for img in imgs:
            new_h, new_w = self.scale_img_size_instance_panoptic(img.shape[-2:])

            # Fix PIL error: convert tensor to uint8 numpy array before Image.fromarray
            # PIL expects uint8 [0-255], not float32
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.dtype in (np.float32, np.float64):
                # Assume values are in [0, 1] range, convert to [0, 255]
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255.0).clip(0, 255)
                img_np = img_np.astype(np.uint8)
            pil_img = Image.fromarray(img_np, mode="RGB")
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
            resized_img = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(img.device)
            )

            pad_h = max(0, self.img_size[-2] - resized_img.shape[-2])
            pad_w = max(0, self.img_size[-1] - resized_img.shape[-1])
            padding = [0, 0, pad_w, pad_h]

            padded_img = pad(resized_img, padding)

            transformed_imgs.append(padded_img)

        return torch.stack(transformed_imgs)

    @torch.compiler.disable
    def revert_resize_and_pad_logits_instance_panoptic(
        self, transformed_logits, img_sizes
    ):
        logits = []
        for i in range(len(transformed_logits)):
            scaled_size = self.scale_img_size_instance_panoptic(img_sizes[i])
            logits_i = transformed_logits[i][:, : scaled_size[0], : scaled_size[1]]
            logits_i = interpolate(
                logits_i[None, ...],
                img_sizes[i],
                mode="bilinear",
            )[0]
            logits.append(logits_i)

        return logits

    def to_per_pixel_preds_panoptic(
        self, mask_logits_list, class_logits, stuff_classes, mask_thresh, overlap_thresh
    ):
        scores, classes = class_logits.softmax(dim=-1).max(-1)
        preds_list = []

        for i in range(len(mask_logits_list)):
            preds = -torch.ones(
                (*mask_logits_list[i].shape[-2:], 2),
                dtype=torch.long,
                device=class_logits.device,
            )
            preds[:, :, 0] = self.num_classes

            keep = classes[i].ne(class_logits.shape[-1] - 1) & (scores[i] > mask_thresh)
            if not keep.any():
                preds_list.append(preds)
                continue

            masks = mask_logits_list[i].sigmoid()
            segments = -torch.ones(
                *masks.shape[-2:],
                dtype=torch.long,
                device=class_logits.device,
            )

            mask_ids = (scores[i][keep][..., None, None] * masks[keep]).argmax(0)
            stuff_segment_ids, segment_id = {}, 0
            segment_and_class_ids = []

            for k, class_id in enumerate(classes[i][keep].tolist()):
                orig_mask = masks[keep][k] >= 0.5
                new_mask = mask_ids == k
                final_mask = orig_mask & new_mask

                orig_area = orig_mask.sum().item()
                new_area = new_mask.sum().item()
                final_area = final_mask.sum().item()
                if (
                    orig_area == 0
                    or new_area == 0
                    or final_area == 0
                    or new_area / orig_area < overlap_thresh
                ):
                    continue

                if class_id in stuff_classes:
                    if class_id in stuff_segment_ids:
                        segments[final_mask] = stuff_segment_ids[class_id]
                        continue
                    else:
                        stuff_segment_ids[class_id] = segment_id

                segments[final_mask] = segment_id
                segment_and_class_ids.append((segment_id, class_id))

                segment_id += 1

            for segment_id, class_id in segment_and_class_ids:
                segment_mask = segments == segment_id
                preds[:, :, 0] = torch.where(segment_mask, class_id, preds[:, :, 0])
                preds[:, :, 1] = torch.where(segment_mask, segment_id, preds[:, :, 1])

            preds_list.append(preds)

        return preds_list

    @staticmethod
    @torch.compiler.disable
    def to_per_pixel_targets_panoptic(targets: list[dict]):
        per_pixel_targets = []
        for target in targets:
            per_pixel_target = -torch.ones(
                (*target["masks"].shape[-2:], 2),
                dtype=target["labels"].dtype,
                device=target["labels"].device,
            )

            for i, mask in enumerate(target["masks"]):
                per_pixel_target[:, :, 0] = torch.where(
                    mask, target["labels"][i], per_pixel_target[:, :, 0]
                )

                per_pixel_target[:, :, 1] = torch.where(
                    mask,
                    torch.tensor(i, device=target["masks"].device),
                    per_pixel_target[:, :, 1],
                )

            per_pixel_targets.append(per_pixel_target)

        return per_pixel_targets

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }
    
    def on_load_checkpoint(self, checkpoint):
        """
        Hook chiamato quando Lightning carica un checkpoint automaticamente (resume).
        Pulisce parametri complessi/NaN/Inf per prevenire ComplexFloat errors.
        Gestisce anche il caso in cui il checkpoint contiene solo i pesi del modello (senza optimizer state).
        """
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cleaned_keys = 0
            
            for k, v in state_dict.items():
                if not isinstance(v, torch.Tensor):
                    continue
                
                # Convert complex to real (take real part)
                if v.is_complex():
                    checkpoint["state_dict"][k] = v.real
                    cleaned_keys += 1
                    logging.warning(f"⚠️ Cleaned complex parameter in checkpoint: {k}")
                
                # Replace NaN/Inf with zeros
                if not torch.isfinite(v).all():
                    num_invalid = (~torch.isfinite(v)).sum().item()
                    checkpoint["state_dict"][k] = torch.where(
                        torch.isfinite(v), v, torch.zeros_like(v)
                    )
                    cleaned_keys += 1
                    logging.warning(f"⚠️ Cleaned {num_invalid} NaN/Inf values in checkpoint: {k}")
                
                # Ensure tensor is real float (not complex)
                if not checkpoint["state_dict"][k].dtype.is_floating_point:
                    checkpoint["state_dict"][k] = checkpoint["state_dict"][k].float()
            
            if cleaned_keys > 0:
                logging.warning(f"⚠️ Cleaned {cleaned_keys} parameters with complex/NaN/Inf values in checkpoint")
        
        # Gestisci checkpoint con solo pesi modello (senza optimizer state)
        # Se il checkpoint non ha optimizer_states, rimuoviamo le chiavi correlate per evitare errori
        # Questo permette di riprendere il training da checkpoint con solo pesi (riparti da capo con nuovo optimizer)
        has_optimizer_state = (
            "optimizer_states" in checkpoint 
            and checkpoint.get("optimizer_states") is not None
            and len(checkpoint.get("optimizer_states", [])) > 0
        )
        
        if not has_optimizer_state:
            # Checkpoint contiene solo pesi del modello → rimuoviamo eventuali chiavi optimizer/scheduler per evitare errori
            if "optimizer_states" in checkpoint:
                logging.warning("⚠️ Checkpoint contains only model weights (no optimizer state) → will re-initialize optimizer")
                del checkpoint["optimizer_states"]
            if "lr_schedulers" in checkpoint:
                del checkpoint["lr_schedulers"]
            if "epoch" in checkpoint:
                # Reset epoch to 0 per ripartire da capo
                logging.info("ℹ️ Resetting epoch to 0 (checkpoint has only model weights)")
                checkpoint["epoch"] = 0
            if "global_step" in checkpoint:
                checkpoint["global_step"] = 0
            logging.info("✅ Checkpoint prepared for restart (only model weights loaded, optimizer will be re-initialized)")
        else:
            # Checkpoint HA optimizer state → mantienili intatti per resume completo
            logging.info("✅ Checkpoint contains optimizer state → will resume training with full state")

    def _zero_init_outside_encoder(
        self, encoder_prefix="network.encoder.", skip_class_head=False
    ):
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, p in self.named_parameters():
                total += p.numel()
                if not name.startswith(encoder_prefix):
                    if skip_class_head and (
                        "class_head" in name or "class_predictor" in name
                    ):
                        continue
                    p.zero_()
                    zeroed += p.numel()
            msg = f"Zeroed {zeroed:,} / {total:,} parameters (everything not under '{encoder_prefix}'"
            if skip_class_head:
                msg += ", skipping class head"
            msg += ")"
            logging.info(msg)

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for k in state_dict1.keys():
            if k not in state_dict2:
                raise KeyError(f"Key {k} not found in second state_dict")

            if state_dict1[k].shape != state_dict2[k].shape:
                raise ValueError(
                    f"Shape mismatch at {k}: "
                    f"{state_dict1[k].shape} vs {state_dict2[k].shape}"
                )

            summed[k] = state_dict1[k] + state_dict2[k]

        return summed

    def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
        """
        Load checkpoint and clean complex/NaN/Inf values from model weights.
        This prevents ComplexFloat errors when resuming from corrupted checkpoints.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        
        # CLEAN COMPLEX/NaN/Inf VALUES: Convert complex to real, replace NaN/Inf with zeros
        # This is critical when resuming from checkpoints that may contain corrupted optimizer state
        cleaned_keys = 0
        for k, v in ckpt.items():
            if not isinstance(v, torch.Tensor):
                continue
            
            original_dtype = v.dtype
            
            # Convert complex to real (take real part)
            if v.is_complex():
                ckpt[k] = v.real
                cleaned_keys += 1
                logging.warning(f"⚠️ Cleaned complex parameter: {k}")
            
            # Replace NaN/Inf with zeros (shouldn't happen, but safety check)
            if not torch.isfinite(v).all():
                num_invalid = (~torch.isfinite(v)).sum().item()
                ckpt[k] = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                cleaned_keys += 1
                logging.warning(f"⚠️ Cleaned {num_invalid} NaN/Inf values in: {k}")
            
            # Ensure tensor is real float (not complex)
            if not ckpt[k].dtype.is_floating_point:
                ckpt[k] = ckpt[k].float()
        
        if cleaned_keys > 0:
            logging.warning(f"⚠️ Cleaned {cleaned_keys} parameters with complex/NaN/Inf values")
        
        # Log which keys were loaded (especially important for debugging)
        class_head_keys = [k for k in ckpt.keys() if "class_head" in k]
        if class_head_keys:
            logging.info(f"✅ Found class_head weights in checkpoint: {class_head_keys}")
        else:
            if load_ckpt_class_head:
                logging.warning("⚠️ class_head weights NOT found in checkpoint!")
                logging.warning(f"   Checkpoint keys (sample): {list(ckpt.keys())[:10]}...")
        
        logging.info(f"Loaded {len(ckpt)} keys from checkpoint")
        return ckpt

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "class_head" not in key and "class_predictor" not in key
                ]
            else:
                missing_keys = incompatible_keys.missing_keys
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")
        if incompatible_keys.unexpected_keys:
            raise ValueError(f"Unexpected keys: {incompatible_keys.unexpected_keys}")
