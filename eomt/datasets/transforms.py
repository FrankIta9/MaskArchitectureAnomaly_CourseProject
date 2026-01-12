# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Union, Optional


def _to_1d_tensor(x: Any, dtype=None, device=None) -> Tensor:
    """
    Normalizza input a Tensor 1D.
    Gestisce sia Tensor [N] che liste di Tensor scalari o scalari Python.
    
    Args:
        x: Input che può essere Tensor [N], lista di Tensor/scalari, o scalare
        dtype: Dtype target (opzionale)
        device: Device target (opzionale)
    
    Returns:
        Tensor 1D [N]
    """
    # Se è una lista, convertila in Tensor
    if isinstance(x, list):
        if len(x) == 0:
            # Lista vuota → Tensor vuoto
            if dtype is not None:
                return torch.empty(0, dtype=dtype, device=device) if device else torch.empty(0, dtype=dtype)
            return torch.empty(0, device=device) if device else torch.empty(0)
        # Stack degli elementi (gestisce sia Tensor che scalari)
        x = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in x])
    
    # Se è un Tensor, assicurati che sia almeno 1D
    if torch.is_tensor(x):
        if x.dim() == 0:
            x = x.unsqueeze(0)
        # Converti dtype/device se specificati
        if dtype is not None:
            x = x.to(dtype=dtype)
        if device is not None:
            x = x.to(device=device)
        return x
    
    # Fallback: se è uno scalare Python, convertilo a Tensor
    return torch.tensor([x], dtype=dtype, device=device)


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        outlier_exposure_transform: Optional[nn.Module] = None,
    ):
        """
        Args:
            outlier_exposure_transform: Optional OutlierExposureTransform to apply
        """
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)
        
        # Outlier Exposure (optional)
        self.outlier_exposure_transform = outlier_exposure_transform

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        H0, W0 = img.shape[-2], img.shape[-1]
        pad_h = max(0, self.img_size[-2] - H0)
        pad_w = max(0, self.img_size[-1] - W0)
        padding = [0, 0, pad_w, pad_h]

        # Pad img con 0 (nero)
        img = F.pad(img, padding, fill=0)
        
        # Pad masks con 0 (va bene per boolean/uint8)
        target["masks"] = F.pad(target["masks"], padding, fill=0)
        
        # Pad semseg con 255 (IGNORE) - questo assicura che padding sia sempre IGNORE
        if "semseg" in target:
            semseg = target["semseg"]
            target["semseg"] = F.pad(semseg, padding, fill=255)

        # NEW: valid_mask - 1 dove c'era immagine originale, 0 dove è padding
        valid = torch.zeros((self.img_size[-2], self.img_size[-1]), dtype=torch.bool, device=img.device)
        valid[:H0, :W0] = True
        target["valid_mask"] = valid

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        """
        Filter target entries by boolean mask.
        
        🔧 P0 Fix: Normalizza labels/is_crowd a Tensor prima di filtrare,
        per evitare crash quando diventano liste durante le trasformazioni.
        
        Args:
            target: Target dictionary
            keep: Boolean mask with shape [N] for filtering arrays of masks/labels
            
        Returns:
            Filtered target dictionary
            
        Note:
            Fields like 'ood_mask' (pixel-wise [H, W]) are excluded from filtering
            as they are not arrays of masks but single pixel-wise masks.
        """
        filtered = {}
        # Fields to exclude from filtering (pixel-wise, not arrays)
        exclude_from_filter = {"ood_mask", "semseg", "valid_mask"}  # pixel-wise masks [H, W], not [N, H, W]
        
        # 🔧 P0 Fix: Normalizza labels/is_crowd a Tensor prima di filtrare
        # Questo evita crash quando diventano liste durante le trasformazioni
        if "labels" in target:
            target["labels"] = _to_1d_tensor(target["labels"], dtype=torch.long)
        if "is_crowd" in target:
            target["is_crowd"] = _to_1d_tensor(target["is_crowd"], dtype=torch.bool)
        
        for k, v in target.items():
            if k in exclude_from_filter:
                # Skip filtering for pixel-wise fields
                filtered[k] = v
            else:
                # Apply filtering for array fields (masks, labels, is_crowd)
                if isinstance(v, (list, tuple)):
                    # For Python lists/tuples, use list comprehension with keep_indices
                    keep_indices = keep.nonzero(as_tuple=False).squeeze(-1).tolist() if isinstance(keep, Tensor) else keep
                    filtered[k] = [v[i] for i in keep_indices]
                elif torch.is_tensor(v):
                    # Handle tensors directly (labels/is_crowd ora sono Tensor garantiti)
                    filtered[k] = v[keep]
                else:
                    # Fallback: use wrap for TVTensors or other types
                    filtered[k] = wrap(v[keep], like=v)
        
        return filtered

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)
        
        # Apply Outlier Exposure (cut-paste) if enabled
        # Apply before filtering to ensure anomaly masks are included
        if self.outlier_exposure_transform is not None:
            img, target = self.outlier_exposure_transform(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target
