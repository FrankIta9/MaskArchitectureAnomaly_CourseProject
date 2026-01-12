#!/usr/bin/env python3
"""
6) Visualizza 1 batch (overlay) e salva su disco
Test per verificare che ood_mask sia presente dopo tutte le trasformazioni.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eomt.datasets.cityscapes_semantic_with_oe import CityscapesSemanticWithOE
from eomt.configs.dinov2.cityscapes.semantic.eomt_base_1024_oe_safe import (
    get_config as get_safe_config
)

def visualize_ood_overlay(img, ood_mask, save_path):
    """
    Visualizza immagine con overlay rosso per pixel OOD.
    
    Args:
        img: Tensor [C, H, W] o [H, W, C] in range [0, 1] o [0, 255]
        ood_mask: Tensor [H, W] con 1=OOD, 0=ID
        save_path: Path dove salvare l'immagine
    """
    # Converti img a numpy
    if torch.is_tensor(img):
        img_np = img.detach().cpu().numpy()
        # Gestisci shape [C, H, W] vs [H, W, C]
        if img_np.shape[0] == 3 or img_np.shape[0] == 1:
            img_np = img_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        # Normalizza a [0, 1] se necessario
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(img)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    
    # Converti ood_mask a numpy
    if torch.is_tensor(ood_mask):
        ood_mask_np = ood_mask.detach().cpu().numpy()
    else:
        ood_mask_np = np.array(ood_mask)
    
    # Assicurati che ood_mask sia [H, W]
    if ood_mask_np.ndim > 2:
        ood_mask_np = ood_mask_np.squeeze()
    
    # Crea overlay rosso
    overlay = img_np.copy()
    if overlay.shape[2] == 1:
        overlay = np.repeat(overlay, 3, axis=2)  # Grayscale -> RGB
    
    # Overlay rosso per pixel OOD (alpha blending)
    red_overlay = np.zeros_like(overlay)
    red_overlay[:, :, 0] = 1.0  # Rosso puro
    alpha = ood_mask_np.astype(np.float32)[:, :, np.newaxis] * 0.5  # 50% opacity
    overlay = overlay * (1 - alpha) + red_overlay * alpha
    
    # Visualizza
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Immagine originale
    axes[0].imshow(img_np if img_np.shape[2] == 3 else np.repeat(img_np, 3, axis=2))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # OOD mask
    axes[1].imshow(ood_mask_np, cmap='Reds', vmin=0, vmax=1)
    axes[1].set_title(f"OOD Mask (sum={ood_mask_np.sum()})")
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (OOD pixels in red)")
    axes[2].axis('off')
    
    # Legenda
    red_patch = mpatches.Patch(color='red', label='OOD pixels')
    axes[2].legend(handles=[red_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualization to {save_path}")
    print(f"   OOD pixels: {ood_mask_np.sum()}")
    print(f"   OOD ratio: {ood_mask_np.sum() / ood_mask_np.size:.6f}")

def main():
    print("="*70)
    print("6) VISUALIZZAZIONE BATCH CON OVERLAY OOD")
    print("="*70)
    
    # Carica config
    config = get_safe_config()
    data_config = config.data.init_args
    
    # Crea dataset
    dataset = CityscapesSemanticWithOE(**data_config)
    dataset.setup()
    
    # Prendi primo batch
    train_loader = dataset.train_dataloader()
    batch = next(iter(train_loader))
    
    # Estrai primo sample del batch
    if isinstance(batch, (list, tuple)):
        imgs, targets = batch[0], batch[1]
    else:
        imgs, targets = batch
    
    # Se è un batch, prendi il primo elemento
    if torch.is_tensor(imgs) and imgs.ndim == 4:
        img = imgs[0]
        target = {k: v[0] if torch.is_tensor(v) and v.ndim > 2 else v for k, v in targets.items()}
    else:
        img = imgs
        target = targets
    
    print(f"\n📊 Sample info:")
    print(f"   Image shape: {img.shape}")
    print(f"   Image dtype: {img.dtype}")
    print(f"   Image range: [{img.min().item():.3f}, {img.max().item():.3f}]")
    print(f"   Target keys: {list(target.keys())}")
    
    # Verifica ood_mask
    if "ood_mask" in target:
        ood_mask = target["ood_mask"]
        ood_sum = int(ood_mask.sum().item())
        print(f"\n✅ ood_mask present:")
        print(f"   Shape: {ood_mask.shape}")
        print(f"   Dtype: {ood_mask.dtype}")
        print(f"   Sum: {ood_sum}")
        print(f"   Unique values: {torch.unique(ood_mask).tolist()}")
        
        if ood_sum == 0:
            print("\n❌ WARNING: ood_mask sum is 0!")
        else:
            print(f"\n✅ ood_mask has {ood_sum} OOD pixels")
    else:
        print("\n❌ ERROR: ood_mask NOT in target!")
        return
    
    # Salva visualizzazione
    output_dir = Path("debug_visualizations")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "ood_overlay_test.png"
    
    # Normalizza img a [0, 1] se necessario
    if img.max() > 1.0:
        img_normalized = img / 255.0
    else:
        img_normalized = img
    
    visualize_ood_overlay(img_normalized, ood_mask, save_path)
    
    print(f"\n✅ Test completato! Visualizzazione salvata in {save_path}")

if __name__ == "__main__":
    main()
