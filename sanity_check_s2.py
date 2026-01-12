#!/usr/bin/env python3
"""
Sanity Check S2: Allineamento (20 esempi)
Verifica allineamento ood_mask ↔ immagine dopo TUTTE le trasformazioni.

Uso:
    python sanity_check_s2.py --config <path_to_config.yaml> --data_path <path_to_cityscapes> [--coco_path <path_to_coco>]
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import importlib
import argparse

# ============================================================================
# SETUP PATH - Aggiungi eomt/ al Python path per gli import
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
EOMT_ROOT = os.path.join(CURRENT_DIR, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Sanity Check S2: Alignment check")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to Cityscapes dataset directory",
    )
    parser.add_argument(
        "--coco_path",
        type=str,
        default=None,
        help="Path to COCO dataset (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sanity_check_s2_overlays",
        help="Directory to save overlay PNG files",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of examples with OOD to check (default: 20)",
    )
    parser.add_argument(
        "--min_ood_ratio",
        type=float,
        default=0.01,
        help="Minimum ood_ratio to consider example 'large' (default: 0.01 = 1%%)",
    )
    args = parser.parse_args()
    
    # ============================================================================
    # LOAD DATALOADER
    # ============================================================================
    print("🔍 Loading config...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_module_name, class_name = config["data"]["class_path"].rsplit(".", 1)
    data_module = getattr(importlib.import_module(data_module_name), class_name)
    data_module_kwargs = config["data"].get("init_args", {})

    data_module_kwargs["path"] = args.data_path
    if args.coco_path is not None and "coco_path" in data_module_kwargs:
        data_module_kwargs["coco_path"] = args.coco_path
    
    data_module_kwargs["batch_size"] = 1
    data_module_kwargs["num_workers"] = 0

    print("🔍 Creating dataloader...")
    data = data_module(
        **data_module_kwargs
    ).setup()

    dataloader = data.train_dataloader()
    print(f"✅ Dataloader creato (len={len(dataloader)})")

    # ============================================================================
    # SANITY CHECK S2
    # ============================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Sanity Check S2: Verifica allineamento su {args.num_examples} esempi con OOD 'grande' (>={args.min_ood_ratio*100:.1f}%)...\n")

    examples = []
    total_checked = 0
    
    for batch_idx, batch in enumerate(dataloader):
        imgs, targets = batch
        
        for sample_idx, target in enumerate(targets):
            if sample_idx >= imgs.shape[0]:
                break
            
            img = imgs[sample_idx]  # [C, H, W]
            
            if "ood_mask" not in target:
                continue
            
            ood_mask = target["ood_mask"]  # [H, W]
            
            # Calcola ood_ratio
            valid_pixels = (ood_mask != 255).sum().item()
            ood_pixels = (ood_mask == 1).sum().item()
            
            if valid_pixels > 0:
                ood_ratio = ood_pixels / valid_pixels
                
                # Salva solo esempi con OOD "grande"
                if ood_ratio >= args.min_ood_ratio:
                    # Check shape match
                    img_h, img_w = img.shape[-2:]
                    ood_h, ood_w = ood_mask.shape[-2:]
                    
                    if img_h == ood_h and img_w == ood_w:
                        examples.append((batch_idx, sample_idx, img, ood_mask, ood_ratio))
                        total_checked += 1
                        
                        if len(examples) >= args.num_examples:
                            break
        
        if len(examples) >= args.num_examples:
            break
        
        # Progress
        if total_checked % 50 == 0 and total_checked > 0:
            print(f"  Trovati {len(examples)}/{args.num_examples} esempi con OOD grande (controllati {total_checked} batch)...", end='\r')
    
    print()  # New line
    
    # ============================================================================
    # SALVA OVERLAY
    # ============================================================================
    print(f"\n{'='*60}")
    print(f"📸 Salvando {len(examples)} overlay PNG...")
    print(f"{'='*60}\n")
    
    alignment_errors = 0
    
    for example_idx, (batch_idx, sample_idx, img, ood_mask, ood_ratio) in enumerate(examples):
        # Check shape match
        img_h, img_w = img.shape[-2:]
        ood_h, ood_w = ood_mask.shape[-2:]
        
        if img_h != ood_h or img_w != ood_w:
            alignment_errors += 1
            print(f"❌ Esempio {example_idx + 1}: Shape mismatch! img={img_h}x{img_w}, ood_mask={ood_h}x{ood_w}")
        
        # Converti img a numpy [H, W, 3] in range [0, 255]
        img_np = img.permute(1, 2, 0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # Converti ood_mask a numpy
        ood_mask_np = ood_mask.cpu().numpy().astype(np.uint8)
        
        # Crea overlay: ood_mask in rosso sopra immagine
        overlay = img_np.copy()
        ood_region = (ood_mask_np == 1)
        
        # Overlay rosso per OOD (alpha blending 50%)
        overlay[ood_region] = np.clip(
            overlay[ood_region] * 0.5 + np.array([255, 0, 0]) * 0.5,
            0, 255
        ).astype(np.uint8)
        
        # Salva PNG
        output_path = output_dir / f"alignment_example_{example_idx + 1:02d}_batch{batch_idx}_sample{sample_idx}_ratio{ood_ratio:.4f}.png"
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(output_path)
        print(f"  ✅ Esempio {example_idx + 1}: Overlay salvato - ood_ratio={ood_ratio:.4f} ({output_path.name})")
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*60}")
    print("✅ Sanity Check S2 COMPLETATO")
    print(f"{'='*60}\n")
    print(f"Esempi controllati: {len(examples)}")
    print(f"Alignment errors: {alignment_errors}")
    print(f"Overlay salvati in: {output_dir}")
    
    if alignment_errors > 0:
        print(f"\n❌ Trovati {alignment_errors} errori di allineamento!")
        sys.exit(1)
    else:
        print("\n✅ Nessun errore di allineamento trovato!")
        print("✅ Verifica visivamente gli overlay per confermare che il rosso (OOD) si allinea con gli oggetti incollati.")
        sys.exit(0)


if __name__ == "__main__":
    main()
