#!/usr/bin/env python3
"""
Sanity Check S3: Semantics Conflict Check (IMPORTANTISSIMO)
Verifica che nei pixel OOD la loss semantica non stia forzando classi Cityscapes a caso.

Uso:
    python sanity_check_s3.py --config <path_to_config.yaml> --data_path <path_to_cityscapes> [--coco_path <path_to_coco>]
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
    parser = argparse.ArgumentParser(description="Sanity Check S3: Semantics Conflict Check")
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
        "--num_batches",
        type=int,
        default=50,
        help="Number of batches to check (default: 50)",
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

    num_classes = data_module_kwargs.get("num_classes", 19)  # Default Cityscapes

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
    print(f"   Num classes: {num_classes} (ignore=255, no-object={num_classes})")

    # ============================================================================
    # SANITY CHECK S3
    # ============================================================================
    print(f"\n🔍 Sanity Check S3: Verifica semantics conflict su {args.num_batches} batch...\n")

    # Note: Per semantic segmentation, il target è gestito come per-pixel targets
    # Non abbiamo direttamente accesso al target semantic per-pixel nel dataloader
    # perché viene creato in training_step tramite to_per_pixel_targets_semantic
    
    # Per ora, verifichiamo che ood_mask sia presente e corretto
    # Il vero check del conflitto semantics richiede di vedere come viene gestito
    # il target nella loss (se pixel OOD sono ignore o no-object)
    
    print("⚠️  NOTA: Questo check verifica che ood_mask sia presente e valido.")
    print("⚠️  Il vero conflitto semantics viene gestito nella loss.")
    print("⚠️  Verifica in mask_classification_loss.py che pixel OOD siano:")
    print("     1. Ignorati (ignore_idx=255) nella loss semantica, OPPURE")
    print("     2. Target = num_classes (no-object) nella loss semantica")
    print("     3. NON devono avere target in [0, num_classes-1] (classi Cityscapes)")
    print()
    
    batches_checked = 0
    batches_with_ood = 0
    conflicts_found = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batches_checked >= args.num_batches:
            break
        
        imgs, targets = batch
        
        for sample_idx, target in enumerate(targets):
            if sample_idx >= imgs.shape[0]:
                break
            
            if "ood_mask" not in target:
                continue
            
            ood_mask = target["ood_mask"]
            
            # Verifica ood_mask valido
            unique_values = torch.unique(ood_mask).tolist()
            valid_values = [0, 1, 255]
            invalid_values = [v for v in unique_values if v not in valid_values]
            
            if invalid_values:
                print(f"❌ Batch {batch_idx}, Sample {sample_idx}: Valori invalidi in ood_mask: {invalid_values}")
                conflicts_found += 1
            
            # Check se ha OOD
            if 1 in unique_values:
                batches_with_ood += 1
                
                # Verifica shape match
                img = imgs[sample_idx]
                img_h, img_w = img.shape[-2:]
                ood_h, ood_w = ood_mask.shape[-2:]
                
                if img_h != ood_h or img_w != ood_w:
                    print(f"❌ Batch {batch_idx}, Sample {sample_idx}: Shape mismatch! img={img_h}x{img_w}, ood_mask={ood_h}x{ood_w}")
                    conflicts_found += 1
        
        batches_checked += 1
        
        if batches_checked % 10 == 0:
            print(f"  Controllati {batches_checked} batch (trovati {batches_with_ood} con OOD)...", end='\r')
    
    print()  # New line
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*60}")
    print("✅ Sanity Check S3: Risultati")
    print(f"{'='*60}\n")
    
    print(f"Batch controllati: {batches_checked}")
    print(f"Batch con OOD: {batches_with_ood}")
    print(f"Conflitti trovati: {conflicts_found}")
    print()
    
    if conflicts_found > 0:
        print("❌ Trovati conflitti nella struttura ood_mask!")
        print("⚠️  Risolvi questi problemi prima di procedere.")
        sys.exit(1)
    else:
        print("✅ Nessun conflitto nella struttura ood_mask trovato!")
        print()
        print("⚠️  IMPORTANTE: Verifica manualmente in mask_classification_loss.py:")
        print("   1. I pixel OOD devono essere ignorati (ignore_idx=255) nella loss semantica, OPPURE")
        print("   2. I pixel OOD devono avere target = num_classes (no-object)")
        print("   3. NON devono avere target in [0, num_classes-1] (classi Cityscapes normali)")
        print()
        print("   Se i pixel OOD hanno target semantic di classi Cityscapes, si crea")
        print("   un conflitto distruttivo: loss semantica forza 'road', energy forza 'incerto/alto'")
        print("   → Risultato: tutto peggiora")
        sys.exit(0)


if __name__ == "__main__":
    main()
