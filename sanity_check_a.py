#!/usr/bin/env python3
"""
Sanity Check A: Ultra-economico (0 training, 0 backprop, solo dataloader)
Verifica: img.shape, ood_mask.shape, unique+counts, device, overlay PNG

Uso:
    python sanity_check_a.py --config <path_to_config.yaml> --data_path <path_to_cityscapes> [--coco_path <path_to_coco>]
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import yaml
import importlib
import argparse


def main():
    parser = argparse.ArgumentParser(description="Sanity Check A: Dataloader verification")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml)",
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
        help="Path to COCO dataset (optional, only if using OE)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sanity_check_overlays",
        help="Directory to save overlay PNG files (default: ./sanity_check_overlays)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=3,
        help="Number of batches to check (default: 3)",
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

    # Override path
    data_module_kwargs["path"] = args.data_path
    if args.coco_path is not None and "coco_path" in data_module_kwargs:
        data_module_kwargs["coco_path"] = args.coco_path

    # Crea dataloader (batch_size=1 per semplicità, num_workers=0 per debug)
    print("🔍 Creating dataloader...")
    data = data_module(
        path=args.data_path,
        batch_size=1,  # Piccolo per risparmiare memoria
        num_workers=0,  # 0 per debug (evita problemi multiprocessing)
        **data_module_kwargs
    ).setup()

    dataloader = data.train_dataloader()
    print(f"✅ Dataloader creato (len={len(dataloader)})")

    # ============================================================================
    # SANITY CHECK
    # ============================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Sanity Check A: Analizzando {args.num_batches} batch...\n")

    errors = []
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_count >= args.num_batches:
            break
        
        imgs, targets = batch
        
        print(f"{'='*60}")
        print(f"Batch {batch_idx}")
        print(f"{'='*60}")
        
        # Check immagini
        print(f"📸 Img shape: {imgs.shape}")
        print(f"   Img device: {imgs.device}")
        print(f"   Img dtype: {imgs.dtype}")
        
        # Check targets
        if not isinstance(targets, list):
            errors.append(f"Batch {batch_idx}: targets non è una lista!")
            print(f"❌ ERRORE: targets non è una lista (type={type(targets)})")
            sys.exit(1)
        
        print(f"📦 Num targets: {len(targets)}")
        
        # Per ogni sample nel batch
        for sample_idx, target in enumerate(targets):
            if sample_idx >= imgs.shape[0]:
                break
            
            img = imgs[sample_idx]  # [C, H, W]
            
            # Check ood_mask
            if "ood_mask" not in target:
                errors.append(f"Batch {batch_idx}, Sample {sample_idx}: ood_mask NON presente!")
                print(f"❌ ERRORE: ood_mask NON presente nel target!")
                sys.exit(1)
            
            ood_mask = target["ood_mask"]
            
            print(f"\n  Sample {sample_idx}:")
            print(f"    ood_mask shape: {ood_mask.shape}")
            print(f"    ood_mask device: {ood_mask.device}")
            print(f"    ood_mask dtype: {ood_mask.dtype}")
            
            # Check shape match
            img_h, img_w = img.shape[-2:]
            ood_h, ood_w = ood_mask.shape[-2:]
            
            if img_h != ood_h or img_w != ood_w:
                errors.append(f"Batch {batch_idx}, Sample {sample_idx}: Shape mismatch! img={img_h}x{img_w}, ood_mask={ood_h}x{ood_w}")
                print(f"❌ ERRORE: Shape mismatch!")
                print(f"   img: {img_h}x{img_w}")
                print(f"   ood_mask: {ood_h}x{ood_w}")
                sys.exit(1)
            
            # Check unique values
            unique_values, counts = torch.unique(ood_mask, return_counts=True)
            print(f"    ood_mask unique values: {unique_values.tolist()}")
            print(f"    ood_mask counts: {counts.tolist()}")
            
            # Check valori validi (0, 1, 255)
            invalid_values = unique_values[(unique_values != 0) & (unique_values != 1) & (unique_values != 255)]
            if len(invalid_values) > 0:
                errors.append(f"Batch {batch_idx}, Sample {sample_idx}: Valori invalidi in ood_mask: {invalid_values.tolist()}")
                print(f"⚠️ WARNING: Valori invalidi in ood_mask: {invalid_values.tolist()}")
            
            # Check device (dovrebbe essere CPU all'inizio, Lightning lo sposta dopo)
            if ood_mask.device != torch.device("cpu"):
                print(f"⚠️ WARNING: ood_mask device = {ood_mask.device} (atteso CPU)")
            if img.device != torch.device("cpu"):
                print(f"⚠️ WARNING: img device = {img.device} (atteso CPU)")
            
            # Device mismatch check
            if ood_mask.device != img.device:
                errors.append(f"Batch {batch_idx}, Sample {sample_idx}: Device mismatch! img={img.device}, ood_mask={ood_mask.device}")
                print(f"❌ ERRORE: Device mismatch!")
                print(f"   img device: {img.device}")
                print(f"   ood_mask device: {ood_mask.device}")
                sys.exit(1)
            
            # ====================================================================
            # SALVA OVERLAY PNG (ood in rosso)
            # ====================================================================
            # Converti img a numpy [H, W, 3] in range [0, 255]
            img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            
            # Converti ood_mask a numpy
            ood_mask_np = ood_mask.cpu().numpy().astype(np.uint8)  # [H, W]
            
            # Crea overlay: ood_mask in rosso sopra immagine
            overlay = img_np.copy()
            ood_region = (ood_mask_np == 1)  # OOD pixels
            
            # Overlay rosso per OOD (alpha blending 50%)
            overlay[ood_region] = np.clip(
                overlay[ood_region] * 0.5 + np.array([255, 0, 0]) * 0.5,
                0, 255
            ).astype(np.uint8)
            
            # Salva PNG
            output_path = output_dir / f"overlay_batch{batch_idx}_sample{sample_idx}.png"
            overlay_pil = Image.fromarray(overlay)
            overlay_pil.save(output_path)
            print(f"    ✅ Overlay salvato: {output_path}")
        
        batch_count += 1
        print()

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print(f"\n{'='*60}")
    print("✅ Sanity Check A COMPLETATO")
    print(f"{'='*60}")
    print(f"Batch analizzati: {batch_count}")
    print(f"Overlay salvati in: {output_dir}")

    if errors:
        print(f"\n❌ ERRORI TROVATI ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n✅ Nessun errore trovato! Tutti i check sono passati.")
        print("✅ Puoi procedere con il training.")


if __name__ == "__main__":
    main()
