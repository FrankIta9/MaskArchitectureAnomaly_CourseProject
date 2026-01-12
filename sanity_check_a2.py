#!/usr/bin/env python3
"""
Sanity Check A2: Conta batch con OOD (obbligatorio, costo zero)
Scorre 50-200 batch e conta quanti hanno OOD per verificare che il paste funzioni.

Uso:
    python sanity_check_a2.py --config <path_to_config.yaml> --data_path <path_to_cityscapes> [--coco_path <path_to_coco>] [--num_batches 200]
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
    parser = argparse.ArgumentParser(description="Sanity Check A2: Count batches with OOD")
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
        default="./sanity_check_a2_overlays",
        help="Directory to save overlay PNG files (default: ./sanity_check_a2_overlays)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=200,
        help="Number of batches to check (default: 200)",
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

    # Override path e altri parametri
    data_module_kwargs["path"] = args.data_path
    if args.coco_path is not None and "coco_path" in data_module_kwargs:
        data_module_kwargs["coco_path"] = args.coco_path
    
    # Override batch_size e num_workers per sanity check
    data_module_kwargs["batch_size"] = 1  # Piccolo per risparmiare memoria
    data_module_kwargs["num_workers"] = 0  # 0 per debug (evita problemi multiprocessing)

    # Crea dataloader
    print("🔍 Creating dataloader...")
    data = data_module(
        **data_module_kwargs
    ).setup()

    dataloader = data.train_dataloader()
    print(f"✅ Dataloader creato (len={len(dataloader)})")

    # ============================================================================
    # SANITY CHECK A2
    # ============================================================================
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Sanity Check A2: Analizzando {args.num_batches} batch per verificare paste COCO...\n")

    # Statistiche
    total_batches = 0
    batches_with_ood = 0
    ood_ratios = []  # Lista di ood_ratio per ogni batch con OOD
    
    # Esempi con OOD (per overlay)
    ood_examples = []  # Lista di (batch_idx, sample_idx, img, ood_mask)
    
    for batch_idx, batch in enumerate(dataloader):
        if total_batches >= args.num_batches:
            break
        
        imgs, targets = batch
        
        # Per ogni sample nel batch
        for sample_idx, target in enumerate(targets):
            if sample_idx >= imgs.shape[0]:
                break
            
            img = imgs[sample_idx]  # [C, H, W]
            
            # Check ood_mask
            if "ood_mask" not in target:
                continue  # Skip se non c'è ood_mask
            
            ood_mask = target["ood_mask"]  # [H, W]
            
            # Calcola statistiche
            unique_values, counts = torch.unique(ood_mask, return_counts=True)
            unique_list = unique_values.tolist()
            counts_list = counts.tolist()
            
            # Check se c'è OOD (valore 1)
            has_ood = 1 in unique_list
            
            if has_ood:
                batches_with_ood += 1
                
                # Calcola ood_ratio (pixel OOD / pixel validi)
                valid_pixels = (ood_mask != 255).sum().item()
                ood_pixels = (ood_mask == 1).sum().item()
                
                if valid_pixels > 0:
                    ood_ratio = ood_pixels / valid_pixels
                    ood_ratios.append(ood_ratio)
                    
                    # Salva esempio con OOD (max 3)
                    if len(ood_examples) < 3:
                        ood_examples.append((batch_idx, sample_idx, img, ood_mask))
            
            total_batches += 1
            
            # Progress ogni 20 batch
            if total_batches % 20 == 0:
                print(f"  Processati {total_batches} batch...", end='\r')
    
    print()  # New line after progress
    
    # ============================================================================
    # STATISTICHE FINALI
    # ============================================================================
    print(f"\n{'='*60}")
    print("📊 Sanity Check A2: Risultati")
    print(f"{'='*60}\n")
    
    # Calcola percentuale batch con OOD
    if total_batches > 0:
        pct_batches_with_ood = (batches_with_ood / total_batches) * 100
        print(f"Batch totali analizzati: {total_batches}")
        print(f"Batch con OOD: {batches_with_ood}")
        print(f"Percentuale batch con OOD: {pct_batches_with_ood:.2f}%")
        print()
    else:
        print("❌ ERRORE: Nessun batch analizzato!")
        sys.exit(1)
    
    # Calcola ood_ratio statistics
    if len(ood_ratios) > 0:
        ood_ratio_min = min(ood_ratios)
        ood_ratio_mean = np.mean(ood_ratios)
        ood_ratio_max = max(ood_ratios)
        ood_ratio_std = np.std(ood_ratios)
        
        print(f"OOD Ratio (solo batch con OOD):")
        print(f"  Min: {ood_ratio_min:.6f}")
        print(f"  Mean: {ood_ratio_mean:.6f}")
        print(f"  Max: {ood_ratio_max:.6f}")
        print(f"  Std: {ood_ratio_std:.6f}")
        print()
    else:
        print("❌ ERRORE: Nessun batch con OOD trovato!")
        print("⚠️  Questo significa che il paste COCO non sta funzionando!")
        print("⚠️  Controlla paste_probability, filtri di posizionamento, ecc.")
        sys.exit(1)
    
    # ============================================================================
    # ESEMPI CON OOD
    # ============================================================================
    print(f"{'='*60}")
    print("📸 Esempi con OOD (primi 3):")
    print(f"{'='*60}\n")
    
    for example_idx, (batch_idx, sample_idx, img, ood_mask) in enumerate(ood_examples):
        # Calcola statistiche
        unique_values, counts = torch.unique(ood_mask, return_counts=True)
        unique_list = unique_values.tolist()
        counts_list = counts.tolist()
        
        valid_pixels = (ood_mask != 255).sum().item()
        ood_pixels = (ood_mask == 1).sum().item()
        ood_ratio = ood_pixels / valid_pixels if valid_pixels > 0 else 0.0
        
        print(f"Esempio {example_idx + 1}: Batch {batch_idx}, Sample {sample_idx}")
        print(f"  ood_mask unique values: {unique_list}")
        print(f"  ood_mask counts: {counts_list}")
        print(f"  ood_ratio: {ood_ratio:.6f} ({ood_pixels}/{valid_pixels} pixel)")
        print()
        
        # Salva overlay PNG
        img_np = img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
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
        output_path = output_dir / f"example_{example_idx + 1}_batch{batch_idx}_sample{sample_idx}.png"
        overlay_pil = Image.fromarray(overlay)
        overlay_pil.save(output_path)
        print(f"  ✅ Overlay salvato: {output_path}\n")
    
    # ============================================================================
    # VALUTAZIONE
    # ============================================================================
    print(f"{'='*60}")
    print("🎯 Valutazione:")
    print(f"{'='*60}\n")
    
    # Target ragionevole: almeno 20-50% dei batch con OOD
    target_min_pct = 20.0
    
    if pct_batches_with_ood >= target_min_pct:
        print(f"✅ Percentuale batch con OOD ({pct_batches_with_ood:.2f}%) >= target ({target_min_pct}%)")
    else:
        print(f"❌ Percentuale batch con OOD ({pct_batches_with_ood:.2f}%) < target ({target_min_pct}%)")
        print(f"⚠️  Il paste COCO non sta funzionando abbastanza!")
        print(f"⚠️  Controlla paste_probability, filtri di posizionamento, ecc.")
    
    if len(ood_ratios) > 0:
        if ood_ratio_mean > 0.0001:
            print(f"✅ OOD ratio medio ({ood_ratio_mean:.6f}) > 0.0001")
        else:
            print(f"❌ OOD ratio medio ({ood_ratio_mean:.6f}) <= 0.0001")
            print(f"⚠️  Gli oggetti COCO sono troppo piccoli o rari!")
    
    print(f"\n✅ Overlay salvati in: {output_dir}")
    
    # Exit code
    if pct_batches_with_ood >= target_min_pct and len(ood_ratios) > 0 and ood_ratio_mean > 0.0001:
        print("\n✅ Sanity Check A2 PASSATO: Il paste COCO sta funzionando!")
        sys.exit(0)
    else:
        print("\n❌ Sanity Check A2 FALLITO: Il paste COCO non sta funzionando abbastanza!")
        print("⚠️  Fai debug dei parametri di paste (paste_probability, filtri, ecc.)")
        sys.exit(1)


if __name__ == "__main__":
    main()
