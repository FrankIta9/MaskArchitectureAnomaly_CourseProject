#!/usr/bin/env python3
"""
Sanity Check S1+S2+S3: DATA-ONLY mega check (500 batch)
Verifica che il paste OE sia "utile" (posizione, dimensione, occlusione GT) prima di fare training.

Cosa calcola:
- S1: OOD placement - % OOD nella fascia bassa dell'immagine (proxy "on-road")
- S2: OOD size - distribuzione ood_ratio, ood_pixels, percentili P50/P90/P99
- S3: GT occlusion - quanto GT "valido" viene coperto da OOD

Uso:
    python eval/sanity/sanity_ood_data_500.py \
        --config configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
        --data_path /path/to/Cityscapes \
        --coco_path /path/to/coco \
        --num_batches 500 \
        --out_dir ./sanity_out/data_500
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
from typing import List, Tuple, Dict, Any
from collections import defaultdict

# ============================================================================
# SETUP PATH - Aggiungi eomt/ al Python path per gli import
# ============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Risali di 2 livelli: eval/sanity/ -> eval/ -> project_root/
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)


def compute_ood_placement_stats(ood_mask: torch.Tensor, h: int, w: int) -> Dict[str, float]:
    """
    S1: Calcola statistiche sul placement OOD (fascia bassa = on-road proxy).
    
    Args:
        ood_mask: [H, W] con valori 0=ID, 1=OOD, 255=ignore
        h, w: altezza e larghezza dell'immagine
        
    Returns:
        Dict con statistiche sul placement
    """
    ood_pixels = (ood_mask == 1)
    if not ood_pixels.any():
        return {
            "ood_in_bottom_ratio": 0.0,
            "ood_in_top_ratio": 0.0,
            "ood_in_middle_ratio": 0.0,
        }
    
    total_ood = ood_pixels.sum().item()
    
    # Fascia bassa: ultimo 40% dell'immagine (proxy "on-road")
    bottom_threshold = int(h * 0.6)  # Da 60% in poi = bottom 40%
    ood_in_bottom = ood_pixels[bottom_threshold:, :].sum().item()
    ood_in_bottom_ratio = ood_in_bottom / total_ood if total_ood > 0 else 0.0
    
    # Fascia alta: primo 30% dell'immagine (cielo)
    top_threshold = int(h * 0.3)
    ood_in_top = ood_pixels[:top_threshold, :].sum().item()
    ood_in_top_ratio = ood_in_top / total_ood if total_ood > 0 else 0.0
    
    # Fascia media: 30%-60%
    ood_in_middle = ood_pixels[top_threshold:bottom_threshold, :].sum().item()
    ood_in_middle_ratio = ood_in_middle / total_ood if total_ood > 0 else 0.0
    
    return {
        "ood_in_bottom_ratio": ood_in_bottom_ratio,
        "ood_in_top_ratio": ood_in_top_ratio,
        "ood_in_middle_ratio": ood_in_middle_ratio,
    }


def compute_ood_size_stats(ood_mask: torch.Tensor, h: int, w: int) -> Dict[str, float]:
    """
    S2: Calcola statistiche sulla dimensione OOD.
    
    Args:
        ood_mask: [H, W] con valori 0=ID, 1=OOD, 255=ignore
        h, w: altezza e larghezza dell'immagine
        
    Returns:
        Dict con statistiche sulla dimensione
    """
    valid_pixels = (ood_mask != 255).sum().item()
    ood_pixels = (ood_mask == 1).sum().item()
    
    if valid_pixels == 0:
        return {
            "ood_ratio": 0.0,
            "ood_pixels": 0,
        }
    
    ood_ratio = ood_pixels / valid_pixels
    
    return {
        "ood_ratio": ood_ratio,
        "ood_pixels": ood_pixels,
    }


def compute_gt_occlusion_stats(
    ood_mask: torch.Tensor, 
    target_masks: torch.Tensor,
    h: int, 
    w: int
) -> Dict[str, float]:
    """
    S3: Calcola quanto GT viene coperto da OOD.
    
    Args:
        ood_mask: [H, W] con valori 0=ID, 1=OOD, 255=ignore
        target_masks: [N, H, W] maschere GT
        h, w: altezza e larghezza dell'immagine
        
    Returns:
        Dict con statistiche sull'occlusione GT
    """
    ood_pixels = (ood_mask == 1)
    
    if not ood_pixels.any() or target_masks.numel() == 0:
        return {
            "gt_occlusion_ratio": 0.0,
            "gt_pixels_occluded": 0,
            "gt_pixels_total": 0,
        }
    
    # Union di tutte le maschere GT
    gt_union = target_masks.any(dim=0)  # [H, W]
    gt_pixels_total = gt_union.sum().item()
    
    if gt_pixels_total == 0:
        return {
            "gt_occlusion_ratio": 0.0,
            "gt_pixels_occluded": 0,
            "gt_pixels_total": 0,
        }
    
    # Pixel GT che sono coperti da OOD
    gt_occluded = (gt_union & ood_pixels).sum().item()
    gt_occlusion_ratio = gt_occluded / gt_pixels_total if gt_pixels_total > 0 else 0.0
    
    return {
        "gt_occlusion_ratio": gt_occlusion_ratio,
        "gt_pixels_occluded": gt_occluded,
        "gt_pixels_total": gt_pixels_total,
    }


def create_overlay(img: torch.Tensor, ood_mask: torch.Tensor, gt_masks: torch.Tensor = None) -> np.ndarray:
    """
    Crea overlay visivo: immagine + OOD (rosso) + GT (verde, opzionale).
    
    Args:
        img: [C, H, W] in range [0, 255]
        ood_mask: [H, W] con valori 0=ID, 1=OOD, 255=ignore
        gt_masks: [N, H, W] maschere GT (opzionale)
        
    Returns:
        Overlay come numpy array [H, W, 3] uint8
    """
    img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # [H, W, 3]
    ood_mask_np = ood_mask.cpu().numpy().astype(np.uint8)  # [H, W]
    
    overlay = img_np.copy()
    
    # OOD in rosso (alpha 50%)
    ood_region = (ood_mask_np == 1)
    overlay[ood_region] = np.clip(
        overlay[ood_region] * 0.5 + np.array([255, 0, 0]) * 0.5,
        0, 255
    ).astype(np.uint8)
    
    # GT in verde (alpha 30%, se disponibile)
    if gt_masks is not None and gt_masks.numel() > 0:
        gt_union = gt_masks.any(dim=0).cpu().numpy()  # [H, W]
        gt_region = gt_union & (~ood_region)  # GT non coperto da OOD
        overlay[gt_region] = np.clip(
            overlay[gt_region] * 0.7 + np.array([0, 255, 0]) * 0.3,
            0, 255
        ).astype(np.uint8)
    
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Sanity Check S1+S2+S3: DATA-ONLY mega check")
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
        help="Path to COCO dataset (optional, only if using OE)",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=500,
        help="Number of batches to analyze (default: 500)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./sanity_out/data_500",
        help="Output directory for results and overlays (default: ./sanity_out/data_500)",
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
    # SANITY CHECK S1+S2+S3
    # ============================================================================
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Sanity Check S1+S2+S3: Analizzando {args.num_batches} batch...\n")

    # Statistiche accumulate
    batches_with_ood = 0
    total_batches = 0
    
    # S1: OOD placement
    ood_in_bottom_ratios = []
    ood_in_top_ratios = []
    ood_in_middle_ratios = []
    
    # S2: OOD size
    ood_ratios = []
    ood_pixels_list = []
    
    # S3: GT occlusion
    gt_occlusion_ratios = []
    
    # Esempi per overlay (top 10 per categoria)
    examples_ood_high = []  # OOD in alto (top_ratio alto)
    examples_ood_huge = []  # OOD enorme (ood_ratio alto)
    examples_gt_occlusion = []  # Occlusione GT alta
    
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
            h, w = ood_mask.shape
            
            # Check se c'è OOD (valore 1)
            has_ood = (ood_mask == 1).any().item()
            
            if has_ood:
                batches_with_ood += 1
                
                # S1: OOD placement
                placement_stats = compute_ood_placement_stats(ood_mask, h, w)
                ood_in_bottom_ratios.append(placement_stats["ood_in_bottom_ratio"])
                ood_in_top_ratios.append(placement_stats["ood_in_top_ratio"])
                ood_in_middle_ratios.append(placement_stats["ood_in_middle_ratio"])
                
                # S2: OOD size
                size_stats = compute_ood_size_stats(ood_mask, h, w)
                ood_ratios.append(size_stats["ood_ratio"])
                ood_pixels_list.append(size_stats["ood_pixels"])
                
                # S3: GT occlusion
                if "masks" in target and target["masks"].numel() > 0:
                    occlusion_stats = compute_gt_occlusion_stats(
                        ood_mask, target["masks"], h, w
                    )
                    gt_occlusion_ratios.append(occlusion_stats["gt_occlusion_ratio"])
                    
                    # Salva esempio per overlay (top 10 occlusione GT)
                    if len(examples_gt_occlusion) < 10:
                        examples_gt_occlusion.append((
                            batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                            occlusion_stats["gt_occlusion_ratio"]
                        ))
                    else:
                        # Mantieni solo i top 10 con occlusione più alta
                        min_occlusion = min(ex[5] for ex in examples_gt_occlusion)
                        if occlusion_stats["gt_occlusion_ratio"] > min_occlusion:
                            examples_gt_occlusion = sorted(
                                examples_gt_occlusion + [(
                                    batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                                    occlusion_stats["gt_occlusion_ratio"]
                                )],
                                key=lambda x: x[5],
                                reverse=True
                            )[:10]
                
                # Salva esempio per overlay (top 10 OOD in alto)
                if len(examples_ood_high) < 10:
                    examples_ood_high.append((
                        batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                        placement_stats["ood_in_top_ratio"]
                    ))
                else:
                    min_top_ratio = min(ex[5] for ex in examples_ood_high)
                    if placement_stats["ood_in_top_ratio"] > min_top_ratio:
                        examples_ood_high = sorted(
                            examples_ood_high + [(
                                batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                                placement_stats["ood_in_top_ratio"]
                            )],
                            key=lambda x: x[5],
                            reverse=True
                        )[:10]
                
                # Salva esempio per overlay (top 10 OOD enorme)
                if len(examples_ood_huge) < 10:
                    examples_ood_huge.append((
                        batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                        size_stats["ood_ratio"]
                    ))
                else:
                    min_ood_ratio = min(ex[5] for ex in examples_ood_huge)
                    if size_stats["ood_ratio"] > min_ood_ratio:
                        examples_ood_huge = sorted(
                            examples_ood_huge + [(
                                batch_idx, sample_idx, img, ood_mask, target.get("masks"),
                                size_stats["ood_ratio"]
                            )],
                            key=lambda x: x[5],
                            reverse=True
                        )[:10]
            
            total_batches += 1
            
            # Progress ogni 50 batch
            if total_batches % 50 == 0:
                print(f"  Processati {total_batches} batch...", end='\r')
    
    print()  # New line after progress
    
    # ============================================================================
    # STATISTICHE FINALI
    # ============================================================================
    print(f"\n{'='*60}")
    print("📊 Sanity Check S1+S2+S3: Risultati")
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
    
    # S1: OOD Placement Statistics
    if len(ood_in_bottom_ratios) > 0:
        print(f"{'='*60}")
        print("S1: OOD Placement (fascia bassa = on-road proxy)")
        print(f"{'='*60}")
        bottom_mean = np.mean(ood_in_bottom_ratios)
        bottom_p50 = np.percentile(ood_in_bottom_ratios, 50)
        bottom_p90 = np.percentile(ood_in_bottom_ratios, 90)
        bottom_p99 = np.percentile(ood_in_bottom_ratios, 99)
        
        top_mean = np.mean(ood_in_top_ratios)
        top_p50 = np.percentile(ood_in_top_ratios, 50)
        
        print(f"OOD in bottom 40% (on-road proxy):")
        print(f"  Mean: {bottom_mean:.4f}")
        print(f"  P50 (median): {bottom_p50:.4f}")
        print(f"  P90: {bottom_p90:.4f}")
        print(f"  P99: {bottom_p99:.4f}")
        print()
        print(f"OOD in top 30% (cielo):")
        print(f"  Mean: {top_mean:.4f}")
        print(f"  P50 (median): {top_p50:.4f}")
        print()
        
        # Warning se OOD finisce spesso in alto
        if bottom_p50 < 0.4:
            print(f"⚠️  WARNING: P50 OOD in bottom ({bottom_p50:.4f}) < 0.4")
            print(f"⚠️  OOD finisce spesso in cielo invece che on-road!")
        else:
            print(f"✅ P50 OOD in bottom ({bottom_p50:.4f}) >= 0.4 - OK")
        print()
    
    # S2: OOD Size Statistics
    if len(ood_ratios) > 0:
        print(f"{'='*60}")
        print("S2: OOD Size Distribution")
        print(f"{'='*60}")
        ratio_mean = np.mean(ood_ratios)
        ratio_p50 = np.percentile(ood_ratios, 50)
        ratio_p90 = np.percentile(ood_ratios, 90)
        ratio_p99 = np.percentile(ood_ratios, 99)
        
        pixels_mean = np.mean(ood_pixels_list)
        pixels_p50 = np.percentile(ood_pixels_list, 50)
        pixels_p90 = np.percentile(ood_pixels_list, 90)
        pixels_p99 = np.percentile(ood_pixels_list, 99)
        
        print(f"OOD Ratio (pixel OOD / pixel validi):")
        print(f"  Mean: {ratio_mean:.6f}")
        print(f"  P50 (median): {ratio_p50:.6f}")
        print(f"  P90: {ratio_p90:.6f}")
        print(f"  P99: {ratio_p99:.6f}")
        print()
        print(f"OOD Pixels (assoluto):")
        print(f"  Mean: {pixels_mean:.1f}")
        print(f"  P50 (median): {pixels_p50:.1f}")
        print(f"  P90: {pixels_p90:.1f}")
        print(f"  P99: {pixels_p99:.1f}")
        print()
        
        # Check P90 requirement
        if ratio_p90 >= 0.005:
            print(f"✅ P90 ood_ratio ({ratio_p90:.6f}) >= 0.005 - OK")
        else:
            print(f"❌ P90 ood_ratio ({ratio_p90:.6f}) < 0.005")
            print(f"⚠️  Gli oggetti OOD sono troppo piccoli!")
        print()
    
    # S3: GT Occlusion Statistics
    if len(gt_occlusion_ratios) > 0:
        print(f"{'='*60}")
        print("S3: GT Occlusion (quanto GT viene coperto da OOD)")
        print(f"{'='*60}")
        occlusion_mean = np.mean(gt_occlusion_ratios)
        occlusion_p50 = np.percentile(gt_occlusion_ratios, 50)
        occlusion_p90 = np.percentile(gt_occlusion_ratios, 90)
        occlusion_p99 = np.percentile(gt_occlusion_ratios, 99)
        
        print(f"GT Occlusion Ratio (pixel GT coperti / pixel GT totali):")
        print(f"  Mean: {occlusion_mean:.6f}")
        print(f"  P50 (median): {occlusion_p50:.6f}")
        print(f"  P90: {occlusion_p90:.6f}")
        print(f"  P99: {occlusion_p99:.6f}")
        print()
        
        # Warning se occlusione GT è alta
        if occlusion_mean > 0.10:
            print(f"⚠️  WARNING: Mean GT occlusion ({occlusion_mean:.4f}) > 0.10 (10%)")
            print(f"⚠️  Stai coprendo troppo GT con OOD, potrebbe rovinare la supervision!")
        elif occlusion_mean > 0.05:
            print(f"⚠️  WARNING: Mean GT occlusion ({occlusion_mean:.4f}) > 0.05 (5%)")
            print(f"⚠️  Considera di ridurre overlap OOD vs GT.")
        else:
            print(f"✅ Mean GT occlusion ({occlusion_mean:.4f}) <= 0.05 - OK")
        print()
    else:
        print(f"{'='*60}")
        print("S3: GT Occlusion")
        print(f"{'='*60}")
        print("⚠️  Nessun dato GT disponibile per calcolare occlusione")
        print()
    
    # ============================================================================
    # SALVA OVERLAY
    # ============================================================================
    print(f"{'='*60}")
    print("📸 Salvataggio overlay...")
    print(f"{'='*60}\n")
    
    # Top 10 OOD in alto
    examples_ood_high = sorted(examples_ood_high, key=lambda x: x[5], reverse=True)[:10]
    for idx, (batch_idx, sample_idx, img, ood_mask, gt_masks, top_ratio) in enumerate(examples_ood_high):
        overlay = create_overlay(img, ood_mask, gt_masks)
        overlay_pil = Image.fromarray(overlay)
        output_path = output_dir / f"top10_ood_high_{idx+1}_batch{batch_idx}_sample{sample_idx}_top{top_ratio:.4f}.png"
        overlay_pil.save(output_path)
        print(f"  ✅ Overlay OOD in alto #{idx+1}: {output_path.name}")
    
    # Top 10 OOD enorme
    examples_ood_huge = sorted(examples_ood_huge, key=lambda x: x[5], reverse=True)[:10]
    for idx, (batch_idx, sample_idx, img, ood_mask, gt_masks, ood_ratio) in enumerate(examples_ood_huge):
        overlay = create_overlay(img, ood_mask, gt_masks)
        overlay_pil = Image.fromarray(overlay)
        output_path = output_dir / f"top10_ood_huge_{idx+1}_batch{batch_idx}_sample{sample_idx}_ratio{ood_ratio:.6f}.png"
        overlay_pil.save(output_path)
        print(f"  ✅ Overlay OOD enorme #{idx+1}: {output_path.name}")
    
    # Top 10 Occlusione GT
    examples_gt_occlusion = sorted(examples_gt_occlusion, key=lambda x: x[5], reverse=True)[:10]
    for idx, (batch_idx, sample_idx, img, ood_mask, gt_masks, occlusion_ratio) in enumerate(examples_gt_occlusion):
        overlay = create_overlay(img, ood_mask, gt_masks)
        overlay_pil = Image.fromarray(overlay)
        output_path = output_dir / f"top10_gt_occlusion_{idx+1}_batch{batch_idx}_sample{sample_idx}_occl{occlusion_ratio:.4f}.png"
        overlay_pil.save(output_path)
        print(f"  ✅ Overlay occlusione GT #{idx+1}: {output_path.name}")
    
    print(f"\n✅ Overlay salvati in: {output_dir}")
    
    # ============================================================================
    # VALUTAZIONE FINALE
    # ============================================================================
    print(f"\n{'='*60}")
    print("🎯 Valutazione Finale (PASS/FAIL)")
    print(f"{'='*60}\n")
    
    pass_criteria = {
        "pct_batches_with_ood": pct_batches_with_ood >= 50.0,
        "p90_ood_ratio": len(ood_ratios) > 0 and np.percentile(ood_ratios, 90) >= 0.005,
        "p50_ood_bottom": len(ood_in_bottom_ratios) > 0 and np.percentile(ood_in_bottom_ratios, 50) >= 0.4,
        "mean_gt_occlusion": len(gt_occlusion_ratios) == 0 or np.mean(gt_occlusion_ratios) <= 0.10,
    }
    
    all_passed = all(pass_criteria.values())
    
    for criterion, passed in pass_criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {criterion}")
    
    print()
    if all_passed:
        print("✅ Sanity Check S1+S2+S3 PASSATO: Il paste OE è configurato correttamente!")
        sys.exit(0)
    else:
        print("❌ Sanity Check S1+S2+S3 FALLITO: Controlla i parametri di paste OE!")
        print("⚠️  NON procedere con training lungo finché S1+S2+S3 non passa!")
        sys.exit(1)


if __name__ == "__main__":
    main()
