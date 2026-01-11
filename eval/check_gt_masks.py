import os
import glob
import random
import numpy as np
from PIL import Image
from collections import Counter, defaultdict
from argparse import ArgumentParser

def find_mask_files(dataset_root):
    # Cerca ricorsivamente png/jpg/webp nelle cartelle labels/masks
    exts = ("png", "jpg", "jpeg", "webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(dataset_root, "**", f"*.{ext}"), recursive=True))
    return files

def load_mask(path):
    # Non convertire in RGB: ci servono i valori raw
    m = Image.open(path)
    arr = np.array(m)
    # Se arriva HxWx3 (mask “colorata”), prova a prenderne un canale e avvisa
    if arr.ndim == 3:
        # spesso sono palette/colored; qui prendiamo il primo canale per diagnosticare
        arr = arr[..., 0]
    return arr

def summarize_dataset(name, mask_files, sample_n=30, seed=42, per_image_debug=5):
    random.seed(seed)
    if len(mask_files) == 0:
        print(f"\n=== {name} ===")
        print("❌ Nessuna mask trovata.")
        return

    sample = mask_files if len(mask_files) <= sample_n else random.sample(mask_files, sample_n)

    global_counts = Counter()
    value_in_images = Counter()
    per_image_uniques = []

    for p in sample:
        arr = load_mask(p)
        uniq, cnt = np.unique(arr, return_counts=True)
        per_image_uniques.append((p, list(zip(uniq.tolist(), cnt.tolist()))))

        # accumula global
        for u, c in zip(uniq, cnt):
            global_counts[int(u)] += int(c)

        # presenza valore nelle immagini
        for u in uniq:
            value_in_images[int(u)] += 1

    total_px = sum(global_counts.values())
    sorted_vals = sorted(global_counts.keys())

    print(f"\n=== {name} ===")
    print(f"Masks trovate: {len(mask_files)} | Campionate: {len(sample)} | Pixel tot campione: {total_px}")

    print("\nValori globali (valore -> pixel_count, %):")
    for v in sorted_vals:
        c = global_counts[v]
        pct = (c / total_px) * 100 if total_px > 0 else 0
        print(f"  {v:>4} -> {c:>12} px  ({pct:6.2f}%)")

    print("\nPresenza valori nelle immagini (valore -> num_immagini_su_campione):")
    for v in sorted_vals:
        print(f"  {v:>4} -> {value_in_images[v]:>4}/{len(sample)}")

    # Debug per-image
    if per_image_debug > 0:
        print("\nEsempi (prime mask) [path -> unique+count]:")
        for p, uc in per_image_uniques[:per_image_debug]:
            print(f"  - {p}")
            print(f"    {uc}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets_root", required=True, help="Cartella che contiene le cartelle dei dataset")
    parser.add_argument("--sample_n", type=int, default=30, help="Numero mask da campionare per dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Qui mettiamo i nomi cartella reali che usi tu
    # (modifica se i folder si chiamano diversamente)
    dataset_folders = [
        "RoadAnomaly21",
        "RoadObsticle21",
        "fs_static",
        "RoadAnomaly",
        "FS_LostFound_full",
    ]

    for ds in dataset_folders:
        ds_path = os.path.join(args.datasets_root, ds)
        if not os.path.isdir(ds_path):
            print(f"\n=== {ds} ===")
            print(f"❌ Cartella non trovata: {ds_path}")
            continue

        # prova a puntare direttamente a labels_masks se esiste
        candidate = os.path.join(ds_path, "labels_masks")
        search_root = candidate if os.path.isdir(candidate) else ds_path

        mask_files = find_mask_files(search_root)
        # Filtro aggressivo: teniamo solo path che contengono 'label' o 'mask'
        mask_files = [p for p in mask_files if ("label" in p.lower() or "mask" in p.lower())]

        summarize_dataset(ds, mask_files, sample_n=args.sample_n, seed=args.seed, per_image_debug=3)

if __name__ == "__main__":
    main()
