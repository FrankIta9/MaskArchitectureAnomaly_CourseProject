# eval_ood_all.py
import os
import sys
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import pandas as pd
from torchvision.transforms import Compose, Resize, ToTensor
from argparse import ArgumentParser

# ---------------------------------------------------------------------
# Import dal progetto (devi lanciare lo script dalla root della repo)
# ---------------------------------------------------------------------
from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule


# -----------------------------
# Config "coerente col training"
# -----------------------------
NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"
MASKED_ATTN_ENABLED = True

IGNORE_VAL = 255
EPS = 1e-8


# -----------------------------
# Trasformazioni (come il tuo eval)
# -----------------------------
input_transform = Compose([
    Resize(IMG_SIZE, Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize(IMG_SIZE, Image.NEAREST),
])


# -----------------------------
# Utility metriche
# -----------------------------
def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    # labels: 1=OOD, 0=ID
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


# -----------------------------
# Loader pesi: supporta .ckpt e .bin/.pth
# -----------------------------
def load_weights_into_lightning(lm: LightningModule, weights_path: str, device: torch.device) -> None:
    ext = os.path.splitext(weights_path)[1].lower()
    obj = torch.load(weights_path, map_location=device)

    # 1) Se è un checkpoint Lightning: state_dict è dentro "state_dict"
    if ext == ".ckpt":
        state = obj.get("state_dict", obj)
        missing, unexpected = lm.load_state_dict(state, strict=False)
        print(f"[LOAD .ckpt] missing={len(missing)} unexpected={len(unexpected)}")
        return

    # 2) Se è .bin/.pth: spesso è già uno state_dict (o contiene "state_dict")
    state = obj.get("state_dict", obj)

    # Prova 2 strategie:
    # A) Carica nel LightningModule (chiavi tipo "network.xxx")
    missing, unexpected = lm.load_state_dict(state, strict=False)
    if len(missing) == 0:
        print(f"[LOAD .bin/.pth -> lm] missing={len(missing)} unexpected={len(unexpected)}")
        return

    # B) Carica direttamente nella network (chiavi "encoder.xxx", "decoder.xxx", ecc.)
    missing2, unexpected2 = lm.network.load_state_dict(state, strict=False)
    print(f"[LOAD .bin/.pth -> network] missing={len(missing2)} unexpected={len(unexpected2)}")


def build_model(device: torch.device, weights_path: str) -> torch.nn.Module:
    # Backbone + network
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=MASKED_ATTN_ENABLED,
    )

    # LightningModule wrapper per compatibilità coi checkpoint
    lm = LightningModule(
        network=network,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
        attn_mask_annealing_enabled=False,
        attn_mask_annealing_start_steps=None,
        attn_mask_annealing_end_steps=None,
        lr=1e-4,
        llrd=0.8,
        llrd_l2_enabled=True,
        lr_mult=1.0,
        weight_decay=0.05,
        poly_power=0.9,
        warmup_steps=(500, 1000),
        ckpt_path=None,                # carichiamo noi manualmente sotto
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    lm.to(device).eval()
    load_weights_into_lightning(lm, weights_path, device)

    model = lm.network.to(device).eval()
    return model


# -----------------------------
# GT: immagini -> labels_masks
# + mapping a {0,1,255}
# -----------------------------
def image_to_gt_path(img_path: str, dataset_name: str) -> str:
    gt = img_path.replace("/images/", "/labels_masks/")

    # estensioni specifiche (come nel tuo script)
    if dataset_name == "RoadObsticle21":
        gt = os.path.splitext(gt)[0] + ".png"  # webp -> png (o qualunque -> png)
    if dataset_name == "fs_static":
        gt = os.path.splitext(gt)[0] + ".png"
    if dataset_name == "RoadAnomaly":
        gt = os.path.splitext(gt)[0] + ".png"

    return gt


def map_gt_to_binary(gt_np: np.ndarray, dataset_name: str) -> np.ndarray:
    """
    Output: 0=ID, 1=OOD, 255=ignore
    Coerente col tuo dump:
      - RoadAnomaly21, RoadObsticle21, fs_static, FS_LostFound_full: già 0/1/255
      - RoadAnomaly: 0/2 (2 = OOD)
    """
    gt = gt_np.astype(np.int32)

    if dataset_name == "RoadAnomaly":
        # 0 = ID, 2 = OOD
        out = np.full_like(gt, IGNORE_VAL, dtype=np.int32)
        out[gt == 0] = 0
        out[gt == 2] = 1
        # qualsiasi altro valore -> ignore
        return out.astype(np.uint8)

    # altri dataset: ci aspettiamo 0/1/255
    out = np.full_like(gt, IGNORE_VAL, dtype=np.int32)
    out[gt == 0] = 0
    out[gt == 1] = 1
    out[gt == 255] = IGNORE_VAL
    # qualsiasi altro valore -> ignore
    return out.astype(np.uint8)


# -----------------------------
# Scoring: MSP / MaxLogit / Entropy / RBA
# (usiamo una combinazione "logits per pixel" robusta)
# -----------------------------
def compute_pixel_logits_and_probs(final_mask_logits: torch.Tensor,
                                   final_class_logits: torch.Tensor,
                                   temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    final_mask_logits: [1,Q,H,W]
    final_class_logits: [1,Q,C+1]
    Ritorna:
      pixel_logits: [C,H,W] (pseudo-logits)
      pixel_probs : [C,H,W] (softmax sui pseudo-logits)
    """

    # mask conf: sigmoid
    mask_probs = torch.sigmoid(final_mask_logits)[0]  # [Q,H,W]

    # class logits senza "no object"
    class_logits = final_class_logits[0, :, :-1]      # [Q,C]

    # Combino query->pixel come somma pesata dai mask_probs
    # pixel_logits[c,h,w] = sum_q class_logits[q,c] * mask_probs[q,h,w]
    pixel_logits = torch.einsum("qc,qhw->chw", class_logits, mask_probs)  # [C,H,W]

    pixel_logits = pixel_logits / float(temperature)
    pixel_probs = F.softmax(pixel_logits, dim=0)

    return pixel_logits, pixel_probs


def anomaly_from_logits(pixel_logits: torch.Tensor, pixel_probs: torch.Tensor, method: str) -> torch.Tensor:
    """
    output: anomaly_map [H,W], valori alti = più OOD
    """
    if method == "msp":
        msp = pixel_probs.max(dim=0).values
        return 1.0 - msp

    if method == "maxlogit":
        mx = pixel_logits.max(dim=0).values
        return -mx

    if method == "maxentropy":
        ent = -(pixel_probs * (pixel_probs + EPS).log()).sum(dim=0)
        return ent

    if method == "rba":
        # come nel tuo script (ma su pixel_logits)
        return -torch.tanh(pixel_logits).sum(dim=0)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# Eval di UN dataset e UN weights
# -----------------------------
def eval_one(dataset_root: str,
             dataset_name: str,
             weights_path: str,
             method: str,
             temperature: float,
             max_pixels: int,
             seed: int,
             device: torch.device) -> Dict:

    # 1) lista immagini
    images_dir = os.path.join(dataset_root, dataset_name, "images")
    patterns = [
        os.path.join(images_dir, "*.png"),
        os.path.join(images_dir, "*.jpg"),
        os.path.join(images_dir, "*.jpeg"),
        os.path.join(images_dir, "*.webp"),
    ]
    img_paths = []
    for p in patterns:
        img_paths.extend(glob.glob(p))
    img_paths = sorted(img_paths)

    if len(img_paths) == 0:
        return {
            "dataset": dataset_name,
            "weights": os.path.basename(weights_path),
            "method": method,
            "temperature": temperature,
            "n_valid": 0,
            "n_ood": 0,
            "ood_ratio": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "fpr95": float("nan"),
            "note": "NO_IMAGES_FOUND"
        }

    # 2) modello
    model = build_model(device, weights_path)

    # 3) accumulo scores/labels (streaming “leggero”)
    all_scores = []
    all_labels = []

    rng = np.random.default_rng(seed)

    for img_path in img_paths:
        # img
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = input_transform(img_pil).unsqueeze(0).float().to(device)

        # gt
        gt_path = image_to_gt_path(img_path, dataset_name)
        if not os.path.exists(gt_path):
            # se manca GT, salto
            continue
        gt_pil = Image.open(gt_path)
        gt_resized = target_transform(gt_pil)
        gt_np = np.array(gt_resized)
        gt_bin = map_gt_to_binary(gt_np, dataset_name)  # 0/1/255

        # infer
        with torch.no_grad():
            mask_logits_layers, class_logits_layers = model(img_tensor)
            final_mask = mask_logits_layers[-1]
            final_class = class_logits_layers[-1]

            final_mask = F.interpolate(final_mask, size=IMG_SIZE, mode="bilinear", align_corners=False)

            pixel_logits, pixel_probs = compute_pixel_logits_and_probs(final_mask, final_class, temperature)
            anomaly = anomaly_from_logits(pixel_logits, pixel_probs, method)  # [H,W]
            anomaly_np = anomaly.detach().float().cpu().numpy()

        # valid mask (non ignore)
        valid = (gt_bin != IGNORE_VAL)
        if valid.sum() == 0:
            continue

        scores = anomaly_np[valid].astype(np.float32)
        labels = gt_bin[valid].astype(np.uint8)  # 0/1

        # opzionale: campiona per non esplodere in RAM
        if max_pixels is not None and max_pixels > 0:
            # campionamento per immagine: massimo ~ max_pixels / Nimmagini (minimo 50k)
            per_img_cap = max(50000, max_pixels // max(1, len(img_paths)))
            if scores.shape[0] > per_img_cap:
                idx = rng.choice(scores.shape[0], size=per_img_cap, replace=False)
                scores = scores[idx]
                labels = labels[idx]

        all_scores.append(scores)
        all_labels.append(labels)

        # cleanup
        del img_tensor, mask_logits_layers, class_logits_layers, final_mask, final_class, pixel_logits, pixel_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_scores) == 0:
        return {
            "dataset": dataset_name,
            "weights": os.path.basename(weights_path),
            "method": method,
            "temperature": temperature,
            "n_valid": 0,
            "n_ood": 0,
            "ood_ratio": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "fpr95": float("nan"),
            "note": "NO_VALID_PIXELS"
        }

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # cap finale max_pixels
    if max_pixels is not None and max_pixels > 0 and scores.shape[0] > max_pixels:
        idx = rng.choice(scores.shape[0], size=max_pixels, replace=False)
        scores = scores[idx]
        labels = labels[idx]

    n_valid = int(labels.shape[0])
    n_ood = int(labels.sum())
    ood_ratio = float(n_ood / max(1, n_valid))

    # metriche (se non hai positivi o non hai negativi, alcune non sono definite)
    note = "OK"
    auroc = float("nan")
    auprc = float("nan")
    fpr95 = float("nan")

    if n_ood == 0 or n_ood == n_valid:
        note = "DEGENERATE_LABELS(no_pos_or_no_neg)"
    else:
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        fpr95 = float(fpr_at_95_tpr(scores, labels))

    return {
        "dataset": dataset_name,
        "weights": os.path.basename(weights_path),
        "method": method,
        "temperature": temperature,
        "n_valid": n_valid,
        "n_ood": n_ood,
        "ood_ratio": ood_ratio,
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "note": note
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--datasets_root", type=str, default="/content/drive/MyDrive/datasets")
    parser.add_argument("--weights", type=str, required=True,
                        help="Lista pesi separati da virgola: path1,path2,path3,... (supporta .ckpt e .bin/.pth)")
    parser.add_argument("--method", type=str, default="msp", choices=["msp", "maxlogit", "maxentropy", "rba"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_csv", type=str, default="eval_results.csv")
    args = parser.parse_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"datasets_root: {args.datasets_root}")
    print(f"method={args.method}  T={args.temperature}  max_pixels={args.max_pixels}  seed={args.seed}")

    datasets = ["RoadAnomaly21", "RoadObsticle21", "fs_static", "RoadAnomaly", "FS_LostFound_full"]
    weights_list = [w.strip() for w in args.weights.split(",") if len(w.strip()) > 0]

    rows = []
    for ds in datasets:
        print("\n" + "=" * 90)
        print(f"DATASET: {ds}")
        print("=" * 90)

        for w in weights_list:
            print(f"\n-> weights: {w}")
            row = eval_one(
                dataset_root=args.datasets_root,
                dataset_name=ds,
                weights_path=w,
                method=args.method,
                temperature=args.temperature,
                max_pixels=args.max_pixels,
                seed=args.seed,
                device=device
            )
            print(
                f"   AUROC={row['auroc'] if not math.isnan(row['auroc']) else 'NA'}  "
                f"AUPRC={row['auprc'] if not math.isnan(row['auprc']) else 'NA'}  "
                f"FPR95={row['fpr95'] if not math.isnan(row['fpr95']) else 'NA'}  "
                f"ood_ratio={row['ood_ratio'] if not math.isnan(row['ood_ratio']) else 'NA'}  note={row['note']}"
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    # salva CSV
    out_csv = args.out_csv
    df.to_csv(out_csv, index=False)
    print("\n✅ Salvato:", out_csv)

    # stampa tabella per dataset
    print("\n" + "#" * 90)
    print("TABELLE (una per dataset)")
    print("#" * 90 + "\n")

    for ds in datasets:
        sub = df[df["dataset"] == ds].copy()
        # ordino per AUPRC (desc)
        sub["auprc_rank"] = sub["auprc"].fillna(-1)
        sub = sub.sort_values("auprc_rank", ascending=False).drop(columns=["auprc_rank"])

        # tabellina compatta
        view = sub[["weights", "auroc", "auprc", "fpr95", "n_valid", "n_ood", "ood_ratio", "note"]].copy()
        print(f"\n## {ds}")
        print(view.to_markdown(index=False))

    # opzionale: tabella globale ordinata per AUPRC
    global_view = df.copy()
    global_view["auprc_rank"] = global_view["auprc"].fillna(-1)
    global_view = global_view.sort_values(["dataset", "auprc_rank"], ascending=[True, False]).drop(columns=["auprc_rank"])

    print("\n\n## (Extra) Tutto insieme ordinato per dataset/AUPRC")
    print(global_view[["dataset", "weights", "auroc", "auprc", "fpr95", "ood_ratio", "note"]].to_markdown(index=False))


if __name__ == "__main__":
    main()
