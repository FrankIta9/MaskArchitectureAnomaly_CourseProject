# eval_ood.py
import os
import sys
import glob
import random
import argparse
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_curve
from torchvision.transforms import Compose, Resize, ToTensor


# -----------------------------------------------------------------------------
# PATH SETUP (adatta se necessario)
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_DIR, "..")
EOMT_ROOT = os.path.join(PROJECT_ROOT, "eomt")

if EOMT_ROOT not in sys.path:
    sys.path.insert(0, EOMT_ROOT)

from models.vit import ViT
from models.eomt import EoMT
from training.lightning_module import LightningModule


# -----------------------------------------------------------------------------
# DEFAULTS (devono matchare il tuo training)
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 19
IMG_SIZE = (1024, 1024)
NUM_QUERIES = 100
NUM_BLOCKS = 3
BACKBONE_NAME = "vit_base_patch14_reg4_dinov2"

IGNORE_INDEX = 255


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """FPR quando TPR >= 95%."""
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    idxs = np.where(tpr >= 0.95)[0]
    if len(idxs) == 0:
        return 1.0
    return float(fpr[idxs[0]])


def build_transforms(img_size: Tuple[int, int]):
    input_transform = Compose([
        Resize(img_size, Image.BILINEAR),
        ToTensor(),
    ])
    target_transform = Compose([
        Resize(img_size, Image.NEAREST),
    ])
    return input_transform, target_transform


def infer_gt_path(img_path: str) -> str:
    """
    Regola base: images -> labels_masks.
    Adatta qui se la tua struttura cambia.
    """
    pathGT = img_path.replace("images", "labels_masks")

    # fix estensioni note
    if "RoadObsticle21" in pathGT:
        pathGT = pathGT.replace(".webp", ".png")
    if "fs_static" in pathGT:
        pathGT = pathGT.replace(".jpg", ".png")
        pathGT = pathGT.replace(".jpeg", ".png")
    if "RoadAnomaly" in pathGT:
        pathGT = pathGT.replace(".jpg", ".png")
        pathGT = pathGT.replace(".jpeg", ".png")

    return pathGT


def detect_dataset_from_path(path: str) -> Optional[str]:
    names = ["RoadAnomaly21", "RoadObsticle21", "fs_static", "RoadAnomaly", "FS_LostFound_full"]
    for n in names:
        if n in path:
            return n
    return None


def map_gt_to_binary(pathGT: str, gt: np.ndarray) -> np.ndarray:
    """
    Output standard:
      0 = ID
      1 = OOD
      255 = IGNORE
    Basato SUI TUOI UNIQUE che hai stampato.
    """
    ds = detect_dataset_from_path(pathGT)
    if ds is None:
        raise ValueError(f"Dataset non riconosciuto dal path: {pathGT}")

    gt = gt.astype(np.int64)

    if ds == "RoadAnomaly":
        # unique: {0,2}
        out = np.full_like(gt, 255, dtype=np.uint8)
        out[gt == 0] = 0
        out[gt == 2] = 1
        return out

    # Gli altri: {0,1,255} (già pronto)
    out = gt.copy()

    allowed = {0, 1, 255}
    uniq = set(np.unique(out).tolist())
    if not uniq.issubset(allowed):
        strange = sorted(list(uniq - allowed))
        print(f"⚠️ Valori inattesi in GT ({ds}): {strange} in {pathGT}. Li setto a 255 (ignore).")
        for v in strange:
            out[out == v] = 255

    return out.astype(np.uint8)


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
def load_eomt_model_from_ckpt(ckpt_path: str, device: torch.device) -> LightningModule:
    """
    Inizializza network + LightningModule e carica pesi con la stessa logica del progetto.
    Ritorna il LightningModule (non solo network) così possiamo riusare helper se servono.
    """
    encoder = ViT(img_size=IMG_SIZE, backbone_name=BACKBONE_NAME)
    network = EoMT(
        encoder=encoder,
        num_classes=NUM_CLASSES,
        num_q=NUM_QUERIES,
        num_blocks=NUM_BLOCKS,
        masked_attn_enabled=True,
    )

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
        ckpt_path=ckpt_path,
        delta_weights=False,
        load_ckpt_class_head=True,
    )

    lm = lm.to(device).eval()
    return lm


# -----------------------------------------------------------------------------
# SEMANTIC PER-PIXEL LOGITS (MaskFormer-style)
# -----------------------------------------------------------------------------
@torch.no_grad()
def to_per_pixel_logits_semantic(mask_logits_bqhw: torch.Tensor,
                                 class_logits_bqc1: torch.Tensor,
                                 eps: float = 1e-8) -> torch.Tensor:
    """
    Implementazione fedele alla formula del notebook:
      per_pixel_logits[c,h,w] = sum_i p_i(c) * sigmoid(mask_i[h,w])

    - class_logits: [B,Q,C+1]
    - mask_logits:  [B,Q,h,w] -> upsample a IMG_SIZE prima di chiamare

    Ritorna: [B,C,H,W] "logits-like" (in realtà scores).
    """
    # class probs (include "no object" last, poi lo togliamo)
    class_probs = F.softmax(class_logits_bqc1, dim=-1)[..., :-1]  # [B,Q,C]
    mask_probs = torch.sigmoid(mask_logits_bqhw)                  # [B,Q,H,W]

    # einsum: (B,Q,C) x (B,Q,H,W) -> (B,C,H,W)
    per_pixel = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    # evitiamo valori 0 per log / stabilità in alcuni metodi
    per_pixel = torch.clamp(per_pixel, min=eps, max=1e6)
    return per_pixel


def compute_anomaly_map(per_pixel_logits: torch.Tensor,
                        method: str,
                        temperature: float = 1.0,
                        eps: float = 1e-8) -> torch.Tensor:
    """
    per_pixel_logits: [C,H,W] (scores per classe)
    Output: anomaly_map [H,W] dove più alto = più anomalo.
    """
    x = per_pixel_logits / max(temperature, eps)

    if method == "msp":
        probs = F.softmax(x, dim=0)
        msp = probs.max(dim=0).values
        return 1.0 - msp

    if method == "entropy":
        probs = F.softmax(x, dim=0)
        ent = -(probs * (probs + eps).log()).sum(dim=0)
        return ent

    if method == "maxlogit":
        # più basso maxlogit => più anomalo
        maxlogit = x.max(dim=0).values
        return -maxlogit

    if method == "energy":
        # Energy: E = -T logsumexp(x/T) (qui T già applicato sopra)
        # con la convenzione: OOD tende ad avere energy più ALTA (meno negativa) => anomaly = energy
        energy = -torch.logsumexp(x, dim=0)
        return energy

    raise ValueError(f"Metodo non supportato: {method}")


# -----------------------------------------------------------------------------
# MAIN EVAL
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_glob", type=str, required=True,
                        help="Glob path immagini, es: /.../RoadAnomaly21/images/*.png")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path completo al checkpoint .ckpt")
    parser.add_argument("--method", type=str, default="msp",
                        choices=["msp", "entropy", "maxlogit", "energy"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=2_000_000,
                        help="Campiona al massimo N pixel validi totali per metriche (evita RAM enorme).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--debug_first_n", type=int, default=3,
                        help="Stampa sanity-check mapping su prime N immagini valide.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")
    print(f"CKPT: {args.ckpt}")
    print(f"Method: {args.method} | T={args.temperature} | max_pixels={args.max_pixels}")

    input_transform, target_transform = build_transforms(IMG_SIZE)

    # carica modello
    lm = load_eomt_model_from_ckpt(args.ckpt, device)
    net = lm.network

    # file list
    file_list = sorted(glob.glob(os.path.expanduser(args.images_glob)))
    if len(file_list) == 0:
        raise RuntimeError(f"Nessuna immagine trovata con glob: {args.images_glob}")
    print(f"Immagini trovate: {len(file_list)}")

    # accumulo (con campionamento)
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    collected = 0
    processed = 0
    skipped_missing_gt = 0
    skipped_no_valid = 0
    debug_printed = 0

    for idx, img_path in enumerate(file_list):
        pathGT = infer_gt_path(img_path)
        if not os.path.exists(pathGT):
            skipped_missing_gt += 1
            continue

        # load image
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = input_transform(img_pil).unsqueeze(0).to(device, dtype=torch.float32)

        # forward
        with torch.no_grad():
            mask_logits_layers, class_logits_layers = net(img_tensor)
            mask_logits = mask_logits_layers[-1]      # [B,Q,h,w]
            class_logits = class_logits_layers[-1]    # [B,Q,C+1]

            mask_logits = F.interpolate(mask_logits, size=IMG_SIZE, mode="bilinear", align_corners=False)
            per_pixel_bchw = to_per_pixel_logits_semantic(mask_logits, class_logits)  # [B,C,H,W]
            per_pixel = per_pixel_bchw[0]  # [C,H,W]

            anomaly = compute_anomaly_map(per_pixel, args.method, args.temperature)  # [H,W]
            anomaly_np = anomaly.detach().cpu().numpy().astype(np.float32)

        # load gt
        gt_pil = Image.open(pathGT)
        gt_pil = target_transform(gt_pil)
        gt_np = np.array(gt_pil)

        # mapping -> {0,1,255}
        gt_bin = map_gt_to_binary(pathGT, gt_np)

        # valid pixels
        valid = (gt_bin != IGNORE_INDEX)
        if valid.sum() == 0:
            skipped_no_valid += 1
            continue

        # sanity check su prime N
        if debug_printed < args.debug_first_n:
            uniq, cnt = np.unique(gt_bin, return_counts=True)
            print(f"\n[DEBUG] {img_path}")
            print(f"        GT uniq after mapping: {list(zip(uniq.tolist(), cnt.tolist()))}")
            debug_printed += 1

        # estrai label e score validi
        scores_v = anomaly_np[valid].reshape(-1)
        labels_v = gt_bin[valid].reshape(-1).astype(np.uint8)

        # campionamento per non esplodere
        remaining = args.max_pixels - collected
        if remaining <= 0:
            break

        if scores_v.shape[0] > remaining:
            # sample casuale (riproducibile con seed globale)
            sel = np.random.choice(scores_v.shape[0], size=remaining, replace=False)
            scores_v = scores_v[sel]
            labels_v = labels_v[sel]

        all_scores.append(scores_v)
        all_labels.append(labels_v)
        collected += scores_v.shape[0]
        processed += 1

        if (idx + 1) % 20 == 0:
            print(f"Progress {idx+1}/{len(file_list)} | processed={processed} | collected_pixels={collected}", end="\r")

    print("\n")
    print(f"Processed images: {processed}")
    print(f"Skipped (missing GT): {skipped_missing_gt}")
    print(f"Skipped (no valid pixels): {skipped_no_valid}")
    print(f"Collected valid pixels: {collected}")

    if collected == 0:
        raise RuntimeError("Nessun pixel valido raccolto. Controlla path GT e mapping.")

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # labels binarie: 0/1 (già così)
    # metriche
    auprc = average_precision_score(labels, scores)
    fpr95 = fpr_at_95_tpr(scores, labels)

    # stats utili
    frac_ood = float((labels == 1).mean())
    print("============================================================")
    print(f"[EVAL] method={args.method} T={args.temperature}")
    print(f"AUPRC: {auprc*100:.2f}%")
    print(f"FPR@95TPR: {fpr95*100:.2f}%")
    print(f"OOD fraction in sampled pixels: {frac_ood*100:.2f}%")
    print("============================================================")


if __name__ == "__main__":
    main()
