#!/usr/bin/env python3
"""
Script per estrarre solo i pesi del modello da un checkpoint Lightning.
Rimuove optimizer state, scheduler state, epoch, global_step, ecc.
Utile per ripartire da capo dopo problemi di instabilità numerica.

Usage:
    python extract_model_weights.py \
        --input /path/to/checkpoint.ckpt \
        --output /path/to/model_weights.bin

Esempio:
    python extract_model_weights.py \
        --input /content/drive/MyDrive/eomt_oe_frozen_backbone/epoch15/epoch15.ckpt \
        --output /content/drive/MyDrive/eomt_oe_frozen_backbone/epoch15_weights_only.bin
"""

import torch
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def extract_model_weights(input_ckpt: str, output_path: str, clean_complex: bool = True):
    """
    Estrae solo i pesi del modello da un checkpoint Lightning.
    
    Args:
        input_ckpt: Path al checkpoint Lightning (.ckpt)
        output_path: Path dove salvare i pesi del modello (.bin)
        clean_complex: Se True, pulisce parametri complessi/NaN/Inf
    
    Returns:
        Numero di chiavi estratte
    """
    input_path = Path(input_ckpt)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {input_path}")
    
    logging.info(f"Caricamento checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=True)
    
    # Estrai solo state_dict (pesi del modello)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Se non ha chiave standard, assume che tutto il checkpoint sia lo state_dict
        state_dict = checkpoint
    
    # Filtra chiavi non necessarie (criterion.empty_weight, optimizer state, ecc.)
    # Mantieni solo pesi del modello (network.*)
    model_weights = {}
    filtered_keys = 0
    cleaned_keys = 0
    
    for k, v in state_dict.items():
        # Skip optimizer state, scheduler state, epoch, global_step, ecc.
        if any(skip in k for skip in [
            "optimizer",
            "lr_scheduler",
            "lr_schedulers",
            "epoch",
            "global_step",
            "state_dict",  # Skip se c'è uno state_dict annidato
            "hyper_parameters",
            "callbacks",
            "loops",
            "criterion.empty_weight",  # Buffer, non necessario
        ]):
            filtered_keys += 1
            continue
        
        # Skip se non è un tensor (metadata, ecc.)
        if not isinstance(v, torch.Tensor):
            filtered_keys += 1
            continue
        
        # CLEAN COMPLEX/NaN/Inf VALUES se richiesto
        if clean_complex:
            # Convert complex to real (take real part)
            if v.is_complex():
                v = v.real
                cleaned_keys += 1
                logging.warning(f"⚠️ Converted complex to real: {k}")
            
            # Replace NaN/Inf with zeros
            if not torch.isfinite(v).all():
                num_invalid = (~torch.isfinite(v)).sum().item()
                v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
                cleaned_keys += 1
                logging.warning(f"⚠️ Cleaned {num_invalid} NaN/Inf values in: {k}")
            
            # Ensure tensor is real float (not complex)
            if not v.dtype.is_floating_point:
                v = v.float()
        
        model_weights[k] = v
    
    logging.info(f"Estratte {len(model_weights)} chiavi del modello")
    logging.info(f"Filtrate {filtered_keys} chiavi (optimizer/scheduler/metadata)")
    if cleaned_keys > 0:
        logging.warning(f"⚠️ Pulite {cleaned_keys} chiavi con valori complessi/NaN/Inf")
    
    # Crea directory output se non esiste
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Salva solo i pesi del modello
    logging.info(f"Salvataggio pesi modello in: {output_path}")
    torch.save(model_weights, output_path)
    
    # Calcola dimensione file
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logging.info(f"✅ Pesi modello salvati: {file_size_mb:.2f} MB")
    
    return len(model_weights)


def main():
    parser = argparse.ArgumentParser(
        description="Estrae solo i pesi del modello da un checkpoint Lightning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path al checkpoint Lightning (.ckpt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path dove salvare i pesi del modello (.bin)"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Non pulire parametri complessi/NaN/Inf (non raccomandato)"
    )
    
    args = parser.parse_args()
    
    try:
        num_keys = extract_model_weights(
            input_ckpt=args.input,
            output_path=args.output,
            clean_complex=not args.no_clean
        )
        print(f"\n✅ SUCCESS: Estratte {num_keys} chiavi del modello")
        print(f"📁 File salvato: {args.output}")
        print(f"\n💡 Ora puoi usare questo file come model.ckpt_path nel config:")
        print(f"   ckpt_path: \"{args.output}\"")
    except Exception as e:
        logging.error(f"❌ ERRORE: {e}")
        raise


if __name__ == "__main__":
    main()
