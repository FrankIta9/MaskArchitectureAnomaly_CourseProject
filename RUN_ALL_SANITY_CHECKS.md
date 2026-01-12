# Come Eseguire Tutti i Sanity Check

Questa guida ti spiega come eseguire tutti i sanity check in ordine.

## Setup

Tutti gli script richiedono:
- Config YAML (es. `eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml`)
- Path a Cityscapes dataset
- Path a COCO dataset (opzionale, ma necessario per OE)

## Ordine di Esecuzione

### 1. Sanity Check A: Dataloader Base (✅ OPZIONALE, già fatto)

Verifica che ood_mask passa correttamente nel dataloader.

```bash
python sanity_check_a.py \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips \
    --num_batches 3
```

**Tempo**: ~1 minuto  
**Costo**: 0 (solo dataloader, nessun training)

---

### 2. Sanity Check A2/S1: Paste Rate & Size Distribution (🔥 OBBLIGATORIO)

Verifica frequenza e dimensioni del paste COCO.

```bash
python sanity_check_a2.py \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips \
    --num_batches 500
```

**Tempo**: ~10-15 minuti  
**Costo**: 0 (solo dataloader)

**Requisiti S1**:
- ✅ ≥50% batch con OOD
- ✅ P90 ood_ratio ≥0.5%
- ✅ Mean ood_ratio > 0.0001

**Se FAIL**: 
- Aumenta `paste_probability` (es. 0.25 → 0.50)
- Aumenta `min_scale` (es. 0.10 → 0.15)
- Aumenta `coco_min_area` (es. 1200 → 2000)
- Riduci filtri troppo stretti (drivable regions, ecc.)

**⚠️ NON procedere con training lungo finché S1 non passa!**

---

### 3. Sanity Check S2: Allineamento (OPZIONALE, ma consigliato)

Verifica allineamento ood_mask ↔ immagine su 20 esempi con OOD "grande".

```bash
python sanity_check_s2.py \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips \
    --num_examples 20 \
    --min_ood_ratio 0.01
```

**Tempo**: ~5 minuti  
**Costo**: 0 (solo dataloader)

**Output**: 20 overlay PNG in `./sanity_check_s2_overlays/`

**Verifica visivamente**: Il rosso (OOD) deve allinearsi con gli oggetti incollati.

---

### 4. Sanity Check S3: Semantics Conflict (🔥 OBBLIGATORIO)

Verifica che ood_mask sia strutturato correttamente.

```bash
python sanity_check_s3.py \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips \
    --num_batches 50
```

**Tempo**: ~2 minuti  
**Costo**: 0 (solo dataloader)

**⚠️ IMPORTANTE**: Questo check verifica solo la struttura di ood_mask.
**Devi verificare manualmente in `mask_classification_loss.py`** che i pixel OOD siano:
1. Ignorati (ignore_idx=255) nella loss semantica, OPPURE
2. Target = num_classes (no-object)
3. **NON** devono avere target in [0, num_classes-1] (classi Cityscapes normali)

**Se i pixel OOD hanno target semantic di classi Cityscapes, si crea un conflitto distruttivo!**

---

### 5. Sanity Check S4: Energy Separation (Early, Cheap)

Verifica separazione energy ID/OOD nei primi 1-2k step.

**⚠️ Richiede training corto (1-2k step)!**

```bash
# Modifica config: max_epochs molto basso (es. 1 epoch = ~2000 step)
# Oppure usa max_steps se supportato

python -m eomt.main \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips
```

**Monitora in WandB**:
- `dbg/energy_id_mean`
- `dbg/energy_ood_mean`
- `dbg/energy_sep = energy_ood_mean - energy_id_mean`

**Check**:
- ✅ `energy_sep` deve diventare **positivo** e **non oscillare a caso**
- ❌ Se `energy_ood_mean <= energy_id_mean` quasi sempre → problema
- ❌ Se `energy_sep` collassa a 0 o negativo → problema

**Se FAIL**: 
- Verifica S3 (conflitto semantics)
- Verifica che ood_mask sia allineata correttamente
- Verifica che margini siano coerenti (Task 4)

**Tempo**: ~30-60 minuti  
**Costo**: ~1 ora Colab GPU

---

### 6. Sanity Check S5: Ablation Mini (2 ore, non 2 giorni)

Confronto rapido baseline vs energy+logitnorm.

**⚠️ Richiede 2 run da 2-5k step!**

**Run 1: Baseline OE senza energy/logitnorm**

Modifica config:
```yaml
energy_ood_enabled: false
logit_norm_enabled: false
max_epochs: 1  # ~2000 step
```

```bash
python -m eomt.main \
    --config <config_baseline.yaml> \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips
```

**Run 2: Con energy + margini dinamici**

Modifica config:
```yaml
energy_ood_enabled: true
energy_ood_max_weight: 0.0003
logit_norm_enabled: false  # o true, a seconda di cosa testi
max_epochs: 1  # ~2000 step
```

```bash
python -m eomt.main \
    --config <config_energy.yaml> \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips
```

**Confronto**:
- Separazione energetica (`energy_sep`): deve migliorare con energy
- Loss stability: non deve esplodere
- **NON** guardare metriche finali (mIoU, AUPRC) - troppo pochi step

**Se non si vede segnale qui**: Non ha senso fare 30 epoche. Risolvi prima S1-S4.

**Tempo**: ~2-4 ore (2 run)  
**Costo**: ~2-4 ore Colab GPU

---

## Riassunto

### Check Obbligatori (prima di training lungo):
1. ✅ **S1 (A2)**: Paste rate & size distribution (500 batch)
2. ✅ **S3**: Semantics conflict check (50 batch)

### Check Consigliati:
3. **S2**: Alignment check (20 esempi)
4. **S4**: Energy separation (1-2k step training)
5. **S5**: Ablation mini (2-5k step training, 2 run)

### Tempo Totale:
- **Solo obbligatori**: ~15-20 minuti (solo dataloader)
- **Con S2**: ~20-25 minuti
- **Con S4**: ~1-2 ore (include training corto)
- **Con S5**: ~3-5 ore (include 2 training corti)

### Costo:
- **S1-S3**: 0 (solo dataloader)
- **S4**: ~1 ora Colab GPU
- **S5**: ~2-4 ore Colab GPU
- **Totale**: ~3-5 ore Colab GPU (molto meno di un training completo da 30 epoche!)

---

## Ordine Consigliato

1. **S1 (A2)** → Se PASS, continua
2. **S3** → Se PASS, continua
3. **S2** → Verifica visivamente gli overlay
4. **S4** → Se PASS, continua
5. **S5** → Se PASS, procedi con training lungo

**Se qualsiasi check FAIL prima di S4**: Risolvi i problemi prima di procedere.
