# Sanity Check S1: Paste Rate & Size Distribution (OBBLIGATORIO)

## Obiettivo
Verificare che il paste COCO funzioni con frequenza e dimensioni adeguate prima di fare training lungo.

## Problema Identificato
Dai risultati A2:
- **Solo 20.5% batch con OOD** (target: 50-70%, possibilmente 80%)
- **ood_ratio medio: 0.1577%** (target: 0.5-2%)
- **Max ~1%, Min ~0.009%**

**Questo è CRITICO**: il segnale OOD è troppo raro e microscopico. La loss "vede" quasi sempre ID puro, quindi energy/logitnorm non possono migliorare significativamente.

## Check da Fare

### Su 500-1000 batch, loggare:

1. **% batch con OOD**
   - Target: ≥50-70% (anche 80% se gestito bene)
   - Se <30% → problema serio, aumenta `paste_probability`

2. **Distribuzione di ood_ratio** (P50/P90/P99)
   - Target: P90 ≥0.5%, P50 ≥0.2%
   - Se P90 <0.2% → oggetti troppo piccoli, aumenta `min_scale` o `coco_min_area`

3. **Area media degli oggetti incollati**
   - Target: almeno 0.5-2% dell'immagine
   - Se <0.1% → oggetti troppo piccoli, non servono per imparare

4. **Numero medio di oggetti per batch con OOD**
   - Target: 1-3 oggetti per batch con OOD
   - Se <1 → troppo pochi oggetti per batch

## Fix Consigliati (da testare con A2)

### Fix 1: Aumenta paste_probability
```yaml
paste_probability: 0.50  # Da 0.25 a 0.50 (50% batch hanno OOD)
```

### Fix 2: Aumenta min_scale per oggetti più grandi
```yaml
min_scale: 0.15  # Da 0.10 a 0.15 (oggetti più grandi)
max_scale: 0.30  # Da 0.20 a 0.30 (range più ampio)
```

### Fix 3: Riduci filtri troppo stretti
- Se usi `use_drivable_regions: true`, considera di allentare i vincoli
- Se `drivable_percentage >= 0.6` è troppo stretto, prova 0.4-0.5

### Fix 4: Aumenta coco_min_area per oggetti più grandi
```yaml
coco_min_area: 2000  # Da 1200 a 2000 (solo oggetti più grandi)
```

## Script per Check S1

Usa `sanity_check_a2.py` con `--num_batches 500` o `1000`:

```bash
python sanity_check_a2.py \
    --config eomt/configs/dinov2/cityscapes/semantic/eomt_base_1024_oe_safe.yaml \
    --data_path /path/to/Cityscapes \
    --coco_path /path/to/coco2017_zips \
    --num_batches 500
```

Aggiungi logging per:
- P50/P90/P99 di ood_ratio
- Area media oggetti incollati
- Numero medio oggetti per batch con OOD

## Valutazione

✅ **PASS**: Se P90 ood_ratio ≥0.5% e %batch con OOD ≥50%
❌ **FAIL**: Se P90 <0.2% o %batch con OOD <30%

**Se FAIL**: Non procedere con training lungo. Prima risolvi i parametri di paste.
