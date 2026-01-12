# Guida Sanity Checks Completi

Questa guida elenca tutti i sanity check da fare PRIMA di fare training lungo, per evitare run buttate.

## Sanity Check A: Dataloader Base (✅ FATTO)

**Obiettivo**: Verificare che ood_mask passa correttamente nel dataloader

**Script**: `sanity_check_a.py`

**Check**:
- img.shape, ood_mask.shape, unique+counts
- device/dtype corretti
- 3 overlay PNG

**Risultato**: ✅ PASS (con fix a `_filter`)

**Nota**: È normale che i primi 3 batch siano tutti 0 se solo ~20% hanno OOD.

---

## Sanity Check A2: Paste Rate & Size (✅ FATTO, ⚠️ PROBLEMA TROVATO)

**Obiettivo**: Contare quanti batch hanno OOD e verificare dimensioni

**Script**: `sanity_check_a2.py --num_batches 200`

**Risultati attuali**:
- ❌ **20.5% batch con OOD** (target: ≥50-70%)
- ❌ **ood_ratio medio: 0.1577%** (target: ≥0.5-2%)
- ❌ **Max ~1%, Min ~0.009%** (troppo piccolo)

**Problema critico**: OOD troppo raro e microscopico. Il segnale è insufficiente per imparare separazione energetica.

**Fix necessario** (vedi `SANITY_CHECK_S1_PASTE_RATE.md`):
1. Aumenta `paste_probability` (0.25 → 0.50)
2. Aumenta `min_scale` (0.10 → 0.15)
3. Aumenta `coco_min_area` (1200 → 2000)
4. Riduci filtri troppo stretti (drivable regions, ecc.)

---

## Sanity Check S1: Paste Rate & Size Distribution (OBBLIGATORIO)

**Obiettivo**: Verificare distribuzione completa di paste su 500-1000 batch

**Script**: Estendi `sanity_check_a2.py` con `--num_batches 500`

**Metriche**:
1. % batch con OOD (target: ≥50-70%)
2. P50/P90/P99 di ood_ratio (target: P90 ≥0.5%)
3. Area media oggetti incollati
4. Numero medio oggetti per batch con OOD

**Valutazione**:
- ✅ PASS: P90 ≥0.5% e %batch ≥50%
- ❌ FAIL: P90 <0.2% o %batch <30%

**Vedi**: `SANITY_CHECK_S1_PASTE_RATE.md` per dettagli

---

## Sanity Check S2: Allineamento (da fare su 20 esempi)

**Obiettivo**: Verificare allineamento ood_mask ↔ immagine dopo TUTTE le trasformazioni

**Script**: Estendi `sanity_check_a.py` per salvare 20 overlay

**Check**:
- Overlay PNG su 20 esempi con OOD "grande" (>1% pixel)
- Verifica visiva che rosso (OOD) si allinei con oggetti incollati
- Dopo resize/crop/pad: ood_mask deve rimanere allineata

**Nota**: Attualmente Outlier Exposure viene applicato DOPO le trasformazioni geometriche, quindi ood_mask è già sulla dimensione corretta. Ma verifica comunque che le trasformazioni torchvision v2 gestiscano ood_mask correttamente.

---

## Sanity Check S3: Semantics Conflict (IMPORTANTISSIMO)

**Obiettivo**: Verificare che pixel OOD non abbiano target semantic di classi Cityscapes

**Problema**: Se pixel OOD hanno target semantic (es. "road"), mentre energy/logitnorm cercano "incerto/alto", si crea conflitto distruttivo.

**Check**:
1. Estrai ood_mask e target semantic da un batch
2. Verifica che pixel OOD abbiano:
   - target = 255 (ignore), OPPURE
   - target = C (no-object, dove C=num_classes)
3. NON devono avere target in [0, C-1] (classi Cityscapes)

**Fix se conflitto**:
- Imposta target = 255 (ignore) per pixel OOD, OPPURE
- Imposta target = C (no-object) per pixel OOD

**Vedi**: `SANITY_CHECK_S3_SEMANTICS_CONFLICT.md` per dettagli

---

## Sanity Check S4: Energy Separation Check (Early, Cheap)

**Obiettivo**: Verificare che energy separi ID/OOD nei primi 1-2k step

**Quando**: Dopo aver sistemato S1-S3, fai training corto (1-2k step)

**Metriche da monitorare** (già implementate in `lightning_module.py`):
- `dbg/energy_id_mean`
- `dbg/energy_ood_mean`
- `dbg/energy_sep = energy_ood_mean - energy_id_mean`

**Check**:
- `energy_sep` deve diventare **positivo** e **non oscillare a caso**
- Se `energy_ood_mean <= energy_id_mean` quasi sempre → problema
- Se `energy_sep` collassa a 0 o negativo → problema

**Se FAIL**: 
- Verifica S3 (conflitto semantics)
- Verifica che ood_mask sia allineata correttamente
- Verifica che margini siano coerenti (Task 4)

---

## Sanity Check S5: Ablation Mini (2 ore, non 2 giorni)

**Obiettivo**: Confronto rapido baseline vs energy+logitnorm

**Setup**: 2 run da 2-5k step (NON 30 epoche!)

1. **Baseline OE senza energy/logitnorm**:
   - `energy_ood_enabled: false`
   - `logit_norm_enabled: false`
   - Monitora: `dbg/energy_id_mean`, `dbg/energy_ood_mean`, `dbg/energy_sep`

2. **Con energy + margini dinamici**:
   - `energy_ood_enabled: true`
   - `energy_ood_max_weight: 0.0003`
   - Monitora: stesso set di metriche

**Confronto**:
- Separazione energetica (`energy_sep`): deve migliorare con energy
- Loss stability: non deve esplodere
- **NON** guardare metriche finali (mIoU, AUPRC) - troppo pochi step

**Se non si vede segnale qui**: Non ha senso fare 30 epoche. Risolvi prima S1-S4.

---

## Ordine di Esecuzione

1. ✅ **S-A**: Dataloader base (3 batch) - FATTO
2. ✅ **S-A2**: Paste rate base (200 batch) - FATTO, PROBLEMA TROVATO
3. 🔥 **S-S1**: Paste rate completo (500-1000 batch) - **OBBLIGATORIO PRIMA DI TRAINING**
4. 🔥 **S-S3**: Semantics conflict check - **OBBLIGATORIO PRIMA DI TRAINING**
5. **S-S2**: Allineamento (20 esempi) - Da fare dopo fix S1
6. **S-S4**: Energy separation (1-2k step) - Dopo S1-S3
7. **S-S5**: Ablation mini (2-5k step) - Dopo S4

---

## Conclusioni Attuali

**Problemi identificati**:
1. ❌ OOD troppo raro (20.5% batch) e piccolo (0.1577% ratio)
2. ⚠️ Necessario verificare S3 (semantics conflict)

**Azioni immediate**:
1. Fix parametri paste (vedi S-S1)
2. Verifica S3 (semantics conflict)
3. Rifa S-A2 dopo fix
4. Se S-A2 passa, procedi con S-S4 e S-S5

**NON procedere con training lungo finché S-S1 e S-S3 non passano!**
