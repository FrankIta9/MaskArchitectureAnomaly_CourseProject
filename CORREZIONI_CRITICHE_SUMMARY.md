# Summary Correzioni Critiche Implementate

## ✅ Correzioni Completate

### 🔴 Punto Critico #2: LogitNorm su OOD (CRITICO)

**Problema**: LogitNorm veniva applicata a tutte le query, inclusi OOD pasted, appiattendo la confidenza proprio dove vogliamo incertezza/energia alta.

**Correzione Implementata**:
- **File**: `eomt/training/mask_classification_loss.py`
- **Modifica**: Disabilitare LogitNorm quando `ood_mask` è disponibile
- **Logica**: Check `ood_mask_available` prima di applicare LogitNorm (linea 132-136)
- **Risultato**: LogitNorm applicata solo su ID puro (instance/panoptic), non su semantic con OE

**Codice**:
```python
# STEP 0: Check if ood_mask is available (used for multiple decisions)
ood_mask_available = targets and any("ood_mask" in target for target in targets)

# STEP 2: Apply Logit Normalization (modifies IN-PLACE)
# PUNTO CRITICO #2: Disabilitare LogitNorm quando OOD è presente
if self.logit_norm_enabled and class_queries_logits is not None and not ood_mask_available:
    # Apply LogitNorm only when no OOD (instance/panoptic segmentation)
    # ... existing LogitNorm code ...
```

---

### 🔴 Punto Critico #3: Energy Loss - Margini Reali (CRITICO - Task 4)

**Problema**: Margini hardcoded (`m_in=-25.0`, `m_out=-7.0`) possono essere totalmente fuori scala per il modello.

**Correzione Implementata**:
- **File 1**: `eomt/training/lightning_module.py`
  - Accumulazione energy values durante primi 5k step (linee 429-458)
  - Calcolo margini da distribuzioni (mean, std, percentili) (linee 487-569)
  - Logging statistiche complete

- **File 2**: `eomt/training/energy_ood_loss.py`
  - Metodo `update_margins()` per aggiornare margini dinamicamente (linee 63-74)

**Logica**:
1. Durante primi 5000 step: accumula `E_id` e `E_ood` (con sampling per evitare memory explosion)
2. Dopo 5000 step: calcola statistiche (mean, std, P10/P50/P90)
3. Imposta margini:
   - `m_in = P90(E_id)` (90° percentile di ID energy)
   - `m_out = P10(E_ood)` (10° percentile di OOD energy)
4. Verifica: `m_out > m_in` (altrimenti usa gap minimo)
5. Aggiorna `EnergyOODLoss` con nuovi margini

**Check**: Dopo ~5000 step, vedi log:
```
📊 Task 4 - Energy Statistics (step 5000):
  E_id: mean=..., std=..., P10=..., P50=..., P90=...
  E_ood: mean=..., std=..., P10=..., P50=..., P90=...
✅ Task 4: Updated energy margins - m_in=..., m_out=...
```

---

### 🟡 Punto Critico #1: Allineamento ood_mask ↔ logits (IMPORTANTE)

**Problema**: Se `ood_mask` non è allineato ai logits, la energy loss vede OOD dove non lo è.

**Verifica Implementata**:
- **File**: `eomt/training/lightning_module.py`
- **Modifica**: Overlay check per primi 3 batch (linee 355-393)
- **Logica**: Crea overlay (ood_mask in rosso sopra immagine) e logga su wandb

**Check**: Durante primi 3 batch, vedi:
- Logging statistiche ood_mask (come prima)
- **NUOVO**: Overlay visualization su wandb (`check/ood_mask_overlay_batch*_sample*`)
- Verifica visivamente che regioni rosse (OOD) corrispondano a pixel incollati

**Nota**: `ood_mask` viene creato DOPO tutte le trasformazioni (linea 123 in `transforms.py`), quindi è già allineato all'immagine finale. L'overlay serve per verificare visivamente.

---

## 📊 Nuove Metriche Loggate

### Task 2 (già presente):
- `dbg/energy_id_mean`
- `dbg/energy_ood_mean`
- `dbg/energy_sep`

### Task 4 (NUOVO):
- `dbg/energy_id_std` (std di E_id)
- `dbg/energy_ood_std` (std di E_ood)
- Statistiche complete loggate quando margini vengono calcolati (P10/P50/P90)

### Punto #1 (NUOVO):
- `check/ood_mask_overlay_batch*_sample*` (wandb images)

---

## 🔍 Check da Fare Durante Training

### Check Punto #1 (Allineamento):
- Guarda overlay su wandb: regioni rosse devono corrispondere a oggetti incollati
- Se non allineato → problema critico, fix necessario

### Check Punto #2 (LogitNorm):
- LogitNorm deve essere DISABILITATA quando `ood_mask` è presente
- Verifica nei log: non vedi errori/conflict

### Check Punto #3 (Margini):
1. Dopo ~5000 step: verifica log "Task 4 - Energy Statistics"
2. Verifica che `m_out > m_in` (altrimenti warning)
3. Verifica che margini siano ragionevoli (non troppo lontani)
4. Monitora `dbg/energy_sep`: deve essere positivo e crescere

---

## ⚠️ Note Importanti

1. **Punto #1**: `ood_mask` viene creato DOPO le trasformazioni, quindi è già allineato. L'overlay serve per verifica visiva.

2. **Punto #2**: LogitNorm è ora disabilitata automaticamente quando OOD è presente. Se abiliti LogitNorm manualmente nel config, verrà ignorata per semantic+OE.

3. **Punto #3**: Margini vengono calcolati automaticamente dopo 5k step. Se vuoi cambiare il numero di step, modifica `_energy_margin_calculation_steps` in `LightningModule.__init__`.

4. **Memoria**: Per evitare memory explosion, Task 4 campiona max 1000 valori per batch durante accumulazione.

---

## 📝 File Modificati

1. `eomt/training/mask_classification_loss.py`: Punto #2 (LogitNorm)
2. `eomt/training/energy_ood_loss.py`: Punto #3 (update_margins method)
3. `eomt/training/lightning_module.py`: Punti #1 (overlay) + #3 (calcolo margini)

---

## ✅ Status

- ✅ Punto Critico #2: COMPLETATO
- ✅ Punto Critico #3: COMPLETATO (Task 4 implementato)
- ✅ Punto Critico #1: COMPLETATO (overlay check implementato)

Tutte le correzioni critiche sono state implementate e sono pronte per il testing!
