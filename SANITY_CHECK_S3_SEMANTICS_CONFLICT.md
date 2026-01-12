# Sanity Check S3: Semantics Conflict Check (IMPORTANTISSIMO)

## Obiettivo
Verificare che nei pixel OOD la loss semantica non stia forzando classi Cityscapes a caso, creando conflitto con energy/logitnorm.

## Problema Potenziale
Se i pixel OOD hanno target semantic che li forza a classi Cityscapes (es. "road", "sidewalk"), mentre energy/logitnorm cercano di renderli "incerti/alti", si crea un **conflitto distruttivo**.

## Check da Fare

### 1. Verifica come sono gestiti i pixel OOD nella loss semantica

Ci sono due opzioni corrette:

**Opzione A**: Pixel OOD sono **ignore (255)** per la loss semantica
- La loss semantica ignora completamente i pixel OOD
- Energy/logitnorm possono gestire OOD senza conflitto

**Opzione B**: Pixel OOD sono gestiti come **"no-object/outlier"** coerentemente
- La loss semantica predice "no-object" (classe C+1) per OOD
- Energy/logitnorm si allineano con questo obiettivo

**❌ NON corretto**: Pixel OOD hanno target semantic di classi Cityscapes (es. "road")
- Questo crea conflitto: loss semantica forza "road", energy forza "incerto/alto"
- Risultato: tutto peggiora

### 2. Come verificare

In `mask_classification_loss.py`, nella funzione `forward`, verifica:

1. **Se ood_mask è presente, i pixel OOD vengono ignorati nella loss semantica?**
   - Dovrebbero essere filtrati prima di calcolare la loss
   
2. **Oppure i pixel OOD hanno target = "no-object" (classe C+1)?**
   - Dovrebbero essere gestiti coerentemente

### 3. Script di verifica

Crea uno script che:
1. Carica un batch con OOD
2. Estrae `ood_mask` e `target` semantic
3. Verifica che:
   - Pixel OOD hanno `target == 255` (ignore), OPPURE
   - Pixel OOD hanno `target == C` (no-object, dove C=num_classes)
4. NON devono avere `target` in `[0, C-1]` (classi Cityscapes normali)

## Fix Se Conflitto Trovato

### Se pixel OOD hanno target semantic di classi Cityscapes:

**Opzione 1**: Imposta target = 255 (ignore) per pixel OOD
```python
# In outlier_exposure.py, dopo aver creato ood_mask:
# Imposta target semantic a ignore (255) per pixel OOD
if "target" in target:  # Se c'è target semantic
    target["target"][ood_mask == 1] = 255  # Ignore per OOD
```

**Opzione 2**: Imposta target = num_classes (no-object) per pixel OOD
```python
# In outlier_exposure.py:
# Imposta target semantic a no-object per pixel OOD
if "target" in target:
    target["target"][ood_mask == 1] = num_classes  # no-object per OOD
```

**Nota**: Controlla come `mask_classification_loss.py` gestisce il target semantic. Potrebbe essere necessario modificare la loss per gestire correttamente OOD.

## Valutazione

✅ **PASS**: Se pixel OOD hanno target = 255 (ignore) OPPURE target = C (no-object)
❌ **FAIL**: Se pixel OOD hanno target in [0, C-1] (classi Cityscapes normali)

**Se FAIL**: Deve essere corretto PRIMA di fare training, altrimenti energy/logitnorm peggiorano sempre.
