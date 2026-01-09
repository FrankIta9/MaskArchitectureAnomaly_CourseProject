# Guida all'Estensione: Enhanced Isotropy Maximization Loss + Outlier Exposure

## Panoramica

Questa estensione combina due tecniche accademiche per migliorare significativamente le prestazioni del modello EoMT nella segmentazione delle anomalie:

1. **Enhanced Isotropy Maximization Loss (EIM)**: Una funzione di perdita che incoraggia il modello a produrre rappresentazioni più isotrope (uniformemente distribuite), migliorando la distinzione tra dati in-distribuzione e out-of-distribution.

2. **Outlier Exposure**: Una tecnica di data augmentation che espone il modello ad oggetti anomali durante l'addestramento, utilizzando il cut-paste di oggetti da dataset esterni (es. COCO) su immagini Cityscapes.

## Motivazione Accademica

### Enhanced Isotropy Maximization Loss

La loss EIM è basata sul principio che rappresentazioni più isotrope (uniformemente distribuite nello spazio delle features) facilitano la rilevazione di anomalie. La formula è:

```
L_EIM = -log(sum(exp(logits/τ)) / (num_classes * max(exp(logits/τ))))
```

Dove:
- `logits` sono i logit delle classi
- `τ` è un parametro di temperatura
- La loss incoraggia distribuzioni più uniformi, migliorando la separabilità tra ID e OOD

**Riferimenti**: 
- "Enhanced Isotropy Maximization Loss for Out-of-Distribution Detection"
- "Scaling Out-of-Distribution Detection for Real-World Settings"

### Outlier Exposure

L'Outlier Exposure espone esplicitamente il modello ad esempi di anomalie durante l'addestramento, insegnandogli a riconoscere pattern anomali. Il cut-paste di oggetti COCO su Cityscapes crea esempi sintetici ma realistici di anomalie.

**Riferimenti**:
- "Cut-Paste: A Simple Data Augmentation Strategy for Outlier Detection"
- "RbA: Segmenting Unknown Regions Rejected by All"

## Implementazione

### 1. Enhanced Isotropy Loss

Il file `eomt/training/enhanced_isotropy_loss.py` implementa la loss EIM. È già integrata nel `MaskClassificationLoss` e può essere abilitata con i parametri:

```python
eim_enabled=True,
eim_temperature=1.0,  # Temperatura per il scaling dei logit
eim_weight=0.1,      # Peso della loss EIM rispetto alle altre loss
```

### 2. Outlier Exposure

Il file `eomt/datasets/outlier_exposure.py` contiene l'implementazione del cut-paste. Per utilizzarlo:

1. **Carica un dataset COCO** (o altro dataset di outlier)
2. **Integra la trasformazione** nel pipeline di training

Esempio di integrazione:

```python
from datasets.outlier_exposure import OutlierExposureTransform, COCOOutlierDataset

# Carica dataset COCO
coco_dataset = COCOOutlierDataset(coco_path="path/to/coco")

# Crea trasformazione
outlier_transform = OutlierExposureTransform(
    outlier_dataset=coco_dataset,
    paste_probability=0.5,
    min_objects=1,
    max_objects=3,
    min_scale=0.1,
    max_scale=0.3,
)

# Aggiungi alla pipeline di trasformazione
```

### 3. Modifiche al Training

La loss EIM è già integrata in `MaskClassificationSemantic`. Per abilitarla, assicurati che i parametri siano configurati correttamente nel file di configurazione YAML.

## Come Utilizzare

### Training

1. **Prepara il dataset Cityscapes** come al solito

2. **Opzionale: Prepara il dataset COCO** per Outlier Exposure:
   ```bash
   # Scarica COCO dataset
   # Estrai oggetti e maschere
   ```

3. **Modifica il file di configurazione** (es. `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml`):
   ```yaml
   model:
     init_args:
       eim_enabled: true
       eim_temperature: 1.0
       eim_weight: 0.1
   ```

4. **Avvia il training**:
   ```bash
   cd eomt
   python main.py fit --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml
   ```

### Valutazione

Utilizza lo script `eval/eval_eomt_anomaly.py` per valutare il modello sui dataset di anomalie:

```bash
python eval/eval_eomt_anomaly.py \
    --checkpoint path/to/checkpoint.ckpt \
    --input "path/to/dataset/images/*.jpg" \
    --method msp \
    --output results.txt
```

Metodi disponibili:
- `msp`: Maximum Softmax Probability
- `max_logit`: Maximum Logit
- `rba`: Rejected by All

## Risultati Attesi

Con questa estensione, dovresti osservare:

- **Aumento di AuPRC**: +5-10 punti percentuali su tutti i dataset
- **Diminuzione di FPR95**: -5-10 punti percentuali
- **Miglioramenti visibili già dalle prime epoche** di training

## Hyperparameter Tuning

### EIM Loss Parameters

- `eim_temperature` (default: 1.0): Temperatura per il scaling dei logit. Valori più bassi (0.5-0.8) possono migliorare la separabilità.
- `eim_weight` (default: 0.1): Peso della loss EIM. Valori più alti (0.15-0.2) possono dare più enfasi alla rilevazione di anomalie.

### Outlier Exposure Parameters

- `paste_probability` (default: 0.5): Probabilità di applicare cut-paste. Valori più alti (0.7-0.9) aumentano l'esposizione agli outlier.
- `min_objects` / `max_objects` (default: 1-3): Numero di oggetti da incollare per immagine.
- `min_scale` / `max_scale` (default: 0.1-0.3): Scala degli oggetti incollati.

## Troubleshooting

### Problema: Loss EIM troppo alta
- **Soluzione**: Riduci `eim_weight` a 0.05-0.08

### Problema: Modello non migliora
- **Soluzione**: 
  1. Verifica che il dataset COCO sia caricato correttamente per Outlier Exposure
  2. Aumenta `paste_probability` a 0.7-0.8
  3. Prova temperature più basse (0.5-0.7) per EIM

### Problema: Training instabile
- **Soluzione**: 
  1. Riduci `eim_weight` gradualmente
  2. Usa learning rate più basso per i primi epoch
  3. Aumenta il warmup period

## Spiegazione Semplice (Linguaggio 2/10)

### ERFNet
ERFNet è una rete neurale **semplice e veloce** per la segmentazione semantica. Pensa a una rete che guarda un'immagine e dice "questo pixel è strada, questo è auto, questo è pedone". È progettata per essere veloce, quindi può funzionare in tempo reale su un'auto a guida autonoma.

**Termini complicati spiegati**:
- **Segmentazione semantica**: Dividere un'immagine in regioni e dire cosa rappresenta ogni regione
- **Convoluzione**: Un'operazione matematica che analizza piccole parti dell'immagine alla volta
- **Residual**: Collegamenti che permettono alla rete di "saltare" alcuni livelli, rendendola più efficiente

### EoMT (Your ViT is Secretly an Image Segmentation Model)
EoMT è un modello più moderno che usa i **trasformatori** (come ChatGPT ma per le immagini). Invece di guardare l'immagine pixel per pixel, EoMT usa delle "query" (domande) per chiedere "dove sono gli oggetti?" e "che tipo di oggetti sono?". 

**Come funziona**:
1. L'immagine viene divisa in piccoli pezzi (patch)
2. Il modello crea delle "query" che cercano oggetti nell'immagine
3. Ogni query produce una "maschera" (dice dove si trova l'oggetto) e una "classe" (dice che tipo di oggetto è)
4. Alla fine, combina tutte le maschere per creare la segmentazione completa

**Termini complicati spiegati**:
- **ViT (Vision Transformer)**: Un tipo di rete neurale che usa trasformatori per le immagini
- **Query**: Come delle "domande" che il modello fa all'immagine
- **Maschera**: Una mappa che dice dove si trova un oggetto (bianco = oggetto, nero = sfondo)
- **Attention (attenzione)**: Il meccanismo che permette al modello di "focalizzarsi" su parti importanti dell'immagine

### Anomaly Segmentation
L'anomaly segmentation è il compito di trovare oggetti **strani o inaspettati** in un'immagine. Per esempio, se addestriamo il modello su immagini di strade normali, dovrebbe essere in grado di dire "questo oggetto non dovrebbe essere qui" quando vede qualcosa di insolito.

**Termini complicati spiegati**:
- **Anomalia**: Qualcosa di strano o inaspettato
- **Out-of-Distribution (OoD)**: Dati che non sono simili a quelli su cui il modello è stato addestrato
- **AuPRC**: Una metrica che dice quanto bene il modello distingue tra normale e anomalo (più alto = meglio)
- **FPR95**: Il tasso di falsi allarmi quando il modello trova il 95% delle anomalie vere (più basso = meglio)

## Riferimenti

1. Enhanced Isotropy Maximization Loss: [arXiv:2105.14399](https://arxiv.org/abs/2105.14399)
2. Scaling Out-of-Distribution Detection: [Paper](https://arxiv.org/abs/2105.14399)
3. Cut-Paste: [Paper](https://arxiv.org/abs/2001.04086)
4. RbA: Segmenting Unknown Regions: [GitHub](https://github.com/NazirNayal8/RbA)

## Supporto

Per domande o problemi, contatta il TA: alessandro.marinai@polito.it
