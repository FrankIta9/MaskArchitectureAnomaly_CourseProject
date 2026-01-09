# Guida Completa: Usare COCO con Outlier Exposure

## ✅ Implementazione Completa

Ho implementato completamente il caricamento di COCO per Outlier Exposure! Ora puoi usarlo facilmente.

## 📦 Installazione Dipendenze

```bash
pip install pycocotools
```

## 🚀 Come Usare

### Opzione 1: Usa il Dataset Cityscapes con Outlier Exposure

Ho creato `cityscapes_semantic_with_oe.py` che estende il dataset Cityscapes con supporto per COCO.

**Modifica il tuo config YAML:**

```yaml
data:
  class_path: datasets.cityscapes_semantic_with_oe.CityscapesSemanticWithOE
  init_args:
    path: "/path/to/cityscapes"
    # ... altri parametri ...
    
    # Parametri Outlier Exposure
    coco_path: "/path/to/coco"  # Path alla directory COCO
    coco_split: "val2017"        # "train2017" o "val2017"
    use_coco_zip: false         # true se COCO è in zip files
    paste_probability: 0.5       # Probabilità di applicare cut-paste
    min_objects: 1               # Min oggetti per immagine
    max_objects: 3               # Max oggetti per immagine
    min_scale: 0.1               # Scala minima oggetti
    max_scale: 0.3               # Scala massima oggetti
    coco_min_area: 1000          # Area minima oggetti COCO (pixel)
```

### Opzione 2: Usa Programmaticamente

```python
from datasets.outlier_exposure import COCOOutlierDataset, OutlierExposureTransform
from datasets.transforms import Transforms

# Carica COCO
coco_dataset = COCOOutlierDataset(
    coco_path="/path/to/coco",
    split="val2017",
    min_area=1000,
    use_zip=False,  # True se COCO è in zip
)

# Crea Outlier Exposure transform
outlier_transform = OutlierExposureTransform(
    outlier_dataset=coco_dataset,
    paste_probability=0.5,
    min_objects=1,
    max_objects=3,
    min_scale=0.1,
    max_scale=0.3,
)

# Aggiungi alle trasformazioni
transforms = Transforms(
    img_size=(640, 640),
    color_jitter_enabled=True,
    scale_range=(0.5, 2.0),
    outlier_exposure_transform=outlier_transform,  # Aggiungi qui
)
```

## 📁 Struttura COCO Richiesta

### Se usi directory (use_zip=False):

```
/path/to/coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   ├── 000000000009.jpg
│   └── ...
└── val2017/
    ├── 000000000139.jpg
    └── ...
```

### Se usi zip files (use_zip=True):

```
/path/to/coco/
├── annotations_trainval2017.zip
│   └── annotations/
│       ├── instances_train2017.json
│       └── instances_val2017.json
├── train2017.zip
│   └── train2017/
│       └── ...
└── val2017.zip
    └── val2017/
        └── ...
```

## 🎯 Esempio Completo di Config

Crea un file `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640_with_oe.yaml`:

```yaml
trainer:
  max_epochs: 107
  logger:
    class_path: lightning.pytorch.loggers.wandb.WandbLogger
    init_args:
      resume: allow
      project: "eomt"
      name: "cityscapes_semantic_eomt_base_640_with_oe"
model:
  class_path: training.mask_classification_semantic.MaskClassificationSemantic
  init_args:
    ckpt_path: "/path/to/eomt_cityscapes.bin"
    attn_mask_annealing_enabled: True
    attn_mask_annealing_start_steps: [3317, 8292, 13268]
    attn_mask_annealing_end_steps: [6634, 11609, 16585]
    network:
      class_path: models.eomt.EoMT
      init_args:
        num_q: 100
        num_blocks: 3
        encoder:
          class_path: models.vit.ViT
          init_args:
            backbone_name: vit_base_patch14_reg4_dinov2
data:
  class_path: datasets.cityscapes_semantic_with_oe.CityscapesSemanticWithOE
  init_args:
    path: "/path/to/cityscapes"
    batch_size: 16
    num_workers: 4
    img_size: [640, 640]
    num_classes: 19
    color_jitter_enabled: true
    scale_range: [0.5, 2.0]
    
    # Outlier Exposure
    coco_path: "/path/to/coco"
    coco_split: "val2017"
    use_coco_zip: false
    paste_probability: 0.5
    min_objects: 1
    max_objects: 3
    min_scale: 0.1
    max_scale: 0.3
    coco_min_area: 1000
```

Poi esegui:

```bash
cd eomt
python main.py fit --config configs/dinov2/cityscapes/semantic/eomt_base_640_with_oe.yaml
```

## 🔧 Parametri Consigliati

### Per iniziare (conservativo):
```yaml
paste_probability: 0.3
min_objects: 1
max_objects: 2
min_scale: 0.1
max_scale: 0.2
coco_min_area: 2000
```

### Per massimizzare AuPRC (aggressivo):
```yaml
paste_probability: 0.7
min_objects: 2
max_objects: 4
min_scale: 0.15
max_scale: 0.4
coco_min_area: 1000
```

### Bilanciato (consigliato):
```yaml
paste_probability: 0.5
min_objects: 1
max_objects: 3
min_scale: 0.1
max_scale: 0.3
coco_min_area: 1000
```

## 🐛 Troubleshooting

### Errore: "pycocotools not available"
```bash
pip install pycocotools
```

### Errore: "COCO annotation file not found"
- Verifica che il path a `coco_path` sia corretto
- Verifica che esista `coco_path/annotations/instances_val2017.json`

### Errore: "No valid objects found"
- Riduci `coco_min_area` (es. da 1000 a 500)
- Verifica che COCO abbia annotazioni valide

### Errore: "Image not found in zip"
- Verifica che `use_coco_zip` sia corretto (true se usi zip, false se usi directory)
- Verifica che i file zip contengano le immagini nella struttura corretta

## 📊 Risultati Attesi

### Solo EIM Loss (senza COCO):
- AuPRC: +5-10 punti
- FPR95: -5-10 punti

### EIM Loss + Outlier Exposure (con COCO):
- AuPRC: +7-15 punti
- FPR95: -7-15 punti

## ✅ Checklist

- [ ] Installato `pycocotools`: `pip install pycocotools`
- [ ] Scaricato COCO dataset (val2017 minimo)
- [ ] Verificata struttura directory/zip COCO
- [ ] Modificato config YAML con parametri COCO
- [ ] Usato `CityscapesSemanticWithOE` come data class
- [ ] Avviato training

## 🎯 Quick Test

Per testare se COCO è caricato correttamente:

```python
from datasets.outlier_exposure import COCOOutlierDataset

# Test caricamento
coco = COCOOutlierDataset(
    coco_path="/path/to/coco",
    split="val2017",
    min_area=1000,
)

print(f"Caricati {len(coco)} oggetti COCO")

# Test caricamento oggetto
img, mask = coco[0]
print(f"Immagine shape: {img.shape}")
print(f"Maschera shape: {mask.shape}")
print(f"Maschera dtype: {mask.dtype}")
```

Se funziona, vedrai:
```
COCOOutlierDataset: Loaded X valid objects from val2017
Caricati X oggetti COCO
Immagine shape: torch.Size([3, H, W])
Maschera shape: torch.Size([H, W])
Maschera dtype: torch.bool
```

## 🚀 Pronto!

Ora puoi usare COCO Outlier Exposure completamente integrato nel tuo training! 🎉
