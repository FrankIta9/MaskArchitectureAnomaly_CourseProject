# Quick Start: Usare i Pesi Pre-addestrati e COCO

## 🎯 Risposte Rapide

### 1. Dataset COCO - È Opzionale!

**Risposta breve**: Il dataset COCO è **opzionale** ma **consigliato** per ottenere i migliori risultati.

- ✅ **Puoi iniziare SUBITO** senza COCO: la **EIM Loss** da sola dovrebbe già dare miglioramenti significativi (+5-10 punti AuPRC)
- ✅ **COCO migliora ulteriormente**: Outlier Exposure con COCO può dare altri +2-5 punti
- ⚠️ **COCO richiede download manuale**: Il codice che ho scritto è solo la struttura, devi scaricare COCO tu stesso

### 2. Pesi Pre-addestrati `eomt_cityscapes.bin`

**Risposta breve**: Sì, puoi e DEVI usare questi pesi! Sono perfetti per partire.

## 📦 Come Usare i Pesi `eomt_cityscapes.bin`

### Opzione 1: Tramite Configurazione YAML

Modifica il file di configurazione (es. `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml`):

```yaml
model:
  class_path: training.mask_classification_semantic.MaskClassificationSemantic
  init_args:
    ckpt_path: "/path/to/eomt_cityscapes.bin"  # Aggiungi questa riga
    # ... resto della configurazione ...
```

Poi esegui:
```bash
cd eomt
python main.py fit --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml
```

### Opzione 2: Tramite Command Line

```bash
cd eomt
python main.py fit \
    --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --model.ckpt_path /path/to/eomt_cityscapes.bin
```

### Opzione 3: Fine-tuning Parziale (Solo Head)

Se vuoi fare fine-tuning solo della testa di classificazione (più veloce):

```yaml
model:
  class_path: training.mask_classification_semantic.MaskClassificationSemantic
  init_args:
    ckpt_path: "/path/to/eomt_cityscapes.bin"
    delta_weights: true  # Usa delta weights mode
    load_ckpt_class_head: false  # Non carica la testa di classificazione
    # ... resto della configurazione ...
```

## 🎨 Dataset COCO - Setup Opzionale

### Perché COCO è Utile?

COCO contiene oggetti vari (persone, animali, oggetti) che non sono presenti in Cityscapes. Incollando questi oggetti su immagini Cityscapes durante il training, il modello impara a riconoscere meglio le anomalie.

### Come Scaricare COCO

1. **Vai al sito COCO**: https://cocodataset.org/#download
2. **Scarica**:
   - `2017 Train images [118K/18GB]` (opzionale, per più dati)
   - `2017 Val images [5K/1GB]` (minimo necessario)
   - `2017 Train/Val annotations [241MB]` (necessario per le maschere)

3. **Estrai** i file in una cartella, esempio:
   ```
   /path/to/coco/
   ├── train2017/
   │   ├── 000000000009.jpg
   │   └── ...
   ├── val2017/
   │   ├── 000000000139.jpg
   │   └── ...
   └── annotations/
       ├── instances_train2017.json
       └── instances_val2017.json
   ```

### Come Integrare COCO nel Codice

Il file `eomt/datasets/outlier_exposure.py` ha una struttura base, ma devi implementare il caricamento COCO. Ecco un esempio semplificato:

```python
# Modifica eomt/datasets/outlier_exposure.py
# Sostituisci la classe COCOOutlierDataset con:

from pycocotools.coco import COCO
from PIL import Image
import torch

class COCOOutlierDataset:
    def __init__(self, coco_path: str, split: str = "val2017"):
        """
        Args:
            coco_path: Path alla cartella COCO (es. "/path/to/coco")
            split: "train2017" o "val2017"
        """
        self.coco_path = coco_path
        self.split = split
        ann_file = f"{coco_path}/annotations/instances_{split}.json"
        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Carica immagine
        img_path = f"{self.coco_path}/{self.split}/{img_info['file_name']}"
        img = Image.open(img_path).convert("RGB")
        
        # Carica maschere degli oggetti
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Prendi un oggetto casuale
        if len(anns) > 0:
            ann = anns[0]  # O scegli casualmente
            mask = self.coco.annToMask(ann)
            
            # Converti in tensor
            from torchvision.transforms import ToTensor
            img_tensor = ToTensor()(img)
            mask_tensor = torch.from_numpy(mask).bool()
            
            return img_tensor, mask_tensor
        else:
            # Se non ci sono annotazioni, ritorna None
            return None, None
```

**Nota**: Per usare `pycocotools`, installalo:
```bash
pip install pycocotools
```

### Integrare nel Training

Dopo aver implementato il caricamento COCO, modifica `eomt/datasets/transforms.py` per aggiungere Outlier Exposure:

```python
from datasets.outlier_exposure import OutlierExposureTransform, COCOOutlierDataset

# Nel metodo __init__ di Transforms:
def __init__(self, ..., coco_path=None):
    # ... codice esistente ...
    
    # Aggiungi Outlier Exposure se COCO è disponibile
    if coco_path:
        coco_dataset = COCOOutlierDataset(coco_path)
        self.outlier_transform = OutlierExposureTransform(
            outlier_dataset=coco_dataset,
            paste_probability=0.5,
        )
    else:
        self.outlier_transform = None

# Nel metodo forward:
def forward(self, img, target):
    # ... trasformazioni esistenti ...
    
    # Applica Outlier Exposure se disponibile
    if self.outlier_transform is not None:
        img, target = self.outlier_transform(img, target)
    
    return img, target
```

## 🚀 Strategia Consigliata

### Fase 1: Training Base con EIM Loss (SENZA COCO)
1. Usa i pesi `eomt_cityscapes.bin` come punto di partenza
2. Addestra con EIM Loss (già abilitata)
3. Valuta sui dataset di anomalie
4. **Dovresti già vedere miglioramenti significativi!**

### Fase 2: Aggiungi COCO (Opzionale, per risultati ancora migliori)
1. Scarica COCO
2. Implementa il caricamento (vedi sopra)
3. Ri-addestra o continua il training con Outlier Exposure
4. Valuta di nuovo

## 📊 Risultati Attesi

### Solo EIM Loss (Senza COCO):
- AuPRC: +5-10 punti percentuali
- FPR95: -5-10 punti percentuali

### EIM Loss + Outlier Exposure (Con COCO):
- AuPRC: +7-15 punti percentuali
- FPR95: -7-15 punti percentuali

## ⚡ Quick Start (Senza COCO)

```bash
# 1. Vai nella directory eomt
cd eomt

# 2. Modifica il config per aggiungere ckpt_path, oppure usa:
python main.py fit \
    --config configs/dinov2/cityscapes/semantic/eomt_base_640.yaml \
    --model.ckpt_path /path/to/eomt_cityscapes.bin

# 3. Dopo il training, valuta:
cd ..
python eval/eval_eomt_anomaly.py \
    --checkpoint eomt/lightning_logs/version_X/checkpoints/last.ckpt \
    --input "path/to/RoadAnomaly21/images/*.webp" \
    --method msp \
    --output results.txt
```

## ❓ FAQ

**Q: Devo per forza scaricare COCO?**
A: No! La EIM Loss da sola dovrebbe già dare ottimi risultati. COCO è un bonus.

**Q: I pesi `eomt_cityscapes.bin` sono compatibili?**
A: Sì, sono perfetti! Sono pesi pre-addestrati su Cityscapes, ideali come punto di partenza.

**Q: Quanto tempo ci vuole per scaricare COCO?**
A: Dipende dalla connessione. Il dataset val2017 è ~1GB, train2017 è ~18GB.

**Q: Posso usare solo val2017 per COCO?**
A: Sì, val2017 ha abbastanza oggetti per Outlier Exposure.

**Q: Cosa succede se non implemento COCO?**
A: Nulla! Il training funziona normalmente, solo senza Outlier Exposure. La EIM Loss funziona comunque.

## 🎯 Conclusione

**Inizia SUBITO con:**
1. ✅ Pesi `eomt_cityscapes.bin` (usa `--model.ckpt_path`)
2. ✅ EIM Loss (già abilitata)
3. ❌ COCO (opzionale, puoi aggiungerlo dopo)

**Buon training! 🚀**
