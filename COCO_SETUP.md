# Setup COCO per Outlier Exposure (Opzionale)

## ⚠️ IMPORTANTE: COCO è Opzionale!

**Puoi iniziare il training SUBITO senza COCO!** La EIM Loss da sola dovrebbe già dare ottimi risultati.

COCO è utile per miglioramenti aggiuntivi, ma non è necessario per iniziare.

## 📥 Come Scaricare COCO

### Step 1: Vai al Sito COCO
https://cocodataset.org/#download

### Step 2: Scarica i File Necessari

**Minimo necessario** (per iniziare velocemente):
- ✅ `2017 Val images [5K/1GB]` - Immagini di validazione
- ✅ `2017 Train/Val annotations [241MB]` - Annotazioni (maschere)

**Opzionale** (per più dati):
- `2017 Train images [118K/18GB]` - Immagini di training (più oggetti)

### Step 3: Estrai i File

Struttura consigliata:
```
/path/to/coco/
├── train2017/          # (opzionale)
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017/            # (necessario)
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
└── annotations/        # (necessario)
    ├── instances_train2017.json
    └── instances_val2017.json
```

## 🔧 Installazione Dipendenze

```bash
pip install pycocotools
```

## 💻 Implementazione Completa

Ecco un'implementazione completa per caricare oggetti COCO:

```python
# Aggiungi questo a eomt/datasets/outlier_exposure.py

from pycocotools.coco import COCO
from PIL import Image
import torch
import numpy as np
import random
from torchvision.transforms import ToTensor

class COCOOutlierDataset:
    def __init__(self, coco_path: str, split: str = "val2017", min_area: int = 1000):
        """
        Carica oggetti da COCO per Outlier Exposure.
        
        Args:
            coco_path: Path alla cartella COCO (es. "/path/to/coco")
            split: "train2017" o "val2017"
            min_area: Area minima dell'oggetto in pixel (filtra oggetti troppo piccoli)
        """
        self.coco_path = coco_path
        self.split = split
        ann_file = f"{coco_path}/annotations/instances_{split}.json"
        
        self.coco = COCO(ann_file)
        self.min_area = min_area
        
        # Pre-carica tutti gli oggetti validi
        self.valid_objects = []
        img_ids = self.coco.getImgIds()
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            for ann in anns:
                # Filtra oggetti troppo piccoli o con area invalida
                if ann['area'] >= min_area and not ann.get('iscrowd', 0):
                    self.valid_objects.append({
                        'img_id': img_id,
                        'ann_id': ann['id'],
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],
                        'area': ann['area']
                    })
        
        print(f"Loaded {len(self.valid_objects)} valid objects from COCO {split}")
    
    def __len__(self):
        return len(self.valid_objects)
    
    def __getitem__(self, idx):
        """
        Carica un oggetto COCO.
        
        Returns:
            img_tensor: Tensor dell'immagine (3, H, W)
            mask_tensor: Tensor della maschera (H, W) di tipo bool
        """
        obj_info = self.valid_objects[idx]
        img_id = obj_info['img_id']
        ann_id = obj_info['ann_id']
        
        # Carica immagine
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.coco_path}/{self.split}/{img_info['file_name']}"
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)
        
        # Carica maschera
        ann = self.coco.loadAnns(ann_id)[0]
        mask = self.coco.annToMask(ann)
        
        # Estrai bounding box
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = [int(v) for v in bbox]
        
        # Crop immagine e maschera
        img_crop = img_array[y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        
        # Converti in tensor
        if img_crop.size > 0 and mask_crop.size > 0:
            img_tensor = ToTensor()(Image.fromarray(img_crop))
            mask_tensor = torch.from_numpy(mask_crop).bool()
            return img_tensor, mask_tensor
        else:
            # Se il crop è vuoto, ritorna un oggetto casuale
            return self.__getitem__(random.randint(0, len(self.valid_objects) - 1))
```

## 🔗 Integrazione nel Training

Dopo aver implementato `COCOOutlierDataset`, modifica `eomt/datasets/outlier_exposure.py`:

```python
# Sostituisci il metodo _load_coco_object:

def _load_coco_object(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Carica un oggetto COCO."""
    if self.outlier_dataset is None:
        # Fallback a oggetto dummy
        dummy_img = torch.rand((3, 64, 64))
        dummy_mask = torch.ones((64, 64), dtype=torch.bool)
        return dummy_img, dummy_mask
    
    return self.outlier_dataset[idx]
```

E aggiorna `_get_random_outlier_object`:

```python
def _get_random_outlier_object(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ottiene un oggetto casuale dal dataset outlier."""
    if self.outlier_dataset is None or len(self.outlier_dataset) == 0:
        dummy_img = torch.rand((3, 64, 64))
        dummy_mask = torch.ones((64, 64), dtype=torch.bool)
        return dummy_img, dummy_mask
    
    idx = random.randint(0, len(self.outlier_dataset) - 1)
    return self.outlier_dataset[idx]
```

## 🎯 Uso nel Training

Per abilitare Outlier Exposure, devi passare il path COCO al dataset. Questo richiede modifiche al data module, che è più complesso.

**Alternativa più semplice**: Puoi aggiungere Outlier Exposure direttamente nelle trasformazioni del dataset Cityscapes.

## ⚡ Quick Test

Per testare se COCO è caricato correttamente:

```python
from datasets.outlier_exposure import COCOOutlierDataset

coco_dataset = COCOOutlierDataset(
    coco_path="/path/to/coco",
    split="val2017"
)

print(f"Numero di oggetti: {len(coco_dataset)}")

# Testa il caricamento
img, mask = coco_dataset[0]
print(f"Immagine shape: {img.shape}")
print(f"Maschera shape: {mask.shape}")
```

## 📊 Risultati Attesi con COCO

- **Senza COCO**: AuPRC +5-10 punti, FPR95 -5-10 punti
- **Con COCO**: AuPRC +7-15 punti, FPR95 -7-15 punti

## ❓ FAQ

**Q: Posso usare solo val2017?**
A: Sì, val2017 ha ~5000 immagini con molti oggetti, è sufficiente.

**Q: Quanto spazio serve?**
A: Val2017 + annotations = ~1.3GB. Train2017 = ~18GB.

**Q: Cosa succede se non implemento COCO?**
A: Nulla! Il training funziona normalmente. Solo la EIM Loss viene usata.

**Q: Posso usare un altro dataset invece di COCO?**
A: Sì! Basta implementare una classe simile a `COCOOutlierDataset` che ritorna (img, mask).

## 🎯 Conclusione

**Raccomandazione**: 
1. ✅ **Inizia SENZA COCO** - usa solo EIM Loss
2. ✅ **Valuta i risultati** - dovresti già vedere miglioramenti
3. ✅ **Aggiungi COCO dopo** - se vuoi spingere ancora di più

**Non aspettare COCO per iniziare!** 🚀
