# 🚀 Guida Rapida: Avviare il Training con COCO

## ✅ File COCO Scaricati

Hai i file in:
```
/content/drive/MyDrive/coco2017_zips/
├── annotations_trainval2017.zip
├── train2017.zip
└── val2017.zip
```

**Nota**: Se il file si chiama `val20217.zip` invece di `val2017.zip`, rinominalo:
```bash
cd /content/drive/MyDrive/coco2017_zips
mv val20217.zip val2017.zip
```

## 📝 Step 1: Verifica i File

Verifica che i file esistano:
```bash
ls -lh /content/drive/MyDrive/coco2017_zips/
```

Dovresti vedere:
- `annotations_trainval2017.zip` (~241MB)
- `train2017.zip` (~18GB)
- `val2017.zip` (~1GB)

## 📝 Step 2: Prepara il Config

Ho creato un file di configurazione pronto: `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640_with_coco.yaml`

**Modifica queste righe nel file:**

1. **Path ai pesi pre-addestrati:**
```yaml
ckpt_path: "/path/to/eomt_cityscapes.bin"  # Sostituisci con il tuo path
```

2. **Path a Cityscapes:**
```yaml
path: "/path/to/cityscapes"  # Sostituisci con il path al dataset Cityscapes
```

3. **Path a COCO (già configurato):**
```yaml
coco_path: "/content/drive/MyDrive/coco2017_zips"  # ✅ Già corretto!
```

## 📝 Step 3: Avvia il Training

### Opzione A: Usa il config pronto

```bash
cd eomt
python main.py fit \
    --config configs/dinov2/cityscapes/semantic/eomt_base_640_with_coco.yaml \
    --model.ckpt_path /path/to/eomt_cityscapes.bin \
    --data.path /path/to/cityscapes
```

### Opzione B: Modifica il config e poi esegui

1. Apri `eomt/configs/dinov2/cityscapes/semantic/eomt_base_640_with_coco.yaml`
2. Modifica i path necessari
3. Esegui:
```bash
cd eomt
python main.py fit --config configs/dinov2/cityscapes/semantic/eomt_base_640_with_coco.yaml
```

## ⚙️ Parametri COCO Configurati

Il config è già ottimizzato con:
- ✅ `use_coco_zip: true` (perché hai i file zip)
- ✅ `coco_split: "val2017"` (usa val2017, più veloce)
- ✅ `paste_probability: 0.5` (50% probabilità di applicare cut-paste)
- ✅ `min_objects: 1, max_objects: 3` (1-3 oggetti per immagine)
- ✅ `min_scale: 0.1, max_scale: 0.3` (oggetti al 10-30% dell'immagine)

## 🔧 Personalizzazione (Opzionale)

Se vuoi modificare i parametri Outlier Exposure, modifica nel config:

```yaml
# Per più esposizione agli outlier (più aggressivo)
paste_probability: 0.7
min_objects: 2
max_objects: 4

# Per meno esposizione (più conservativo)
paste_probability: 0.3
min_objects: 1
max_objects: 2
```

## 📊 Cosa Aspettarsi

All'avvio del training, dovresti vedere:
```
COCOOutlierDataset: Loaded X valid objects from val2017
Outlier Exposure enabled with X COCO objects
```

Questo conferma che COCO è caricato correttamente!

## 🐛 Troubleshooting

### Errore: "COCO annotation file not found in zip"
- Verifica che `annotations_trainval2017.zip` contenga `annotations/instances_val2017.json`
- Estrai temporaneamente per verificare:
```bash
unzip -l /content/drive/MyDrive/coco2017_zips/annotations_trainval2017.zip | grep instances_val2017
```

### Errore: "Image not found in zip"
- Verifica che `val2017.zip` contenga la cartella `val2017/`
- Estrai temporaneamente per verificare:
```bash
unzip -l /content/drive/MyDrive/coco2017_zips/val2017.zip | head -20
```

### Errore: "No valid objects found"
- Riduci `coco_min_area` nel config (es. da 1000 a 500)
- Verifica che le annotazioni siano valide

### Errore: "pycocotools not available"
```bash
pip install pycocotools
```

## ✅ Checklist Pre-Training

- [ ] File COCO scaricati e nella directory corretta
- [ ] File rinominati correttamente (val2017.zip, non val20217.zip)
- [ ] Config modificato con path corretti:
  - [ ] `ckpt_path` ai pesi eomt_cityscapes.bin
  - [ ] `data.path` al dataset Cityscapes
  - [ ] `coco_path` già corretto ✅
- [ ] `pycocotools` installato: `pip install pycocotools`
- [ ] Pronto per avviare il training!

## 🎯 Comando Finale

```bash
cd eomt
python main.py fit \
    --config configs/dinov2/cityscapes/semantic/eomt_base_640_with_coco.yaml \
    --model.ckpt_path /path/to/eomt_cityscapes.bin \
    --data.path /path/to/cityscapes \
    --data.coco_path /content/drive/MyDrive/coco2017_zips
```

**Buon training! 🚀**
