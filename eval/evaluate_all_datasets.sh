#!/bin/bash
# ============================================================================
# Evaluation script per tutti i 5 dataset richiesti dal progetto
# ============================================================================

# CONFIGURAZIONE - Modifica questi path
CHECKPOINT_DIR="${1:-/content/drive/MyDrive/eomt_checkpoints_warmup/epoch7}"
CHECKPOINT_FILE="${2:-epoch7.ckpt}"
OUTPUT_DIR="${3:-results_epoch7}"
TEMPERATURE="${4:-1.0}"
METHOD="${5:-msp}"

echo "============================================"
echo "EoMT Anomaly Segmentation Evaluation"
echo "============================================"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Checkpoint file: $CHECKPOINT_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Temperature: $TEMPERATURE"
echo "Method: $METHOD"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Dataset paths (modifica questi per il tuo ambiente)
ROADANOMALY21="/content/drive/MyDrive/datasets/RoadAnomaly21"
ROADOBSTICLE21="/content/drive/MyDrive/datasets/RoadObsticle21"
FS_STATIC="/content/drive/MyDrive/datasets/fs_static"
ROADANOMALY="/content/drive/MyDrive/datasets/RoadAnomaly"
FS_LOSTFOUND="/content/drive/MyDrive/datasets/FS_LostFound_full"

# ============================================================================
# 1. RoadAnomaly21
# ============================================================================
echo ""
echo "📊 Evaluating RoadAnomaly21..."
python eval_eomt_anomaly.py \
  --input "$ROADANOMALY21/images/*.jpg" \
  --loadDir "$CHECKPOINT_DIR" \
  --loadWeights "$CHECKPOINT_FILE" \
  --method "$METHOD" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadanomaly21_${METHOD}.txt"

# ============================================================================
# 2. RoadObsticle21
# ============================================================================
echo ""
echo "📊 Evaluating RoadObsticle21..."
python eval_eomt_anomaly.py \
  --input "$ROADOBSTICLE21/images/*.webp" \
  --loadDir "$CHECKPOINT_DIR" \
  --loadWeights "$CHECKPOINT_FILE" \
  --method "$METHOD" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadobsticle21_${METHOD}.txt"

# ============================================================================
# 3. fs_static
# ============================================================================
echo ""
echo "📊 Evaluating fs_static..."
python eval_eomt_anomaly.py \
  --input "$FS_STATIC/images/*.jpg" \
  --loadDir "$CHECKPOINT_DIR" \
  --loadWeights "$CHECKPOINT_FILE" \
  --method "$METHOD" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/fs_static_${METHOD}.txt"

# ============================================================================
# 4. RoadAnomaly (original)
# ============================================================================
echo ""
echo "📊 Evaluating RoadAnomaly..."
python eval_eomt_anomaly.py \
  --input "$ROADANOMALY/images/*.jpg" \
  --loadDir "$CHECKPOINT_DIR" \
  --loadWeights "$CHECKPOINT_FILE" \
  --method "$METHOD" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadanomaly_${METHOD}.txt"

# ============================================================================
# 5. FS_LostFound_full
# ============================================================================
echo ""
echo "📊 Evaluating FS_LostFound_full..."
python eval_eomt_anomaly.py \
  --input "$FS_LOSTFOUND/images/*.png" \
  --loadDir "$CHECKPOINT_DIR" \
  --loadWeights "$CHECKPOINT_FILE" \
  --method "$METHOD" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/fs_lostfound_${METHOD}.txt"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================"
echo "✅ EVALUATION COMPLETE"
echo "============================================"
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Summary:"
cat "$OUTPUT_DIR"/*_${METHOD}.txt

echo ""
echo "📈 Per vedere i risultati dettagliati:"
echo "   cat $OUTPUT_DIR/*_${METHOD}.txt"
