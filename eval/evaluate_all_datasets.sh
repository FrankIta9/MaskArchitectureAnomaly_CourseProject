#!/bin/bash
# ============================================================================
# Evaluation script per tutti i 5 dataset richiesti dal progetto
# ============================================================================

# CONFIGURAZIONE - Modifica questi path
CHECKPOINT="${1:-/content/drive/MyDrive/eomt_checkpoints_warmup/eomt_1024_oe_energy_warmup-epoch=007-metrics_val_iou_all=0.8190.ckpt}"
OUTPUT_DIR="${2:-results_epoch7}"
IMG_SIZE="${3:-1024}"
NUM_BLOCKS="${4:-3}"
TEMPERATURE="${5:-1.0}"
METHOD="${6:-msp}"

echo "============================================"
echo "EoMT Anomaly Segmentation Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output dir: $OUTPUT_DIR"
echo "Image size: ${IMG_SIZE}x${IMG_SIZE}"
echo "Num blocks: $NUM_BLOCKS"
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
  --checkpoint "$CHECKPOINT" \
  --input "$ROADANOMALY21/images/*.jpg" \
  --method "$METHOD" \
  --img_size "$IMG_SIZE" \
  --num_blocks "$NUM_BLOCKS" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadanomaly21_${METHOD}.txt"

# ============================================================================
# 2. RoadObsticle21
# ============================================================================
echo ""
echo "📊 Evaluating RoadObsticle21..."
python eval_eomt_anomaly.py \
  --checkpoint "$CHECKPOINT" \
  --input "$ROADOBSTICLE21/images/*.webp" \
  --method "$METHOD" \
  --img_size "$IMG_SIZE" \
  --num_blocks "$NUM_BLOCKS" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadobsticle21_${METHOD}.txt"

# ============================================================================
# 3. fs_static
# ============================================================================
echo ""
echo "📊 Evaluating fs_static..."
python eval_eomt_anomaly.py \
  --checkpoint "$CHECKPOINT" \
  --input "$FS_STATIC/images/*.jpg" \
  --method "$METHOD" \
  --img_size "$IMG_SIZE" \
  --num_blocks "$NUM_BLOCKS" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/fs_static_${METHOD}.txt"

# ============================================================================
# 4. RoadAnomaly (original)
# ============================================================================
echo ""
echo "📊 Evaluating RoadAnomaly..."
python eval_eomt_anomaly.py \
  --checkpoint "$CHECKPOINT" \
  --input "$ROADANOMALY/images/*.jpg" \
  --method "$METHOD" \
  --img_size "$IMG_SIZE" \
  --num_blocks "$NUM_BLOCKS" \
  --temperature "$TEMPERATURE" \
  --output "$OUTPUT_DIR/roadanomaly_${METHOD}.txt"

# ============================================================================
# 5. FS_LostFound_full
# ============================================================================
echo ""
echo "📊 Evaluating FS_LostFound_full..."
python eval_eomt_anomaly.py \
  --checkpoint "$CHECKPOINT" \
  --input "$FS_LOSTFOUND/images/*.png" \
  --method "$METHOD" \
  --img_size "$IMG_SIZE" \
  --num_blocks "$NUM_BLOCKS" \
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
