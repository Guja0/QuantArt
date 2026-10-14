#!/usr/bin/env bash
# Encoder-only experiment: decoder + post_quant_conv frozen, encoder_real + quant_conv_real trained.
#   1. baseline   (--epochs 0: untouched weights, baseline val metrics + images)
#   2. training   (encoder only)
#   3. compare    (test reconstruction + 256 px style-transfer grids, original vs adapted encoder)
#   4. stylize    (512 px photo -> art for each style in STYLES, original vs adapted encoder)
#
# Usage (from QuantArt/):  CKPT=/path/to/coco2art/last.ckpt bash run_encoder_only.sh
# Any variable below can be overridden the same way, e.g. RUN=enc_lr5e-6 EXTRA="--lr 5e-6".
set -euo pipefail

PY=${PY:-python}
CONFIG=${CONFIG:-configs/coco2art.yaml}
CKPT=${CKPT:?set CKPT to the coco2art checkpoint (last.ckpt)}
OUT=${OUT:-outputs/finetune_encoder}
RUN=${RUN:-enc_60ep}
DEVICE=${DEVICE:-auto}
EXTRA=${EXTRA:-}                                   # extra flags for training, e.g. "--epochs 1 --max_steps 2"
CONTENT=${CONTENT:-test/style_image.HEIC}          # photo for step 4
STYLES=${STYLES:-"test/face_style_baseline.heic test/style_baseline.heic test/style_baseline_2.heic test/style_baseline_3.heic test/style_image_2.avif test/tree_style.HEIC"}

CFG=(--config "$CONFIG" --ckpt "$CKPT" --device "$DEVICE")

echo "== 1. baseline (no training)"
$PY -u finetune_artwork_decoder.py "${CFG[@]}" --train_part encoder --epochs 0 --out "$OUT" --run baseline
$PY -u compare_decoders.py "${CFG[@]}" --run "$OUT/baseline"

echo "== 2. train encoder only"
# shellcheck disable=SC2086
$PY -u finetune_artwork_decoder.py "${CFG[@]}" --train_part encoder --out "$OUT" --run "$RUN" $EXTRA

echo "== 3. compare original vs adapted encoder"
$PY -u compare_decoders.py "${CFG[@]}" --run "$OUT/$RUN"

echo "== 4. 512 px photo -> art"
if [ -f "$CONTENT" ]; then
  for style in $STYLES; do
    [ -f "$style" ] || { echo "skip missing style $style"; continue; }
    name=$(basename "${style%.*}")
    $PY -u stylize_pair.py "${CFG[@]}" --encoder "$OUT/$RUN/checkpoints/best_encoder.pt" \
      --content "$CONTENT" --style "$style" --out "outputs/style_transfer/${RUN}/${name}"
  done
else
  echo "skip: content photo $CONTENT not found"
fi
echo "done -> $OUT/$RUN/comparison, outputs/style_transfer/$RUN"
