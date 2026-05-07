#!/bin/bash

CONFIG="config.yaml"
CATEGORY="xray"
LOAD_CHP="500"
CHECKPOINT_DIR="/data2/rohan/ckpts/DDAD/DvXray/MVTec"
DATA_PATH="/data/akheirandish3/mvtec_ad/xray/test"

SEEDS=(421 1231 4561 7891 13371 20241 991 71 3141 27181)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_LOG="results_xray_${TIMESTAMP}.txt"
AVG_FILE="results_xray_avg_${TIMESTAMP}.txt"

exec > >(tee -a "$FULL_LOG") 2>&1

echo "=========================================="
echo "  DDAD Detection — xray"
echo "  Checkpoint : ${CHECKPOINT_DIR}/${CATEGORY}/${LOAD_CHP}"
echo "  Data       : ${DATA_PATH}/scissors"
echo "  Seeds      : ${SEEDS[*]}"
echo "=========================================="

AUROC_VALS=()
SNR_VALS=()

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "-------- Seed ${SEED} --------"

    TMPOUT=$(mktemp)

    CUDA_VISIBLE_DEVICES=6 python main.py \
        --config         "$CONFIG" \
        --detection      True \
        --category       "$CATEGORY" \
        --load_chp       "$LOAD_CHP" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_path      "$DATA_PATH" \
        --seed           "$SEED" \
        2>&1 | tee "$TMPOUT"

    AUROC=$(grep "Per-image Pixel AUROC (mean):" "$TMPOUT" \
            | tail -1 | awk -F': ' '{print $2}' | tr -d '[:space:]')
    SNR=$(  grep "^SNR:"                          "$TMPOUT" \
            | tail -1 | awk '{print $2}'          | tr -d '[:space:]')
    rm -f "$TMPOUT"

    if [ -n "$AUROC" ]; then
        AUROC_VALS+=("$AUROC")
        SNR_VALS+=("$SNR")
        echo "  → Pixel AUROC=${AUROC}  SNR=${SNR}"
    else
        echo "  [WARN] Could not parse metrics for seed ${SEED}"
    fi
done

# ── Compute averages ──────────────────────────────────────────────────────────
python3 -c "
import sys, math

auroc = [float(x) for x in '${AUROC_VALS[*]}'.split()]
snr   = [float(x) for x in '${SNR_VALS[*]}'.split()]

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((x-m)**2 for x in vals) / (n-1)) if n > 1 else 0.0
    return m, s

am, as_ = stats(auroc)
sm, ss  = stats(snr)

print()
print('==========================================')
print(f'  Results over {len(auroc)} seeds')
print('==========================================')
print(f'  Pixel AUROC : {am:.2f} ± {as_:.2f}')
print(f'  SNR         : {sm:.4f} ± {ss:.4f}')
print(f'  Raw AUROC   : {auroc}')
print(f'  Raw SNR     : {snr}')
print('==========================================')
" | tee "$AVG_FILE"

echo ""
echo "Full log : $FULL_LOG"
echo "Averages : $AVG_FILE"