#!/bin/bash
# X-ray pilot: generate recons + score with 4 combinations, on the 500-image
# positive test set produced by tools/prepare_xray_dataset.py.
#
# Prereqs:
#   1. tools/prepare_xray_dataset.py has been run → /data/.../xray_mvtec/ exists
#   2. DDAD UNet trained → /data/.../ddad_xray_ckpts/MVTec/xray/<epoch>
#   3. configs/experiment_ddad_{native,dps}_xray.yaml paths point at the right dirs
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> bash scripts/run_xray_pilot.sh <checkpoint_path>
#
# Example:
#   bash scripts/run_xray_pilot.sh /data/akheirandish3/ddad_xray_ckpts/MVTec/xray/500

set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CKPT="${1:-/data/akheirandish3/ddad_xray_ckpts/MVTec/xray/500}"
IMAGE_DIR="${XRAY_IMAGE_DIR:-/data/akheirandish3/xray_mvtec/test/prohibited}"

mkdir -p logs
LOG="$(pwd)/logs/run_xray_pilot.log"
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "X-ray pilot — $(date)"
log "  ckpt:      $CKPT"
log "  image_dir: $IMAGE_DIR"
log "  GPU:       $CUDA_VISIBLE_DEVICES"
log "=========================================="

# ── Discover sample IDs from the test set ──
if [ ! -d "$IMAGE_DIR" ]; then
    log "ERROR: image dir not found: $IMAGE_DIR"
    exit 1
fi
# Extract stems → "samples_<stem>"
SAMPLE_NAMES=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | xargs -n1 basename | sed 's/\.png$//' | sed 's/^/samples_/' | tr '\n' ' ')
SAMPLE_IDS=$(ls "$IMAGE_DIR"/*.png 2>/dev/null | xargs -n1 basename | sed 's/\.png$//' | tr '\n' ' ')
N=$(echo "$SAMPLE_NAMES" | wc -w)
log "  discovered $N test samples"
if [ "$N" = "0" ]; then
    log "ERROR: no test images found under $IMAGE_DIR"
    exit 1
fi

# ── Ensure per-sample SP masks exist ──
log ""
log "Generating per-sample superpixel masks (one-time)"
for sid in $SAMPLE_IDS; do
    sp_dir="figures_xray/samples_${sid}"
    if [ ! -f "$sp_dir/mask.png" ]; then
        mkdir -p "$sp_dir"
        python -u super_pixel_generation.py \
            --input_image="$IMAGE_DIR/${sid}.png" \
            --output_dir="$sp_dir" 2>&1 | tee -a "$LOG" > /dev/null
    fi
done
log "  SP masks ready"

# ── Phase 1: DDAD-native recons ──
log ""
log "Phase 1: DDAD-native recons (N=20 seeds)"
python -u tools/run_ddad_reconstruction.py \
    --ckpt "$CKPT" \
    --image_dir "$IMAGE_DIR" \
    --samples $SAMPLE_IDS \
    --num_seeds 20 \
    --out_root ./results_patches_xray_native 2>&1 | tee -a "$LOG"
log "  Phase 1 exit=${PIPESTATUS[0]}"

# ── Phase 2: additive-noise DPS recons ──
log ""
log "Phase 2: additive-noise DPS recons (σ=0.1, scale=0.5, N=20 seeds)"
python -u tools/run_ddad_dps_sampling.py \
    --ckpt "$CKPT" \
    --image_dir "$IMAGE_DIR" \
    --samples $SAMPLE_IDS \
    --sigma 0.1 --scale 0.5 --num_seeds 20 --skip 25 \
    --out_root ./results_patches_xray_dps 2>&1 | tee -a "$LOG"
log "  Phase 2 exit=${PIPESTATUS[0]}"

# ── Phase 3: score DDAD-native with typical_set (with + without PCA) ──
log ""
log "Phase 3: typical_set scorers on DDAD-native recons"
python -u evaluate.py \
    --config configs/experiment_ddad_native_xray.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_xray_native_pmf \
    --sample_names $SAMPLE_NAMES 2>&1 | tee -a "$LOG"
log "  Phase 3a (PMF+PCA) exit=${PIPESTATUS[0]}"

python -u evaluate.py \
    --config configs/experiment_ddad_native_xray.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_xray_native_pmf_nopca \
    --sample_names $SAMPLE_NAMES 2>&1 | tee -a "$LOG"
log "  Phase 3b (PMF no-PCA) exit=${PIPESTATUS[0]}"

# ── Phase 4: score DPS with typical_set ──
log ""
log "Phase 4: typical_set scorers on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_xray.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_xray_dps_pmf \
    --sample_names $SAMPLE_NAMES 2>&1 | tee -a "$LOG"
log "  Phase 4a (PMF+PCA) exit=${PIPESTATUS[0]}"

python -u evaluate.py \
    --config configs/experiment_ddad_dps_xray.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_xray_dps_pmf_nopca \
    --sample_names $SAMPLE_NAMES 2>&1 | tee -a "$LOG"
log "  Phase 4b (PMF no-PCA) exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Done — $(date)"
log "Results under ./results_eval_xray_*/sweep_*.json"
log "Decision gate: mean Px AUC σ=5 ≥ 0.65 → proceed to full scale"
log "=========================================="

# ── Auto-summary of the 4 cells ──
log ""
log "── SUMMARY ──"
for dir in results_eval_xray_native_pmf results_eval_xray_native_pmf_nopca \
           results_eval_xray_dps_pmf results_eval_xray_dps_pmf_nopca; do
    json=$(ls ${dir}/sweep_*.json 2>/dev/null | head -1)
    if [ -n "$json" ]; then
        python3 -c "
import json
d = json.load(open('$json'))
a = d['averaged']
print(f\"  ${dir}:  SP(raw)=%.4f Px(raw)=%.4f Px(σ=5)=%.4f\" %
      (a['raw']['sp_roc_auc'], a['raw']['px_roc_auc'], a['sigma_5.0']['px_roc_auc']))
" | tee -a "$LOG"
    else
        log "  ${dir}:  no JSON"
    fi
done
