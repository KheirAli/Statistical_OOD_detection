#!/bin/bash
# R6 overnight: DDAD UNet + DPS+inpainting (scale=5, box=128)
#   Phase 1: Sample 11 images × 20 seeds
#   Phase 2: Eval with our typical-set PMF scorer
#   Phase 3: Eval with rohan's local-Gaussian scorer
# Usage: screen -dmS ddad_dps bash run_ddad_dps_overnight.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs
LOG=logs/ddad_dps_overnight.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

log "=========================================="
log "R6: DDAD UNet + DPS+inpainting — $(date)"
log "  scale=5.0, box=128, num_seeds=20, skip=25"
log "=========================================="

# Phase 1 — sample
log ""
log "Phase 1: DPS+inpainting sampling (DDAD UNet)"
CUDA_VISIBLE_DEVICES=0 python -u tools/run_ddad_dps_inpainting.py \
    --num_seeds 20 --scale 5.0 --box 128 --skip 25 \
    2>&1 | tee -a "$LOG"
log "  Phase 1 exit=${PIPESTATUS[0]}"

# Phase 2 — our PMF scorer
log ""
log "Phase 2: typical_set scorer"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_inpaint.yaml \
    --skip_sampling --no_plots \
    --n_pca 5 --bins_pca 16 \
    --scorer typical_set \
    --output_dir ./results_eval_ddad_dps_inpaint_pmf \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 2 exit=${PIPESTATUS[0]}"

# Phase 3 — rohan local-Gaussian scorer (note: σ is a placeholder, may need tuning)
log ""
log "Phase 3: local_gaussian scorer (σ placeholder=1.0 — may need tuning)"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_inpaint.yaml \
    --skip_sampling --no_plots \
    --scorer local_gaussian \
    --output_dir ./results_eval_ddad_dps_inpaint_lg \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 3 exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Done — $(date)"
log "  recons:  ./results_patches_ddad_dps_inpaint/"
log "  eval pmf: ./results_eval_ddad_dps_inpaint_pmf/"
log "  eval lg:  ./results_eval_ddad_dps_inpaint_lg/"
log "=========================================="
