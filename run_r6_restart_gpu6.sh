#!/bin/bash
# Restart failed Phases 2+3 on GPU 6 (GPU 0 is full from another user).
# Phase 4 may also need restarting if it OOM'd.
# Usage: screen -dmS r6_gpu6 bash run_r6_restart_gpu6.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6

mkdir -p logs
LOG=logs/r6_restart_gpu6.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

log "=========================================="
log "R6 restart on GPU 6 — $(date)"
log "=========================================="

# ── Phase 2 restart: typical_set (with PCA) on R6 recons ──
log ""
log "Phase 2 (restart): typical_set with PCA"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_dps_pmf \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 2 exit=${PIPESTATUS[0]}"

# ── Phase 3 restart: typical_set (no PCA) on R6 recons ──
log ""
log "Phase 3 (restart): typical_set no PCA"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_ddad_dps_pmf_nopca \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 3 exit=${PIPESTATUS[0]}"

# ── Phase 4 restart (if needed): local_gaussian ──
log ""
log "Phase 4 (restart if needed): local_gaussian"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer local_gaussian \
    --sigma_rohan 0.1 \
    --output_dir ./results_eval_ddad_dps_lg \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 4 exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Done — $(date)"
log "=========================================="
