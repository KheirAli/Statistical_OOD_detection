#!/bin/bash
# R3 + R4: evaluate DDAD-native recons (already generated) with two scorers.
# Usage: screen -dmS three_way bash run_three_way.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs results_eval_ddad_native
LOG=logs/three_way.log
: > "$LOG"

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "R3 + R4 — started $(date)"
log "=========================================="

# ── R3: our typical-set PMF scorer on DDAD recons ──
log ""
log "R3: typical_set scorer — $(date)"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots \
    --n_pca 5 --bins_pca 16 \
    --scorer typical_set \
    --output_dir ./results_eval_ddad_native_pmf \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  R3 exit=${PIPESTATUS[0]}"

# ── R4: rohan local-Gaussian scorer on DDAD recons ──
log ""
log "R4: local_gaussian scorer — $(date)"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots \
    --scorer local_gaussian \
    --output_dir ./results_eval_ddad_native_lg \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  R4 exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Done — $(date)"
log "R3 (our PMF):       ./results_eval_ddad_native_pmf/"
log "R4 (rohan local-G): ./results_eval_ddad_native_lg/"
log "=========================================="
