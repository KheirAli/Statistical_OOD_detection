#!/bin/bash
# R6 corrected: DDAD UNet + additive-noise DPS (y = x + 0.1*eps, identity forward, scale=0.5)
# + R3 no-PCA ablation on existing DDAD-native recons
# Usage: screen -dmS r6_corrected bash run_r6_corrected.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs
LOG=logs/r6_corrected.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

log "=========================================="
log "R6 corrected + ablations — $(date)"
log "=========================================="

# ── Phase 0: R3 no-PCA ablation (runs on existing DDAD-native recons) ──
log ""
log "Phase 0: R3 no-PCA (bins_pca=1) on DDAD-native recons"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set --bins_pca 1 \
    --output_dir ./results_eval_ddad_native_pmf_nopca \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 0 exit=${PIPESTATUS[0]}"

# ── Phase 1: DPS sampling (additive noise, identity forward, sigma=0.1, scale=0.5) ──
log ""
log "Phase 1: additive-noise DPS sampling (DDAD UNet, sigma=0.1, scale=0.5)"
CUDA_VISIBLE_DEVICES=0 python -u tools/run_ddad_dps_sampling.py \
    --samples 000 001 002 003 004 005 006 007 008 009 010 \
    --sigma 0.1 --scale 0.5 --num_seeds 20 --skip 25 \
    2>&1 | tee -a "$LOG"
log "  Phase 1 exit=${PIPESTATUS[0]}"

# ── Phase 2: typical_set scorer (with PCA) on R6 recons ──
log ""
log "Phase 2: typical_set (with PCA) on R6 recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_dps_pmf \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 2 exit=${PIPESTATUS[0]}"

# ── Phase 3: typical_set scorer (no PCA, bins_pca=1) on R6 recons ──
log ""
log "Phase 3: typical_set (no PCA) on R6 recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_ddad_dps_pmf_nopca \
    --sample_names $SAMPLES  2>&1 | tee -a "$LOG"
log "  Phase 3 exit=${PIPESTATUS[0]}"

# ── Phase 4: local_gaussian scorer on R6 recons (sigma=0.1, the true value) ──
log ""
log "Phase 4: local_gaussian on R6 recons (sigma=0.1, variance_explained=0.95)"
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
log "  R3 no-PCA:     ./results_eval_ddad_native_pmf_nopca/"
log "  R6 recons:     ./results_patches_ddad/"
log "  R6 PMF:        ./results_eval_ddad_dps_pmf/"
log "  R6 PMF noPCA:  ./results_eval_ddad_dps_pmf_nopca/"
log "  R6 local-G:    ./results_eval_ddad_dps_lg/"
log "=========================================="
