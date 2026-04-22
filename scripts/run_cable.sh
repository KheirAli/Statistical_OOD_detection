#!/bin/bash
# Full cable experiment: DDAD native recons + all scorer combinations.
#
# Reproduces rows R3, R3np, R4, R6a, R6a-np, R6b of RESULTS.md on
# samples_000..samples_010 (test/combined, 11 images).
#
# Phases:
#   0. Generate DDAD native recons (tools/run_ddad_reconstruction.py)
#   1. Score DDAD native recons with typical_set (PCA)
#   2. Score DDAD native recons with typical_set (no PCA)
#   3. Score DDAD native recons with local_gaussian
#   4. Generate additive-noise DPS recons (tools/run_ddad_dps_sampling.py)
#   5. Score DPS recons with typical_set (PCA)
#   6. Score DPS recons with typical_set (no PCA)
#   7. Score DPS recons with local_gaussian
#
# Usage:  screen -dmS cable bash scripts/run_cable.sh
# Edit CUDA_VISIBLE_DEVICES below to pick a free GPU.

set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}

mkdir -p logs
LOG="$(pwd)/logs/run_cable.log"
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

log "=========================================="
log "Cable experiments — $(date)"
log "GPU: $CUDA_VISIBLE_DEVICES"
log "=========================================="

# ── Phase 0: DDAD native recons ──
log ""
log "Phase 0: DDAD native recons (11 × 20 seeds)"
if [ ! -f results_patches_ddad_native/sigma.txt ]; then
    python -u tools/generate_recons.py \
        --recon_config configs/recon/ddad_native_cable.yaml 2>&1 | tee -a "$LOG"
    log "  Phase 0 exit=${PIPESTATUS[0]}"
else
    log "  DDAD native recons already exist — skipping"
fi

# ── Phase 1: typical_set with PCA on DDAD native ──
log ""
log "Phase 1: typical_set + PCA on DDAD native recons"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_native_pmf \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 1 exit=${PIPESTATUS[0]}"

# ── Phase 2: typical_set no-PCA on DDAD native ──
log ""
log "Phase 2: typical_set no-PCA on DDAD native recons"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_ddad_native_pmf_nopca \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 2 exit=${PIPESTATUS[0]}"

# ── Phase 3: local_gaussian on DDAD native (σ from sigma.txt) ──
log ""
log "Phase 3: local_gaussian on DDAD native recons"
python -u evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer local_gaussian \
    --output_dir ./results_eval_ddad_native_lg \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 3 exit=${PIPESTATUS[0]}"

# ── Phase 4: additive-noise DPS recons ──
log ""
log "Phase 4: additive-noise DPS recons (σ=0.1, scale=0.5, 11 × 20 seeds)"
if [ ! -f results_patches_ddad/sigma.txt ]; then
    python -u tools/generate_recons.py \
        --recon_config configs/recon/additive_dps_cable.yaml 2>&1 | tee -a "$LOG"
    log "  Phase 4 exit=${PIPESTATUS[0]}"
else
    log "  DPS recons already exist — skipping"
fi

# ── Phase 5: typical_set with PCA on DPS ──
log ""
log "Phase 5: typical_set + PCA on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_dps_pmf \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 5 exit=${PIPESTATUS[0]}"

# ── Phase 6: typical_set no-PCA on DPS ──
log ""
log "Phase 6: typical_set no-PCA on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_ddad_dps_pmf_nopca \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 6 exit=${PIPESTATUS[0]}"

# ── Phase 7: local_gaussian on DPS ──
log ""
log "Phase 7: local_gaussian on DPS recons (σ=0.1)"
python -u evaluate.py \
    --config configs/experiment_ddad_dps.yaml \
    --skip_sampling --no_plots --scorer local_gaussian \
    --sigma_rohan 0.1 \
    --output_dir ./results_eval_ddad_dps_lg \
    --sample_names $SAMPLES 2>&1 | tee -a "$LOG"
log "  Phase 7 exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Cable experiments done — $(date)"
log "=========================================="
