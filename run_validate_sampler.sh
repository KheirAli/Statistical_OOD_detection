#!/bin/bash
# Validate our sampler two ways:
#   1) Filter our recons to the same 7 patches theirs has, re-run eval
#   2) Side-by-side recon comparison + mean-abs-diff figures
# Usage: screen -dmS validate bash run_validate_sampler.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs figures_validation results_eval_ours_filtered
LOG=logs/validate_sampler.log
: > "$LOG"  # truncate

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "Sampler validation — started $(date)"
log "=========================================="

# ── Step 1: Build filtered recon dir (symlinks to patches 3,5,6,7,8,9,10) ──
log ""
log "Step 1: Build filtered recon dir (symlinks)"
FILTERED_ROOT=./results_patches_filtered/samples_000
rm -rf "$FILTERED_ROOT"
mkdir -p "$FILTERED_ROOT"
for p in 3 5 6 7 8 9 10; do
    src="$(realpath ./results_patches/samples_000/Combined_half_sigma_batched_${p}_4)"
    if [ -d "$src" ]; then
        ln -s "$src" "$FILTERED_ROOT/Combined_half_sigma_batched_${p}_4"
        log "  linked patch $p"
    else
        log "  MISSING: $src"
    fi
done

# Ensure figures/samples_000 SP mask exists (reused from earlier runs)
if [ ! -f figures/samples_000/mask.png ]; then
    log "  generating superpixels for samples_000..."
    python -u super_pixel_generation.py \
        --input_image=/data/akheirandish3/mvtec_ad/cable/test/combined/000.png \
        --output_dir=figures/samples_000 2>&1 | tee -a "$LOG"
fi

# ── Step 2: Eval on filtered recons ──
log ""
log "Step 2: Eval on filtered recons (7 patches) — $(date)"
python -u evaluate.py \
    --config configs/experiment_ours_filtered.yaml \
    --skip_sampling --no_plots \
    --n_pca 5 --bins_pca 16 \
    --sample_name samples_000 2>&1 | tee -a "$LOG"
log "  exit=${PIPESTATUS[0]}"

# ── Step 3: Side-by-side recon comparison ──
log ""
log "Step 3: Side-by-side recon comparison — $(date)"
python -u tools/compare_recons.py 2>&1 | tee -a "$LOG"
log "  exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "Done — $(date)"
log "Figures: figures_validation/"
log "Eval:    results_eval_ours_filtered/samples_000/results.json"
log "=========================================="
