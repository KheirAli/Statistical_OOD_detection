#!/bin/bash
# Overnight: reproduce samples_000 end-to-end with our DPS implementation.
# Usage: screen -dmS reproduce000 bash run_reproduce_000.sh
set -uo pipefail
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

mkdir -p logs results_eval_reproduce
LOG=logs/reproduce_samples_000.log

echo "=========================================="        | tee -a "$LOG"
echo "Reproduce samples_000 — started $(date)"           | tee -a "$LOG"
echo "GPUs: 0,1,2,4,7  |  config: configs/experiment_reproduce_000.yaml" | tee -a "$LOG"
echo "=========================================="        | tee -a "$LOG"

# ── Phase 1: DPS sampling (24 patches x 4 measurements) ──
echo "Phase 1: DPS sampling — $(date)"                    | tee -a "$LOG"
python -u evaluate.py \
  --config configs/experiment_reproduce_000.yaml \
  --sampling_only \
  --sample_names samples_000  2>&1 | tee -a "$LOG"
P1=${PIPESTATUS[0]}
echo "Phase 1 exit=$P1 — $(date)"                          | tee -a "$LOG"
if [ "$P1" != "0" ]; then
  echo "Phase 1 FAILED — skipping eval"                    | tee -a "$LOG"
  exit $P1
fi

# ── Phase 2: Eval with n_pca=5 / bins_pca=16 (matches earlier 0.9571 run) ──
echo ""                                                    | tee -a "$LOG"
echo "Phase 2: Eval — $(date)"                             | tee -a "$LOG"
python -u evaluate.py \
  --config configs/experiment_reproduce_000.yaml \
  --skip_sampling --no_plots \
  --n_pca 5 --bins_pca 16 \
  --sample_name samples_000  2>&1 | tee -a "$LOG"
P2=${PIPESTATUS[0]}
echo "Phase 2 exit=$P2 — $(date)"                          | tee -a "$LOG"

echo ""                                                    | tee -a "$LOG"
echo "=========================================="          | tee -a "$LOG"
echo "Done — $(date)"                                      | tee -a "$LOG"
echo "Results: results_eval_reproduce/samples_000/results.json" | tee -a "$LOG"
echo "=========================================="          | tee -a "$LOG"
