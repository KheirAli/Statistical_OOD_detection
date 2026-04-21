#!/bin/bash
# Full cable experiment: DPS sampling + eval sweeps for resnet18/resnet50
# Run: screen -S cable bash run_cable_experiments.sh
set -e
cd /home/rohan/ood/Statistical_OOD_detection

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:$PYTHONPATH

SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"
CONFIG=configs/experiment.yaml

mkdir -p logs

echo "============================================"
echo "Phase 1: DPS Sampling for all 11 cable images"
echo "Using DDIM 100 steps, 24 patches x 4 measurements"
echo "GPUs: 0,1,2,4,6"
echo "Start: $(date)"
echo "============================================"

python evaluate.py --config $CONFIG --sampling_only \
    --sample_names $SAMPLES 2>&1 | tee logs/phase1_sampling.log

echo ""
echo "Phase 1 complete at $(date)"
echo ""
echo "============================================"
echo "Phase 2: Eval sweeps (resnet18 + resnet50)"
echo "============================================"

# PCA=5, RGB=64, PCA_bins=16
for BACKBONE in resnet18 resnet50; do
    echo ""
    echo "--- $BACKBONE pca=5 rgb=64 pcabin=16 [$(date)] ---"
    python evaluate.py --config $CONFIG --skip_sampling --no_plots \
        --backbone $BACKBONE --n_pca 5 --bins_rgb 64 --bins_pca 16 \
        --sample_names $SAMPLES 2>&1 | tee logs/sweep_${BACKBONE}_pca5_rgb64_pcabin16.log
done

# PCA=5, RGB=32, PCA_bins=8
for BACKBONE in resnet18 resnet50; do
    echo ""
    echo "--- $BACKBONE pca=5 rgb=32 pcabin=8 [$(date)] ---"
    python evaluate.py --config $CONFIG --skip_sampling --no_plots \
        --backbone $BACKBONE --n_pca 5 --bins_rgb 32 --bins_pca 8 \
        --sample_names $SAMPLES 2>&1 | tee logs/sweep_${BACKBONE}_pca5_rgb32_pcabin8.log
done

# PCA=6, RGB=32, PCA_bins=8
for BACKBONE in resnet18 resnet50; do
    echo ""
    echo "--- $BACKBONE pca=6 rgb=32 pcabin=8 [$(date)] ---"
    python evaluate.py --config $CONFIG --skip_sampling --no_plots \
        --backbone $BACKBONE --n_pca 6 --bins_rgb 32 --bins_pca 8 \
        --sample_names $SAMPLES 2>&1 | tee logs/sweep_${BACKBONE}_pca6_rgb32_pcabin8.log
done

# PCA=7, RGB=32, PCA_bins=8
for BACKBONE in resnet18 resnet50; do
    echo ""
    echo "--- $BACKBONE pca=7 rgb=32 pcabin=8 [$(date)] ---"
    python evaluate.py --config $CONFIG --skip_sampling --no_plots \
        --backbone $BACKBONE --n_pca 7 --bins_rgb 32 --bins_pca 8 \
        --sample_names $SAMPLES 2>&1 | tee logs/sweep_${BACKBONE}_pca7_rgb32_pcabin8.log
done

echo ""
echo "============================================"
echo "All experiments complete at $(date)"
echo "Sweep JSON results in: results_eval/"
echo "Logs in: logs/"
echo "============================================"
