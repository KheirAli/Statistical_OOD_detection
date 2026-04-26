#!/usr/bin/env bash
# Train PatchCore (Roth et al., CVPR'22) on every standard MVTec-AD category.
#
# Outputs land at:
#   /data2/rohan/baseline_ckpts/patchcore_mvtec_full/MVTecAD_Results/<log_group>/models/mvtec_<class>/
#
# Each save dir contains `nnscorer_search_index.faiss` + `patchcore_params.pkl`,
# which is what `ood/baselines/patchcore.py` loads via PatchCore.load_from_path().
#
# Hyperparameters match the IM224 baseline from upstream sample_training.sh:
#   wide_resnet50_2 backbone, layers 2+3, coreset 10%, embed 1024→1024,
#   patchsize 3, num_nn 1, image 256→224.
# Reported in the upstream README to give 99.2% I-AUROC, 98.1% Px-AUROC, 94.4 PRO.
#
# "Training" is fast (~30-90 sec/class) — feature extraction + greedy coreset
# selection + FAISS index build, no gradient updates.

set -euo pipefail

REPO="${PATCHCORE_REPO:-/home/rohan/ood/baseline-algos-clone/patchcore-inspection}"
DATA="${MVTEC_DATA:-/data/akheirandish3/mvtec_ad}"
OUT="${PATCHCORE_OUT:-/data2/rohan/baseline_ckpts/patchcore_mvtec_full}"
GPU="${PATCHCORE_GPU:-0}"

CONDA_BASE="${CONDA_BASE:-/home/rohan/miniconda3}"
CONDA_ENV="${CONDA_ENV:-ood}"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CATS=(bottle cable capsule carpet grid hazelnut leather metal_nut pill
      screw tile toothbrush transistor wood zipper)
CAT_FLAGS=()
for c in "${CATS[@]}"; do CAT_FLAGS+=("-d" "$c"); done

cd "$REPO"
CUDA_VISIBLE_DEVICES="$GPU" env PYTHONPATH=src python bin/run_patchcore.py \
    --gpu 0 --seed 0 --save_patchcore_model \
    --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
    --log_project MVTecAD_Results \
    "$OUT" \
    patch_core \
        -b wideresnet50 -le layer2 -le layer3 \
        --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 1 --patchsize 3 \
    sampler -p 0.1 approx_greedy_coreset \
    dataset --resize 256 --imagesize 224 "${CAT_FLAGS[@]}" mvtec "$DATA"
