#!/bin/bash

# Stop if a command fails
set -e

# CONFIG="configs/experiment_ddad_native.yaml"
# CONFIG="configs/experiment_ddad_native_faces.yaml"
CONFIG="configs/experiment_ddad_native_xray.yaml"
RESULTS_ROOT="./results_patches_ddad_native"
MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

# CATEGORIES=(
#   bottle
#   cable
#   capsule
#   carpet
#   grid
#   hazelnut
#   leather
#   metal_nut
#   pill
#   screw
#   tile
#   toothbrush
#   transistor
#   wood
#   zipper
# )
# CATEGORIES=(
#   faces
# )
CATEGORIES=(
  CT
)
for CATEGORY in "${CATEGORIES[@]}"; do
    echo "======================================"
    echo "Running category: ${CATEGORY}"
    echo "======================================"

    GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

    CUDA_VISIBLE_DEVICES="7" python evaluate.py \
        --config "${CONFIG}" \
        --category "${CATEGORY}" \
        --all_subcategories \
        --results_root "${RESULTS_ROOT}" \
        --gt_root "${GT_ROOT}" \
        --skip_sampling \
        --n_pca 3 \
        --output_dir "./results_eval_ddad_native_CT_pca_16_resnet_101_autoEncoder_CT" \
        --bins_pca 16 \
        --bins_rgb 32 \
        --sigma_rohan 1.0 \
        --autoencoder_path "/home/akheirandish3/diffusion-posterior-sampling/models/pixel_autoencoder_CT_same_data.pth"

    echo "Finished category: ${CATEGORY}"
    echo ""
done