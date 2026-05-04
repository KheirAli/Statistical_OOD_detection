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

    CUDA_VISIBLE_DEVICES="6" python evaluate.py \
        --config "${CONFIG}" \
        --category "${CATEGORY}" \
        --all_subcategories \
        --results_root "${RESULTS_ROOT}" \
        --gt_root "${GT_ROOT}" \
        --skip_sampling \
        --n_pca 3 \
        --output_dir "./results_eval_ddad_native_CT_pca_64_resnet_101_RGB_64_autoEncoder_target_30_noABS_CT_sigma_1" \
        --bins_pca 32 \
        --bins_rgb 32 \
        --sigma_rohan 1.0 \
        --superpixel_target_size 30 \
        --smooth_sigma 1 \
        --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_CT.pth"
    echo "Finished category: ${CATEGORY}"
    echo ""
done