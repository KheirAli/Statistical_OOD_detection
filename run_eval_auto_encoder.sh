#!/bin/bash

# Stop if a command fails
# set -e

# CONFIG="configs/experiment_ddad_native.yaml"
# # CONFIG="configs/experiment_ddad_native_faces.yaml"
# # CONFIG="configs/experiment_ddad_native_xray.yaml"
# RESULTS_ROOT="./results_patches_ddad_native"
# MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

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
# # CATEGORIES=(
# #   faces
# # )
# # CATEGORIES=(
# #   CT
# # )
# for CATEGORY in "${CATEGORIES[@]}"; do
#     echo "======================================"
#     echo "Running category: ${CATEGORY}"
#     echo "======================================"

#     GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

#     CUDA_VISIBLE_DEVICES="6" python evaluate.py \
#         --config "${CONFIG}" \
#         --category "${CATEGORY}" \
#         --all_subcategories \
#         --results_root "${RESULTS_ROOT}" \
#         --gt_root "${GT_ROOT}" \
#         --skip_sampling \
#         --n_pca 3 \
#         --output_dir "./results_eval_ddad_native_CT_pca_32_resnet_101_autoEncoder_RGB_32_target_20_noABS_0.1" \
#         --bins_pca 32 \
#         --bins_rgb 32 \
#         --sigma_rohan 0.1 \
#         --results_suffix _1 \
#         --superpixel_target_size 20 \
#         --autoencoder_path "models/mvtec_embedders/pixel_autoencoder_${CATEGORY}.pth" \
#         --smooth_sigma 0.1

        

#     echo "Finished category: ${CATEGORY}"
#     echo ""
# done



# CONFIG="configs/experiment_ddad_native.yaml"
CONFIG="configs/experiment_ddad_native_faces.yaml"
# CONFIG="configs/experiment_ddad_native_xray.yaml"
RESULTS_ROOT="./results_patches_ddad_native"
MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

CATEGORIES=(
  faces
)
# CATEGORIES=(
#   CT
# )
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
        --output_dir "./results_eval_ddad_native_CT_pca_32_resnet_101_autoEncoder_RGB_32_target_30_noABS_new_method_faces_smooth_1_autoencoder" \
        --bins_pca 32 \
        --bins_rgb 32 \
        --sigma_rohan 1.0 \
        --results_suffix _1 \
        --superpixel_target_size 30 \
        --smooth_sigma 1 \
        --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_faces.pth" \

        

    echo "Finished category: ${CATEGORY}"
    echo ""
done
