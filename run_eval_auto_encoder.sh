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



# # CONFIG="configs/experiment_ddad_native.yaml"
# CONFIG="configs/experiment_ddad_native_faces.yaml"
# # CONFIG="configs/experiment_ddad_native_xray.yaml"
# RESULTS_ROOT="./results_patches_ddad_native"
# MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

# CATEGORIES=(
#   faces
# )
# # CATEGORIES=(
# #   CT
# # )
# for CATEGORY in "${CATEGORIES[@]}"; do
#     echo "======================================"
#     echo "Running category: ${CATEGORY}"
#     echo "======================================"

#     GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

#     CUDA_VISIBLE_DEVICES="0" python evaluate.py \
#         --config "${CONFIG}" \
#         --category "${CATEGORY}" \
#         --all_subcategories \
#         --results_root "${RESULTS_ROOT}" \
#         --gt_root "${GT_ROOT}" \
#         --skip_sampling \
#         --n_pca 3 \
#         --output_dir "./results_eval_ddad_native_CT_pca_32_resnet_101_autoEncoder_RGB_32_target_30_noABS_new_method_faces_smooth_1_autoencoder" \
#         --bins_pca 32 \
#         --bins_rgb 32 \
#         --sigma_rohan 1.0 \
#         --results_suffix _1 \
#         --superpixel_target_size 30 \
#         --smooth_sigma 1 \
#         --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_faces.pth" \

        

#     echo "Finished category: ${CATEGORY}"
#     echo ""
# done


# CONFIG="configs/experiment_ddad_native.yaml"
# CONFIG="configs/experiment_ddad_native_faces.yaml"
CONFIG="configs/experiment_ddad_native_xray_real.yaml"
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
  xray
)
for CATEGORY in "${CATEGORIES[@]}"; do
    echo "======================================"
    echo "Running category: ${CATEGORY}"
    echo "======================================"

    GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --config "${CONFIG}" \
        --category "${CATEGORY}" \
        --subcategory "scissors" \
        --results_root "${RESULTS_ROOT}" \
        --gt_root "${GT_ROOT}" \
        --skip_sampling \
        --n_pca 3 \
        --output_dir "./results_eval_ddad_native_CT_pca_64_resnet_101_RGB_64_autoEncoder_target_30_noABS_CT_sigma_1" \
        --bins_pca 32 \
        --bins_rgb 32 \
        --superpixel_target_size 10 \
        --results_suffix _1 \
        --smooth_sigma 1 \
        --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_xray_3.pth" \
        --max_reconstructions 1
    echo "Finished category: ${CATEGORY}"
    echo ""
done


#!/bin/bash
# run_ablation_max_recon.sh
# Ablation: vary max_reconstructions across categories
# Each category uses its own config + autoencoder

# ── Per-category settings ──────────────────────────────────────────────────
# Format: "CATEGORY|CONFIG|AUTOENCODER|SUBCATEGORY_MODE|SUBCATEGORY_VALUE|GT_ROOT_OVERRIDE"
# SUBCATEGORY_MODE: "all" → --all_subcategories, "single" → --subcategory VALUE
# GT_ROOT_OVERRIDE: "auto" → built from MVTEC_ROOT, or an explicit path


# ## Ablation: vary max_reconstructions across categories, so we keep the same config + autoencoder for each category across runs
# MVTEC_ROOT="/data/akheirandish3/mvtec_ad"
# RESULTS_ROOT="./results_patches_ddad_native"
# GPU="0"

# declare -A CATEGORY_CONFIG=(
#   [bottle]="configs/experiment_ddad_native.yaml"
#   [cable]="configs/experiment_ddad_native.yaml"
#   [capsule]="configs/experiment_ddad_native.yaml"
#   [carpet]="configs/experiment_ddad_native.yaml"
#   [grid]="configs/experiment_ddad_native.yaml"
#   [hazelnut]="configs/experiment_ddad_native.yaml"
#   [leather]="configs/experiment_ddad_native.yaml"
#   [metal_nut]="configs/experiment_ddad_native.yaml"
#   [pill]="configs/experiment_ddad_native.yaml"
#   [screw]="configs/experiment_ddad_native.yaml"
#   [tile]="configs/experiment_ddad_native.yaml"
#   [toothbrush]="configs/experiment_ddad_native.yaml"
#   [transistor]="configs/experiment_ddad_native.yaml"
#   [wood]="configs/experiment_ddad_native.yaml"
#   [zipper]="configs/experiment_ddad_native.yaml"
#   [faces]="configs/experiment_ddad_native_faces.yaml"
#   [xray]="configs/experiment_ddad_native_xray_real.yaml"
#   [CT]="configs/experiment_ddad_native.yaml"
# )

# declare -A CATEGORY_AE=(
#   [bottle]="models/mvtec_embedders/pixel_autoencoder_bottle.pth"
#   [cable]="models/mvtec_embedders/pixel_autoencoder_cable.pth"
#   [capsule]="models/mvtec_embedders/pixel_autoencoder_capsule.pth"
#   [carpet]="models/mvtec_embedders/pixel_autoencoder_carpet.pth"
#   [grid]="models/mvtec_embedders/pixel_autoencoder_grid.pth"
#   [hazelnut]="models/mvtec_embedders/pixel_autoencoder_hazelnut.pth"
#   [leather]="models/mvtec_embedders/pixel_autoencoder_leather.pth"
#   [metal_nut]="models/mvtec_embedders/pixel_autoencoder_metal_nut.pth"
#   [pill]="models/mvtec_embedders/pixel_autoencoder_pill.pth"
#   [screw]="models/mvtec_embedders/pixel_autoencoder_screw.pth"
#   [tile]="models/mvtec_embedders/pixel_autoencoder_tile.pth"
#   [toothbrush]="models/mvtec_embedders/pixel_autoencoder_toothbrush.pth"
#   [transistor]="models/mvtec_embedders/pixel_autoencoder_transistor.pth"
#   [wood]="models/mvtec_embedders/pixel_autoencoder_wood.pth"
#   [zipper]="models/mvtec_embedders/pixel_autoencoder_zipper.pth"
#   [faces]="/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_faces.pth"
#   [xray]="/data/akheirandish3/Statistical_OOD_detection/models/pixel_autoencoder_xray_3.pth"
#   [CT]="models/mvtec_embedders/pixel_autoencoder_CT.pth"
# )

# # "all" = --all_subcategories, otherwise treated as a single --subcategory value
# declare -A CATEGORY_SUBCAT=(
#   [bottle]="all"
#   [cable]="all"
#   [capsule]="all"
#   [carpet]="all"
#   [grid]="all"
#   [hazelnut]="all"
#   [leather]="all"
#   [metal_nut]="all"
#   [pill]="all"
#   [screw]="all"
#   [tile]="all"
#   [toothbrush]="all"
#   [transistor]="all"
#   [wood]="all"
#   [zipper]="all"
#   [faces]="all"
#   [xray]="scissors"   # single subcategory
#   [CT]="all"
# )

# # GT root — "auto" builds from MVTEC_ROOT, otherwise use literal path
# declare -A CATEGORY_GT_ROOT=(
#   [bottle]="auto"
#   [cable]="auto"
#   [capsule]="auto"
#   [carpet]="auto"
#   [grid]="auto"
#   [hazelnut]="auto"
#   [leather]="auto"
#   [metal_nut]="auto"
#   [pill]="auto"
#   [screw]="auto"
#   [tile]="auto"
#   [toothbrush]="auto"
#   [transistor]="auto"
#   [wood]="auto"
#   [zipper]="auto"
#   [faces]="auto"
#   [xray]="auto"
#   [CT]="auto"
# )

# # ── Which categories to run ────────────────────────────────────────────────
# CATEGORIES=(
#   faces
#   xray
#   bottle cable capsule carpet grid hazelnut leather
#   metal_nut pill screw tile toothbrush transistor wood zipper
#   CT
# )

# # ── Ablation values ────────────────────────────────────────────────────────
# MAX_RECON_VALUES=(1 10 20 40)

# # ── Shared eval hyperparameters ────────────────────────────────────────────
# N_PCA=3
# BINS_PCA=32
# BINS_RGB=32
# SUPERPIXEL_TARGET_SIZE=30
# SMOOTH_SIGMA=1
# RESULTS_SUFFIX="_1"

# # ── Main loop ──────────────────────────────────────────────────────────────
# for CATEGORY in "${CATEGORIES[@]}"; do
#     CONFIG="${CATEGORY_CONFIG[$CATEGORY]}"
#     AE_PATH="${CATEGORY_AE[$CATEGORY]}"
#     SUBCAT="${CATEGORY_SUBCAT[$CATEGORY]}"
#     GT_ROOT_VAL="${CATEGORY_GT_ROOT[$CATEGORY]}"

#     # Resolve GT root
#     if [ "$GT_ROOT_VAL" == "auto" ]; then
#         GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"
#     else
#         GT_ROOT="$GT_ROOT_VAL"
#     fi

#     # Subcategory flag
#     if [ "$SUBCAT" == "all" ]; then
#         SUBCAT_FLAG="--all_subcategories"
#     else
#         SUBCAT_FLAG="--subcategory ${SUBCAT}"
#     fi

#     for MAX_RECON in "${MAX_RECON_VALUES[@]}"; do
#         OUTPUT_DIR="./results_ablation_maxrecon_${MAX_RECON}/${CATEGORY}"

#         echo ""
#         echo "══════════════════════════════════════════════════"
#         echo "  Category: ${CATEGORY}  |  max_reconstructions: ${MAX_RECON}"
#         echo "  Output  : ${OUTPUT_DIR}"
#         echo "══════════════════════════════════════════════════"

#         CUDA_VISIBLE_DEVICES="${GPU}" python evaluate.py \
#             --config           "${CONFIG}" \
#             --category         "${CATEGORY}" \
#             ${SUBCAT_FLAG} \
#             --results_root     "${RESULTS_ROOT}" \
#             --gt_root          "${GT_ROOT}" \
#             --skip_sampling \
#             --n_pca            "${N_PCA}" \
#             --bins_pca         "${BINS_PCA}" \
#             --bins_rgb         "${BINS_RGB}" \
#             --superpixel_target_size "${SUPERPIXEL_TARGET_SIZE}" \
#             --smooth_sigma     "${SMOOTH_SIGMA}" \
#             --results_suffix   "${RESULTS_SUFFIX}" \
#             --autoencoder_path "${AE_PATH}" \
#             --max_reconstructions "${MAX_RECON}" \
#             --output_dir       "${OUTPUT_DIR}"

#         echo "  Done: ${CATEGORY} @ max_recon=${MAX_RECON}"
#     done
# done

# echo ""
# echo "All ablation runs complete."
# echo "Results saved under: ./results_ablation_maxrecon_*/"