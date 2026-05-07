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

        
# #superpixel 30
#     echo "Finished category: ${CATEGORY}"
#     echo ""
# done


# CONFIG="configs/experiment_ddad_native.yaml"
# CONFIG="configs/experiment_ddad_native_faces.yaml"
# CONFIG="configs/experiment_ddad_native_xray_real.yaml"
# RESULTS_ROOT="./results_patches_ddad_native"
# MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

# # CATEGORIES=(
# #   bottle
# #   cable
# #   capsule
# #   carpet
# #   grid
# #   hazelnut
# #   leather
# #   metal_nut
# #   pill
# #   screw
# #   tile
# #   toothbrush
# #   transistor
# #   wood
# #   zipper
# # )
# # CATEGORIES=(
# #   faces
# # )
# CATEGORIES=(
#   xray
# )
# for CATEGORY in "${CATEGORIES[@]}"; do
#     echo "======================================"
#     echo "Running category: ${CATEGORY}"
#     echo "======================================"

#     GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

#     CUDA_VISIBLE_DEVICES="3" python evaluate.py \
#         --config "${CONFIG}" \
#         --category "${CATEGORY}" \
#         --subcategory "scissors" \
#         --results_root "${RESULTS_ROOT}" \
#         --gt_root "${GT_ROOT}" \
#         --skip_sampling \
#         --n_pca 3 \
#         --output_dir "./results_eval_ddad_native_CT_pca_64_resnet_101_RGB_64_autoEncoder_target_30_noABS_CT_sigma_1" \
#         --bins_pca 32 \
#         --bins_rgb 32 \
#         --superpixel_target_size 10 \
#         --results_suffix _1 \
#         --smooth_sigma 1 \
#         --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_xray_3.pth" \
#         --max_reconstructions 40
#     echo "Finished category: ${CATEGORY}"
#     echo ""
# done



# CONFIG="configs/experiment_ddad_native.yaml"
# CONFIG="configs/experiment_ddad_native_faces.yaml"
# CONFIG="configs/experiment_ddad_native_xray.yaml"
# RESULTS_ROOT="./results_patches_ddad_native"
# MVTEC_ROOT="/data/akheirandish3/mvtec_ad"

# # CATEGORIES=(
# #   bottle
# #   cable
# #   capsule
# #   carpet
# #   grid
# #   hazelnut
# #   leather
# #   metal_nut
# #   pill
# #   screw
# #   tile
# #   toothbrush
# #   transistor
# #   wood
# #   zipper
# # )
# # CATEGORIES=(
# #   faces
# # )
# CATEGORIES=(
#   CT
# )
# for CATEGORY in "${CATEGORIES[@]}"; do
#     echo "======================================"
#     echo "Running category: ${CATEGORY}"
#     echo "======================================"

#     GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

#     CUDA_VISIBLE_DEVICES="7" python evaluate.py \
#         --config "${CONFIG}" \
#         --category "${CATEGORY}" \
#         --all_subcategories \
#         --results_root "${RESULTS_ROOT}" \
#         --gt_root "${GT_ROOT}" \
#         --skip_sampling \
#         --n_pca 3 \
#         --output_dir "./results_eval_ddad_native_CT_pca_32_resnet_101_RGB_32_autoEncoder_target_30_noABS_CT_sigma_1" \
#         --bins_pca 32 \
#         --bins_rgb 32 \
#         --sigma_rohan 1.0 \
#         --superpixel_target_size 30 \
#         --smooth_sigma 1 \
#         --autoencoder_path "/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_CT.pth" \
#         --max_reconstructions 40 
#     echo "Finished category: ${CATEGORY}"
#     echo ""
# done



#!/bin/bash
# run_ablation_max_recon.sh
# Ablation: vary max_reconstructions across categories
# Each category uses its own config + autoencoder

# ── Per-category settings ──────────────────────────────────────────────────
# Format: "CATEGORY|CONFIG|AUTOENCODER|SUBCATEGORY_MODE|SUBCATEGORY_VALUE|GT_ROOT_OVERRIDE"
# SUBCATEGORY_MODE: "all" → --all_subcategories, "single" → --subcategory VALUE
# GT_ROOT_OVERRIDE: "auto" → built from MVTEC_ROOT, or an explicit path


# ## Ablation: vary max_reconstructions across categories, so we keep the same config + autoencoder for each category across runs

MVTEC_ROOT="/data/akheirandish3/mvtec_ad"
RESULTS_ROOT="./results_patches_ddad_native"

# ── Per-category: config file ──────────────────────────────────────────────
declare -A CATEGORY_CONFIG=(
  [bottle]="configs/experiment_ddad_native.yaml"
  [cable]="configs/experiment_ddad_native.yaml"
  [capsule]="configs/experiment_ddad_native.yaml"
  [carpet]="configs/experiment_ddad_native.yaml"
  [grid]="configs/experiment_ddad_native.yaml"
  [hazelnut]="configs/experiment_ddad_native.yaml"
  [leather]="configs/experiment_ddad_native.yaml"
  [metal_nut]="configs/experiment_ddad_native.yaml"
  [pill]="configs/experiment_ddad_native.yaml"
  [screw]="configs/experiment_ddad_native.yaml"
  [tile]="configs/experiment_ddad_native.yaml"
  [toothbrush]="configs/experiment_ddad_native.yaml"
  [transistor]="configs/experiment_ddad_native.yaml"
  [wood]="configs/experiment_ddad_native.yaml"
  [zipper]="configs/experiment_ddad_native.yaml"
  [faces]="configs/experiment_ddad_native_faces.yaml"
  [xray]="configs/experiment_ddad_native_xray_real.yaml"
  [CT]="configs/experiment_ddad_native_xray.yaml"        # ← CT uses xray config
)

# ── Per-category: autoencoder path ────────────────────────────────────────
declare -A CATEGORY_AE=(
  [bottle]="models/mvtec_embedders/pixel_autoencoder_bottle.pth"
  [cable]="models/mvtec_embedders/pixel_autoencoder_cable.pth"
  [capsule]="models/mvtec_embedders/pixel_autoencoder_capsule.pth"
  [carpet]="models/mvtec_embedders/pixel_autoencoder_carpet.pth"
  [grid]="models/mvtec_embedders/pixel_autoencoder_grid.pth"
  [hazelnut]="models/mvtec_embedders/pixel_autoencoder_hazelnut.pth"
  [leather]="models/mvtec_embedders/pixel_autoencoder_leather.pth"
  [metal_nut]="models/mvtec_embedders/pixel_autoencoder_metal_nut.pth"
  [pill]="models/mvtec_embedders/pixel_autoencoder_pill.pth"
  [screw]="models/mvtec_embedders/pixel_autoencoder_screw.pth"
  [tile]="models/mvtec_embedders/pixel_autoencoder_tile.pth"
  [toothbrush]="models/mvtec_embedders/pixel_autoencoder_toothbrush.pth"
  [transistor]="models/mvtec_embedders/pixel_autoencoder_transistor.pth"
  [wood]="models/mvtec_embedders/pixel_autoencoder_wood.pth"
  [zipper]="models/mvtec_embedders/pixel_autoencoder_zipper.pth"
  [faces]="/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_faces.pth"
  [xray]="/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_xray_3.pth"
  [CT]="/data/akherandish3/Statistical_OOD_detection/models/pixel_autoencoder_CT.pth"
)

# ── Per-category: subcategory ("all" or a specific name) ──────────────────
declare -A CATEGORY_SUBCAT=(
  [bottle]="all"
  [cable]="all"
  [capsule]="all"
  [carpet]="all"
  [grid]="all"
  [hazelnut]="all"
  [leather]="all"
  [metal_nut]="all"
  [pill]="all"
  [screw]="all"
  [tile]="all"
  [toothbrush]="all"
  [transistor]="all"
  [wood]="all"
  [zipper]="all"
  [faces]="all"
  [xray]="scissors"   # ← xray uses single subcategory
  [CT]="all"
)

# ── Per-category: results suffix ("" = no suffix) ─────────────────────────
declare -A CATEGORY_SUFFIX=(
  [bottle]="_1"
  [cable]="_1"
  [capsule]="_1"
  [carpet]="_1"
  [grid]="_1"
  [hazelnut]="_1"
  [leather]="_1"
  [metal_nut]="_1"
  [pill]="_1"
  [screw]="_1"
  [tile]="_1"
  [toothbrush]="_1"
  [transistor]="_1"
  [wood]="_1"
  [zipper]="_1"
  [faces]="_1"
  [xray]="_1"         # ← xray uses _2
  [CT]=""             # ← CT has no suffix
)

# ── Per-category: GPU ──────────────────────────────────────────────────────
declare -A CATEGORY_GPU=(
  [bottle]="5"
  [cable]="5"
  [capsule]="5"
  [carpet]="5"
  [grid]="5"
  [hazelnut]="5"
  [leather]="5"
  [metal_nut]="5"
  [pill]="5"
  [screw]="5"
  [tile]="5"
  [toothbrush]="5"
  [transistor]="5"
  [wood]="5"
  [zipper]="5"
  [faces]="5"
  [xray]="5"
  [CT]="5"            # ← CT uses GPU 7
)

# ── Per-category: sigma_rohan (only used for local_gaussian scorer) ────────
declare -A CATEGORY_SIGMA_ROHAN=(
  [bottle]=""
  [cable]=""
  [capsule]=""
  [carpet]=""
  [grid]=""
  [hazelnut]=""
  [leather]=""
  [metal_nut]=""
  [pill]=""
  [screw]=""
  [tile]=""
  [toothbrush]=""
  [transistor]=""
  [wood]=""
  [zipper]=""
  [faces]=""
  [xray]=""
  [CT]="1.0"          # ← CT uses sigma_rohan
)

# ── Which categories to run ────────────────────────────────────────────────
CATEGORIES=(
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
#   faces
  xray
#   CT
)

# ── Ablation values ────────────────────────────────────────────────────────
# MAX_RECON_VALUES=(1 10 20 40)
# MAX_RECON_VALUES=(1)
SUPERPIXEL_TARGET_SIZES=(10)
# ── Shared eval hyperparameters ────────────────────────────────────────────
N_PCA=3
BINS_PCA=32
BINS_RGB=32
# SUPERPIXEL_TARGET_SIZE=30
MAX_RECON=40
SMOOTH_SIGMA=1

# ── Main loop ──────────────────────────────────────────────────────────────
for CATEGORY in "${CATEGORIES[@]}"; do
    CONFIG="${CATEGORY_CONFIG[$CATEGORY]}"
    AE_PATH="${CATEGORY_AE[$CATEGORY]}"
    SUBCAT="${CATEGORY_SUBCAT[$CATEGORY]}"
    SUFFIX="${CATEGORY_SUFFIX[$CATEGORY]}"
    GPU="${CATEGORY_GPU[$CATEGORY]}"
    SIGMA_ROHAN="${CATEGORY_SIGMA_ROHAN[$CATEGORY]}"
    GT_ROOT="${MVTEC_ROOT}/${CATEGORY}/ground_truth"

    # Build subcategory flag
    if [ "$SUBCAT" == "all" ]; then
        SUBCAT_FLAG="--all_subcategories"
    else
        SUBCAT_FLAG="--subcategory ${SUBCAT}"
    fi

    # Build optional sigma_rohan flag
    if [ -n "$SIGMA_ROHAN" ]; then
        SIGMA_FLAG="--sigma_rohan ${SIGMA_ROHAN}"
    else
        SIGMA_FLAG=""
    fi

    # Build optional results_suffix flag
    if [ -n "$SUFFIX" ]; then
        SUFFIX_FLAG="--results_suffix ${SUFFIX}"
    else
        SUFFIX_FLAG=""
    fi

    # for MAX_RECON in "${MAX_RECON_VALUES[@]}"; do
    for SUPERPIXEL_TARGET_SIZE in "${SUPERPIXEL_TARGET_SIZES[@]}"; do
        # OUTPUT_DIR="./results_ablation_maxrecon_${MAX_RECON}/${CATEGORY}"
        OUTPUT_DIR="./results_ablation_superpixel_${SUPERPIXEL_TARGET_SIZE}/${CATEGORY}"

        echo ""
        echo "══════════════════════════════════════════════════"
        echo "  Category : ${CATEGORY}"
        echo "  MaxRecon : ${MAX_RECON}"
        echo "  Config   : ${CONFIG}"
        echo "  Suffix   : '${SUFFIX}'"
        echo "  GPU      : ${GPU}"
        echo "  Output   : ${OUTPUT_DIR}"
        echo "══════════════════════════════════════════════════"

        CUDA_VISIBLE_DEVICES="${GPU}" python evaluate.py \
            --config                  "${CONFIG}" \
            --category                "${CATEGORY}" \
            ${SUBCAT_FLAG} \
            --results_root            "${RESULTS_ROOT}" \
            --gt_root                 "${GT_ROOT}" \
            --skip_sampling \
            --n_pca                   "${N_PCA}" \
            --bins_pca                "${BINS_PCA}" \
            --bins_rgb                "${BINS_RGB}" \
            --superpixel_target_size  "${SUPERPIXEL_TARGET_SIZE}" \
            --smooth_sigma            "${SMOOTH_SIGMA}" \
            --autoencoder_path        "${AE_PATH}" \
            --max_reconstructions     "${MAX_RECON}" \
            --output_dir              "${OUTPUT_DIR}" \
            ${SUFFIX_FLAG} \
            ${SIGMA_FLAG}

        STATUS=$?
        if [ $STATUS -ne 0 ]; then
            echo "  [WARN] ${CATEGORY} @ max_recon=${MAX_RECON} exited with status ${STATUS}"
        else
            echo "  Done: ${CATEGORY} @ max_recon=${MAX_RECON}"
        fi
    done
done

echo ""
echo "All ablation runs complete."
echo "Results under: ./results_ablation_maxrecon_*/"