#!/bin/bash
# reproduce_results.sh
# Reproduce the best-AE results (pixel ROC-AUC @ sigma=5) for all 15 MVTec
# categories + CT + xray + faces (FFHQ), using the per-category PixelAutoEncoder
# on the ImageNet-pretrained ResNet-101 backbone.
#
# Every command is evaluation-only (--skip_sampling): it consumes the DPS
# reconstructions already present in ./results_patches_ddad_native_* and the
# AE / backbone checkpoints listed in REPRODUCE_RESULTS.md.
#
# Usage (env `ood` must be active, run from the repo root):
#   GPU=0 bash reproduce_results.sh                 # everything
#   GPU=0 bash reproduce_results.sh carpet pill CT  # a subset
#
# Results land in $OUT/<category>/ as per-subcategory JSONs + a summary JSON.
# Expected numbers: see the table in REPRODUCE_RESULTS.md.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU="${GPU:-0}"
OUT="${OUT:-./results_eval_reproduction}"
DIAG="experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402"
MVTEC="/data/akheirandish3/mvtec_ad"
RES="./results_patches_ddad_native"

# 12 categories: old-recipe AE on the pretrained (feat0) backbone
declare -A AE=(
  [bottle]=all_categories_embedder/models/pixel_ae_pretrained/bottle.pth
  [cable]=all_categories_embedder/models/pixel_ae_pretrained/cable.pth
  [capsule]=all_categories_embedder/models/pixel_ae_pretrained/capsule.pth
  [grid]=all_categories_embedder/models/pixel_ae_pretrained/grid.pth
  [hazelnut]=all_categories_embedder/models/pixel_ae_pretrained/hazelnut.pth
  [leather]=all_categories_embedder/models/pixel_ae_pretrained/leather.pth
  [metal_nut]=all_categories_embedder/models/pixel_ae_pretrained/metal_nut.pth
  [screw]=all_categories_embedder/models/pixel_ae_pretrained/screw.pth
  [tile]=all_categories_embedder/models/pixel_ae_pretrained/tile.pth
  [toothbrush]=all_categories_embedder/models/pixel_ae_pretrained/toothbrush.pth
  [transistor]=all_categories_embedder/models/pixel_ae_pretrained/transistor.pth
  [zipper]=all_categories_embedder/models/pixel_ae_pretrained/zipper.pth
  # carpet + wood: corrected-recipe (seeded) AEs — the reported numbers use these
  [carpet]=all_categories_embedder/models/pixel_ae_pretrained_fixed/carpet.pth
  [wood]=all_categories_embedder/models/pixel_ae_pretrained_fixed/wood.pth
)

run_mvtec() {  # $1 = category  (backbone-injection path, suffix _1)
  local cat="$1"
  DDAD_FE_WEIGHTS="$DIAG/$cat/checkpoints/feat0.pth" CUDA_VISIBLE_DEVICES="$GPU" python \
    all_categories_embedder/evaluate_ddad_fe.py \
    --config configs/experiment_ddad_native.yaml --category "$cat" --all_subcategories \
    --results_root "$RES" --gt_root "$MVTEC/$cat/ground_truth" --skip_sampling \
    --autoencoder_path "${AE[$cat]}" \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 30 --smooth_sigma 1 \
    --max_reconstructions 50 --results_suffix _1 --output_dir "$OUT/$cat"
}

run_pill() {  # pill: plain evaluate.py (torchvision backbone), original AE, suffix _3
  CUDA_VISIBLE_DEVICES="$GPU" python evaluate.py \
    --config configs/experiment_ddad_native.yaml --category pill --all_subcategories \
    --results_root "$RES" --gt_root "$MVTEC/pill/ground_truth" --skip_sampling \
    --autoencoder_path models/mvtec_embedders/pixel_autoencoder_pill.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 30 --smooth_sigma 1 \
    --max_reconstructions 50 --results_suffix _3 --output_dir "$OUT/pill"
}

run_CT() {  # CT: domain AE, bins 64/64, no recon suffix
  CUDA_VISIBLE_DEVICES="$GPU" python evaluate.py \
    --config configs/experiment_ddad_native_xray.yaml --category CT --all_subcategories \
    --results_root "$RES" --gt_root "$MVTEC/CT/ground_truth" --skip_sampling \
    --autoencoder_path models/pixel_autoencoder_CT.pth \
    --n_pca 3 --bins_pca 64 --bins_rgb 64 --sigma_rohan 1.0 \
    --superpixel_target_size 30 --smooth_sigma 1 \
    --max_reconstructions 40 --output_dir "$OUT/CT"
}

run_xray() {  # xray/scissors: last-layer fine-tuned backbone + its AE (best variant)
  DDAD_FE_WEIGHTS=experiments/xray_finetune_mine/last_layer/checkpoints/feat4.pth \
  CUDA_VISIBLE_DEVICES="$GPU" python all_categories_embedder/evaluate_ddad_fe.py \
    --config configs/experiment_ddad_native_xray_real.yaml \
    --category xray --subcategory scissors \
    --results_root "$RES" --gt_root "$MVTEC/xray/ground_truth" --skip_sampling \
    --autoencoder_path all_categories_embedder/models/xray_mine/last_layer.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 10 --smooth_sigma 1 \
    --max_reconstructions 40 --results_suffix _1 --output_dir "$OUT/xray"
}

run_faces() {  # FFHQ faces: domain AE, suffix _1
  CUDA_VISIBLE_DEVICES="$GPU" python evaluate.py \
    --config configs/experiment_ddad_native_faces.yaml --category faces --all_subcategories \
    --results_root "$RES" --gt_root "$MVTEC/faces/ground_truth" --skip_sampling \
    --autoencoder_path models/pixel_autoencoder_faces.pth \
    --n_pca 3 --bins_pca 32 --bins_rgb 32 --sigma_rohan 1.0 --results_suffix _1 \
    --superpixel_target_size 30 --smooth_sigma 1 --output_dir "$OUT/faces"
}

ALL=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile
     toothbrush transistor wood zipper CT xray faces)
TARGETS=("$@"); [ "${#TARGETS[@]}" -eq 0 ] && TARGETS=("${ALL[@]}")

for t in "${TARGETS[@]}"; do
  echo "════════ $t (GPU $GPU) ════════"
  case "$t" in
    pill)  run_pill ;;
    CT)    run_CT ;;
    xray)  run_xray ;;
    faces) run_faces ;;
    *)     run_mvtec "$t" ;;
  esac || echo "[WARN] $t failed"
done
echo "Done -> $OUT   (report px_roc_auc at eval_sigma sigma_5.0, defect subcats only)"
