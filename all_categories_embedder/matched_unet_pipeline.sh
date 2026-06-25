#!/bin/bash
# matched_unet_pipeline.sh  CAT  UNET_CKPT
# Re-fine-tune a category's ResNet-101 backbone against its OWN category-specific
# reconstruction UNet (the model that produced its recons), then train the 4
# per-method pixel-AEs and eval all 4 methods through OUR detector. Same eval
# hyperparameters as the figure (AE->3, SP=30, smooth=1, _1 recons).
# Writes to SEPARATE dirs; does not touch the figure until folded in explicitly.
#
#   GPU_FT=0 bash all_categories_embedder/matched_unet_pipeline.sh wood   /data/akherandish3/MVTec/wood/2000
#   GPU_FT=0 bash all_categories_embedder/matched_unet_pipeline.sh carpet /data/akherandish3/MVTec/carpet/2500
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SCRIPT_DIR")"; cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
CAT="${1:?category}"; UNET="${2:?unet_ckpt}"
GPU_FT="${GPU_FT:-0}"
S="experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts"
CK="experiments/${CAT}_catunet_finetune/checkpoints"
DATA="/data/akheirandish3/mvtec_ad/${CAT}/train/good"
GT="/data/akheirandish3/mvtec_ad/${CAT}/ground_truth"
EVAL_ROOT="results_eval_${CAT}_catunet"
ALPHA=0.3
LOG="/tmp/catunet_${CAT}"; mkdir -p "$LOG" "$CK/logs"

echo "######## ${CAT}  UNet=${UNET}  GPU_FT=${GPU_FT}  $(date) ########"

# ---- Phase 1: backbone fine-tune (ddad_env) ----
if [ ! -f "$CK/feat8.pth" ]; then
  echo "[${CAT}] full FT"
  CUDA_VISIBLE_DEVICES=$GPU_FT conda run --no-capture-output -n ddad_env python \
    $S/ddad_da_finetune.py --category "$CAT" --unet_ckpt "$UNET" --out_dir "$CK" --da_epochs 8 \
    > "$CK/logs/train.log" 2>&1 || { echo "[FAIL fullft $CAT]"; exit 1; }
fi
if [ ! -f "$CK/lora/r8_100/feat8.pth" ]; then
  echo "[${CAT}] LoRA r8_100"
  CUDA_VISIBLE_DEVICES=$GPU_FT conda run --no-capture-output -n ddad_env python \
    $S/ddad_da_finetune.py --category "$CAT" --unet_ckpt "$UNET" --out_dir "$CK/lora/r8_100" \
    --da_epochs 8 --lora_rank 8 --lora_alpha 1.0 > "$CK/logs/lora.log" 2>&1 || echo "[FAIL lora $CAT]"
fi
# interp backbone
IW="$CK/_interp/${CAT}_a${ALPHA}.pth"; mkdir -p "$CK/_interp"
[ -f "$IW" ] || conda run --no-capture-output -n ood python all_categories_embedder/_interp_states.py \
    --a "$CK/feat0.pth" --b "$CK/feat8.pth" --alpha "$ALPHA" --out "$IW"

# ---- Phase 2: per-method AE + eval (ood), 4 methods in parallel ----
declare -A BW=( [pretrained]="$CK/feat0.pth" [fullft]="$CK/feat8.pth"
                [lora]="$CK/lora/r8_100/feat8.pth" [interp]="$IW" )
declare -A KGPU=( [pretrained]="${GPU_PT:-0}" [fullft]="${GPU_FF:-1}" [lora]="${GPU_LO:-4}" [interp]="${GPU_IN:-6}" )

run_kind() {
  local KIND="$1" GPU="${KGPU[$1]}" W="${BW[$1]}"
  local AE="all_categories_embedder/models/pixel_ae_${KIND}_catunet/${CAT}.pth"
  local OUT="${EVAL_ROOT}/ae_${KIND}/${CAT}"
  mkdir -p "$(dirname "$AE")"
  echo "[$CAT/$KIND] GPU=$GPU  $(date)"
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n ood python \
    all_categories_embedder/train_pixel_ae_interp.py --weights "$W" --data_dir "$DATA" \
    --save_model "$AE" --latent_dim 3 --num_epochs 30 \
    --seed "${AE_SEED:-0}" --n_restarts "${AE_RESTARTS:-3}" > "$LOG/ae_${KIND}.log" 2>&1
  DDAD_FE_WEIGHTS="$W" CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n ood python \
    all_categories_embedder/evaluate_ddad_fe.py --config configs/experiment_ddad_native.yaml \
    --category "$CAT" --all_subcategories --results_root ./results_patches_ddad_native --gt_root "$GT" \
    --skip_sampling --autoencoder_path "$AE" --n_pca 3 --bins_pca 32 --bins_rgb 32 \
    --superpixel_target_size 30 --smooth_sigma 1 --max_reconstructions 50 \
    --results_suffix _1 --output_dir "$OUT" > "$LOG/eval_${KIND}.log" 2>&1
  echo "[$CAT/$KIND] done  $(date)"
}
for KIND in pretrained fullft lora interp; do run_kind "$KIND" & done
wait
echo "==== matched_unet ${CAT} DONE  $(date) ===="
