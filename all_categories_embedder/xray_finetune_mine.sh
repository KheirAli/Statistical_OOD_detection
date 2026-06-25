#!/bin/bash
# xray_finetune_mine.sh
# Our own xray fine-tuning study (NOT rohan's checkpoints). DDAD-DA fine-tune of
# ImageNet ResNet-101 against rohan's xray diffusion UNet (xray/2000), in THREE
# scopes, then a fixed AE (MSE+cos) per backbone, then eval xray/scissors:
#   pretrained  : feat0 (ImageNet anchor)
#   last_layer  : freeze backbone, train only layer3   (--train_scope last_layer)
#   full        : full fine-tune                       (--train_scope full)
#   lora        : LoRA rank-8                           (--lora_rank 8)
#
#   bash all_categories_embedder/xray_finetune_mine.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SCRIPT_DIR")"; cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
S="experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts"
UNET="/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000"
EXP="experiments/xray_finetune_mine"
DATA="/data/akheirandish3/mvtec_ad/xray/train/good"
GT="/data/akheirandish3/mvtec_ad/xray/ground_truth"
CONFIG="configs/experiment_ddad_native_xray_real.yaml"
AE_DIR="all_categories_embedder/models/xray_mine"; mkdir -p "$AE_DIR"
OUT_ROOT="results_eval_xray_mine"
EPOCHS="${EPOCHS:-4}"
LOG=/tmp/xraymine; mkdir -p "$LOG"

# GPU assignment (free: 2 4 5 7)
declare -A FTGPU=( [full]=2 [last_layer]=4 [lora]=5 )

# ---- Phase 1: fine-tune 3 scopes in parallel (ddad_env) ----
finetune() {  # scope gpu extra_args...
    local scope="$1" gpu="$2"; shift 2
    local out="$EXP/$scope/checkpoints"; mkdir -p "$out"
    [ -f "$out/feat${EPOCHS}.pth" ] && { echo "[skip ft $scope]"; return; }
    echo "[ft $scope] GPU=$gpu  $(date +%H:%M)"
    CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n ddad_env python \
        "$S/ddad_da_finetune.py" --category xray --unet_ckpt "$UNET" \
        --out_dir "$out" --da_epochs "$EPOCHS" --seed 0 "$@" \
        > "$LOG/ft_${scope}.log" 2>&1 && echo "[ft $scope] done $(date +%H:%M)" || echo "[FAIL ft $scope]"
}
finetune full       "${FTGPU[full]}"       --train_scope full &
finetune last_layer "${FTGPU[last_layer]}" --train_scope last_layer &
finetune lora       "${FTGPU[lora]}"       --lora_rank 8 --lora_alpha 1.0 &
wait
echo "==== phase1 fine-tune done  $(date) ===="

# ---- Phase 2: AE (MSE+cos) + eval, 4 methods in parallel ----
declare -A BW=(
  [pretrained]="$EXP/full/checkpoints/feat0.pth"
  [last_layer]="$EXP/last_layer/checkpoints/feat${EPOCHS}.pth"
  [full]="$EXP/full/checkpoints/feat${EPOCHS}.pth"
  [lora]="$EXP/lora/checkpoints/feat${EPOCHS}.pth"
)
declare -A KGPU=( [pretrained]=2 [last_layer]=4 [full]=5 [lora]=7 )

run_method() {
    local m="$1" gpu="${KGPU[$1]}" W="${BW[$1]}"
    [ -f "$W" ] || { echo "[SKIP $m] missing $W"; return; }
    local AE="$AE_DIR/${m}.pth" OUT="$OUT_ROOT/${m}"
    echo "[$m] GPU=$gpu  $(date +%H:%M)"
    DDAD_FE_WEIGHTS="$W" CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n ood python \
        all_categories_embedder/train_pixel_ae_interp.py --weights "$W" --data_dir "$DATA" \
        --save_model "$AE" --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 2 \
        --max_images 500 --ae_loss mse_cos > "$LOG/ae_${m}.log" 2>&1 || { echo "[FAIL ae $m]"; return; }
    DDAD_FE_WEIGHTS="$W" CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n ood python \
        all_categories_embedder/evaluate_ddad_fe.py --config "$CONFIG" \
        --category xray --subcategory scissors --results_root ./results_patches_ddad_native \
        --gt_root "$GT" --skip_sampling --autoencoder_path "$AE" \
        --n_pca 3 --bins_pca 32 --bins_rgb 32 --superpixel_target_size 10 --smooth_sigma 1 \
        --max_reconstructions 40 --results_suffix _1 --output_dir "$OUT" \
        > "$LOG/eval_${m}.log" 2>&1 && echo "[$m] DONE $(date +%H:%M)" || echo "[FAIL eval $m]"
}
for m in pretrained last_layer full lora; do run_method "$m" & done
wait
echo "==== xray_finetune_mine ALL DONE  $(date) ===="
