#!/usr/bin/env bash
# Redo LoRA with the BatchNorm-freeze fix, then lora-only re-eval.
# Leaves full-FT checkpoint_sweep + interpolation_sweep untouched.
#
#   GPU=3 CATS="cable leather wood" bash scripts/redo_lora.sh
set -uo pipefail
EXP="/data/akherandish3/Statistical_OOD_detection/experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402"
ENVNAME="ddad_env"; SCRIPTS="$EXP/scripts"; CACHE="$EXP/cache"
GPU="${GPU:-3}"; DA_EPOCHS="${DA_EPOCHS:-8}"; HALF=$(( DA_EPOCHS / 2 ))
NO_PRO="${NO_PRO:-0}"
LORA_RUNS=("r4_50:4:$HALF" "r8_50:8:$HALF" "r8_100:8:$DA_EPOCHS")
declare -A TEST_EXCLUDE=( ["cable"]="combined" )
CATS="${CATS:-cable leather pill toothbrush capsule hazelnut wood metal_nut screw transistor carpet tile}"

run() { CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$ENVNAME" "$@"; }

for CAT in $CATS; do
    CDIR="$EXP/$CAT"
    [ -f "$CDIR/checkpoints/feat${DA_EPOCHS}.pth" ] || { echo "[skip] $CAT: full FT not done"; continue; }
    UNET=$(ls "$EXP/unet_ckpts/$CAT/"* 2>/dev/null | head -1)
    [ -z "$UNET" ] && { echo "[skip] $CAT: no unet link"; continue; }
    echo "==== redo LoRA: $CAT (unet=$UNET) ===="
    rm -rf "$CDIR/checkpoints/lora"
    LORA_EVAL_ARGS=()
    for ENTRY in "${LORA_RUNS[@]}"; do
        NAME="${ENTRY%%:*}"; REST="${ENTRY#*:}"; RANK="${REST%%:*}"; EPS="${REST##*:}"
        LDIR="$CDIR/checkpoints/lora/$NAME"
        run python "$SCRIPTS/ddad_da_finetune.py" --category "$CAT" --unet_ckpt "$UNET" \
            --out_dir "$LDIR" --da_epochs "$EPS" --lora_rank "$RANK" \
            > "$CDIR/logs/lora_${NAME}_fixed.log" 2>&1
        LORA_EVAL_ARGS+=("$NAME:$LDIR")
    done
    EXCL_ARG=""; [ -n "${TEST_EXCLUDE[$CAT]:-}" ] && EXCL_ARG="--exclude_test_subdirs ${TEST_EXCLUDE[$CAT]}"
    PRO_ARG=""; [ "$NO_PRO" = "1" ] && PRO_ARG="--no_pro"
    run python "$SCRIPTS/evaluate_all_checkpoints.py" --category "$CAT" --unet_ckpt "$UNET" \
        --feat_dir "$CDIR/checkpoints" --out_dir "$CDIR" --cache_dir "$CACHE" \
        --lora_only $PRO_ARG $EXCL_ARG --lora_dirs "${LORA_EVAL_ARGS[@]}" \
        > "$CDIR/logs/eval_lora_fixed.log" 2>&1
    echo "  done $CAT -> $(tail -1 $CDIR/logs/eval_lora_fixed.log 2>/dev/null)"
done
echo "redo_lora complete for: $CATS"
