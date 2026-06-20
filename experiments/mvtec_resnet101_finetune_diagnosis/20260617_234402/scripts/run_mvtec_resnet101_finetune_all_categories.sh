#!/usr/bin/env bash
# Category-wise launcher for the MVTec ResNet-101 fine-tuning diagnosis.
#
# For each category:
#   1. symlink the shared combined diffusion UNet as <unet>/<cat>/1000
#   2. full DDAD DA fine-tuning of ResNet-101 (feat0..featN, every epoch)
#   3. LoRA fine-tuning (priority subset of rank x training-time)
#   4. cached-reconstruction evaluation: checkpoint sweep + interpolation + LoRA
#   5. qualitative grid
# Each category is independent -> safe to interrupt (partial results kept).
#
# Env knobs:
#   GPU=5                 CUDA device
#   CATS="bottle grid"    subset (default: all 15)
#   DA_EPOCHS=8           full FT epochs (LoRA fracs derived from this)
#   DO_LORA=1 DO_INTERP=1 DO_QUAL=1
#   NO_PRO=0              set 1 to skip slow PRO during eval
#   LIMIT=                limit #test images (debug)
set -uo pipefail

EXP_REL="experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402"
REPO="/data/akherandish3/Statistical_OOD_detection"
EXP="$REPO/$EXP_REL"
ENVNAME="ddad_env"
COMBINED_UNET="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/combined/1000"
CKPT_ROOT="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec"
# Per-category diffusion UNet checkpoints ("category load_chp"). Only categories
# with a real per-category UNet yield a VALID DDAD detector. Others fall back to
# the combined UNet (documented as not valid — see reports/).
declare -A PERCAT_UNET=( ["cable"]="3000" )
declare -A TEST_EXCLUDE=( ["cable"]="combined" )   # drop curated/duplicate test subdir
SCRIPTS="$EXP/scripts"
CACHE="$EXP/cache"
UNET_BASE="$EXP/unet_ckpts"

GPU="${GPU:-5}"
DA_EPOCHS="${DA_EPOCHS:-8}"
DO_LORA="${DO_LORA:-1}"
DO_INTERP="${DO_INTERP:-1}"
DO_QUAL="${DO_QUAL:-1}"
NO_PRO="${NO_PRO:-0}"
LIMIT="${LIMIT:-}"
ALL_CATS="carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper"
CATS="${CATS:-$ALL_CATS}"

# LoRA priority subset: "rank:epochs"
HALF=$(( DA_EPOCHS / 2 ))
LORA_RUNS=("r4_50:4:$HALF" "r8_50:8:$HALF" "r8_100:8:$DA_EPOCHS")

mkdir -p "$CACHE" "$UNET_BASE"
RUN_LOG="$EXP/logs/launcher_$(date +%Y%m%d_%H%M%S).log"
echo "Launcher start $(date)  GPU=$GPU  DA_EPOCHS=$DA_EPOCHS  CATS=$CATS" | tee -a "$RUN_LOG"

run() { echo "+ $*" | tee -a "$RUN_LOG"; CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n "$ENVNAME" "$@"; }

for CAT in $CATS; do
    echo "======== $CAT ========" | tee -a "$RUN_LOG"
    CDIR="$EXP/$CAT"
    mkdir -p "$CDIR"/{checkpoints,logs,metrics,figures,qualitative}
    # 1. choose UNet: per-category if available, else combined. Use a fixed
    #    symlink name (load_chp) so feat*/cache paths are stable.
    mkdir -p "$UNET_BASE/$CAT"
    if [ -n "${PERCAT_UNET[$CAT]:-}" ] && [ -e "$CKPT_ROOT/$CAT/${PERCAT_UNET[$CAT]}" ]; then
        SRC="$CKPT_ROOT/$CAT/${PERCAT_UNET[$CAT]}"; TAG="${PERCAT_UNET[$CAT]}"; VALID="per-category(valid)"
    else
        SRC="$COMBINED_UNET"; TAG="1000"; VALID="combined(NOT-valid)"
    fi
    UNET_LINK="$UNET_BASE/$CAT/$TAG"
    ln -sf "$SRC" "$UNET_LINK"
    UNET="$UNET_LINK"
    echo "  UNet: $SRC  [$VALID]" | tee -a "$RUN_LOG"

    EXCL_ARG=""; [ -n "${TEST_EXCLUDE[$CAT]:-}" ] && EXCL_ARG="--exclude_test_subdirs ${TEST_EXCLUDE[$CAT]}"
    LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
    PRO_ARG=""; [ "$NO_PRO" = "1" ] && PRO_ARG="--no_pro"

    # 2. full DA fine-tune (skip if already done)
    if [ ! -f "$CDIR/checkpoints/feat${DA_EPOCHS}.pth" ]; then
        run python "$SCRIPTS/ddad_da_finetune.py" \
            --category "$CAT" --unet_ckpt "$UNET" \
            --out_dir "$CDIR/checkpoints" --da_epochs "$DA_EPOCHS" \
            > "$CDIR/logs/train.log" 2>&1
    else
        echo "  [skip] full FT already done" | tee -a "$RUN_LOG"
    fi

    # 3. LoRA runs
    LORA_EVAL_ARGS=()
    if [ "$DO_LORA" = "1" ]; then
        for ENTRY in "${LORA_RUNS[@]}"; do
            NAME="${ENTRY%%:*}"; REST="${ENTRY#*:}"; RANK="${REST%%:*}"; EPS="${REST##*:}"
            LDIR="$CDIR/checkpoints/lora/$NAME"
            if [ ! -f "$LDIR/feat${EPS}.pth" ]; then
                run python "$SCRIPTS/ddad_da_finetune.py" \
                    --category "$CAT" --unet_ckpt "$UNET" \
                    --out_dir "$LDIR" --da_epochs "$EPS" --lora_rank "$RANK" \
                    > "$CDIR/logs/lora_${NAME}.log" 2>&1
            else
                echo "  [skip] lora $NAME already done" | tee -a "$RUN_LOG"
            fi
            LORA_EVAL_ARGS+=("$NAME:$LDIR")
        done
    fi

    # 4. evaluation (checkpoint sweep + interpolation + lora)
    INTERP_ARG=""; [ "$DO_INTERP" = "1" ] || INTERP_ARG="--no_interp"
    LORA_ARG=""
    [ ${#LORA_EVAL_ARGS[@]} -gt 0 ] && LORA_ARG="--lora_dirs ${LORA_EVAL_ARGS[*]}"
    run python "$SCRIPTS/evaluate_all_checkpoints.py" \
        --category "$CAT" --unet_ckpt "$UNET" \
        --feat_dir "$CDIR/checkpoints" --out_dir "$CDIR" --cache_dir "$CACHE" \
        $INTERP_ARG $PRO_ARG $LIMIT_ARG $EXCL_ARG $LORA_ARG \
        > "$CDIR/logs/eval.log" 2>&1
    echo "  eval done -> $CDIR/metrics" | tee -a "$RUN_LOG"

    # 5. qualitative
    if [ "$DO_QUAL" = "1" ]; then
        run python "$SCRIPTS/make_qualitative.py" \
            --category "$CAT" --exp_dir "$EXP" --feat_dir "$CDIR/checkpoints" \
            --unet_ckpt "$UNET" --cache_dir "$CACHE" $EXCL_ARG \
            > "$CDIR/logs/qualitative.log" 2>&1 || echo "  [warn] qualitative failed" | tee -a "$RUN_LOG"
    fi
    echo "  DONE $CAT $(date)" | tee -a "$RUN_LOG"
done

# aggregate plots (cheap, CPU)
conda run --no-capture-output -n "$ENVNAME" python "$SCRIPTS/plot_results.py" --exp_dir "$EXP" \
    >> "$RUN_LOG" 2>&1
echo "Launcher done $(date)" | tee -a "$RUN_LOG"
