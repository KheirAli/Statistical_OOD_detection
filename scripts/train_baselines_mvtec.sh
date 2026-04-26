#!/usr/bin/env bash
# Train SimpleNet and CutPaste on every standard MVTec-AD category.
#
# What this does:
#   - SimpleNet: 14 categories (skips `cable` — already trained at
#     /data/akheirandish3/SimpleNet/results/MVTecAD_Results/.../mvtec_cable/ckpt.pth)
#   - CutPaste:  15 categories (none currently trained)
#
# Distribution: round-robin across the GPU slots in `GPUS` (default: 6
# free GPUs at the time of writing). Each slot runs jobs sequentially
# in waves. Total wall clock ≈ 4 hrs for the full sweep at 6 GPUs.
#
# Outputs land outside this repo, on the cluster:
#   SimpleNet: /data/akheirandish3/SimpleNet/results/MVTecAD_Results_full/simplenet_mvtec/run/models/0/mvtec_<class>/ckpt.pth
#   CutPaste:  /data/akheirandish3/cutpaste_models_full/model-<class>-<date>.tch
#
# Per-job stdout+stderr → ./logs/train_<baseline>_<class>.log.
#
# Usage:
#   bash scripts/train_baselines_mvtec.sh                 # train both
#   ONLY=simplenet bash scripts/train_baselines_mvtec.sh  # SimpleNet only
#   ONLY=cutpaste  bash scripts/train_baselines_mvtec.sh  # CutPaste only
#   GPUS="4,5"     bash scripts/train_baselines_mvtec.sh  # restrict GPUs
#   SKIP_EXISTING=1 ...                                   # skip jobs whose ckpt already exists
#
# Designed to be safely re-runnable: if a ckpt for a class already
# exists, set SKIP_EXISTING=1 to skip that class.

set -euo pipefail

# ── config ─────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SIMPLENET_REPO="${SIMPLENET_REPO:-/home/rohan/ood/baseline-algos-clone/SimpleNet}"
CUTPASTE_REPO="${CUTPASTE_REPO:-/home/rohan/ood/baseline-algos-clone/pytorch-cutpaste}"
MVTEC_DATA="${MVTEC_DATA:-/data/akheirandish3/mvtec_ad}"

SIMPLENET_OUT="${SIMPLENET_OUT:-/data2/rohan/baseline_ckpts/simplenet_mvtec_full}"
CUTPASTE_OUT="${CUTPASTE_OUT:-/data2/rohan/baseline_ckpts/cutpaste_mvtec_full}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "$LOG_DIR"

# Which classes? Standard MVTec-AD has 15.
ALL_CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut
             pill screw tile toothbrush transistor wood zipper)
# SimpleNet skips cable (we already have that one).
SIMPLENET_CLASSES=(bottle capsule carpet grid hazelnut leather metal_nut
                   pill screw tile toothbrush transistor wood zipper)

GPUS_RAW="${GPUS:-4,5,6,1,3,2}"
IFS=',' read -ra GPUS_ARR <<<"$GPUS_RAW"
N_GPUS=${#GPUS_ARR[@]}

ONLY="${ONLY:-both}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# ── per-baseline launchers ─────────────────────────────────────────────
launch_simplenet() {
  # stdout = bare PID (consumed by caller); diagnostics go to stderr.
  local cls="$1" gpu="$2" log="$3"
  local ckpt="${SIMPLENET_OUT}/simplenet_mvtec/run/models/0/mvtec_${cls}/ckpt.pth"
  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$ckpt" ]; then
    echo "[skip] simplenet/$cls — ckpt exists at $ckpt" >&2
    return 0
  fi
  echo "[start] simplenet/$cls on GPU $gpu → $log" >&2
  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup python "${REPO_ROOT}/tools/run_simplenet_train.py" \
      --gpu 0 --seed 0 \
      --log_group simplenet_mvtec --log_project "$(basename "$SIMPLENET_OUT")" \
      --results_path "$(dirname "$SIMPLENET_OUT")" --run_name run \
      net -b wideresnet50 -le layer2 -le layer3 \
      --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
      --patchsize 3 --meta_epochs 40 --embedding_size 256 \
      --gan_epochs 4 --noise_std 0.015 --dsc_hidden 1024 \
      --dsc_layers 2 --dsc_margin .5 --pre_proj 1 \
      dataset --batch_size 8 --resize 329 --imagesize 288 \
      -d "$cls" mvtec "$MVTEC_DATA" \
      >"$log" 2>&1 &
  echo $!
}

launch_cutpaste() {
  # stdout = bare PID (consumed by caller); diagnostics go to stderr.
  local cls="$1" gpu="$2" log="$3"
  # CutPaste model filename includes a date stamp; the existence test
  # globs for any pre-existing model for this class.
  if [ "$SKIP_EXISTING" = "1" ] && compgen -G "${CUTPASTE_OUT}/model-${cls}-*.tch" >/dev/null; then
    echo "[skip] cutpaste/$cls — model exists" >&2
    return 0
  fi
  echo "[start] cutpaste/$cls on GPU $gpu → $log" >&2
  mkdir -p "$CUTPASTE_OUT"
  # CutPaste hard-codes Data/ as the dataset root; symlink it into the repo
  # if not already present.
  if [ ! -e "${CUTPASTE_REPO}/Data" ]; then
    ln -s "$MVTEC_DATA" "${CUTPASTE_REPO}/Data"
  fi
  ( cd "$CUTPASTE_REPO" && CUDA_VISIBLE_DEVICES="$gpu" \
      nohup python run_training.py \
        --model_dir "$CUTPASTE_OUT" \
        --type "$cls" \
        --epochs 256 \
        --variant 3way --head_layer 2 \
        --cuda 1 \
        >"$log" 2>&1 &
    echo $! )
}

# ── build job queue: one entry per (baseline, class) ─────────────────
JOBS=()  # each element: "baseline:class"
if [ "$ONLY" = "simplenet" ] || [ "$ONLY" = "both" ]; then
  for c in "${SIMPLENET_CLASSES[@]}"; do JOBS+=("simplenet:$c"); done
fi
if [ "$ONLY" = "cutpaste" ] || [ "$ONLY" = "both" ]; then
  for c in "${ALL_CLASSES[@]}"; do JOBS+=("cutpaste:$c"); done
fi

echo "queueing ${#JOBS[@]} jobs across ${N_GPUS} GPUs (${GPUS_RAW})"
echo "  SIMPLENET_OUT=$SIMPLENET_OUT"
echo "  CUTPASTE_OUT=$CUTPASTE_OUT"
echo

# ── round-robin dispatch in waves ────────────────────────────────────
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  PIDS=()
  for ((j=0; j<N_GPUS && i+j<${#JOBS[@]}; j++)); do
    job="${JOBS[$((i+j))]}"
    baseline="${job%:*}"
    cls="${job#*:}"
    gpu="${GPUS_ARR[$j]}"
    log="${LOG_DIR}/train_${baseline}_${cls}.log"
    case "$baseline" in
      simplenet) pid=$(launch_simplenet "$cls" "$gpu" "$log") ;;
      cutpaste)  pid=$(launch_cutpaste  "$cls" "$gpu" "$log") ;;
    esac
    [ -n "${pid:-}" ] && PIDS+=("$pid")
  done
  echo "wave: waiting on PIDs ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid" || echo "[warn] pid $pid exited non-zero"; done
  i=$((i+N_GPUS))
done

echo
echo "all jobs done."
echo "  SimpleNet ckpts → ${SIMPLENET_OUT}/simplenet_mvtec/run/models/0/mvtec_<class>/ckpt.pth"
echo "  CutPaste models → ${CUTPASTE_OUT}/model-<class>-<date>.tch"
