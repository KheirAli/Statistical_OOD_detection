#!/bin/bash
# Sigma sweep: additive-noise DPS with DDAD UNet at varying sigma.
# Runs sequentially on GPU 5. Each sigma: sample 20 recons × 11 cable images,
# then eval with typical_set PMF (with PCA + no-PCA).
#
# Usage: screen -dmS sigma_sweep bash run_sigma_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6

mkdir -p logs
LOG="$(pwd)/logs/sigma_sweep.log"
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

SAMPLES_IDS="000 001 002 003 004 005 006 007 008 009 010"
SAMPLES_NAMES="samples_000 samples_001 samples_002 samples_003 samples_004 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"
CKPT="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000"
IMAGE_DIR="/data/akheirandish3/mvtec_ad/cable/test/combined"

# Sigmas to sweep (0.1 already done but we re-run for consistency)
SIGMAS="0.15 0.2 0.25 0.3 0.35 0.4"

log "=========================================="
log "Sigma sweep — cable — $(date)"
log "  sigmas: $SIGMAS"
log "  GPU: $CUDA_VISIBLE_DEVICES"
log "=========================================="

for SIGMA in $SIGMAS; do
    SIGMA_TAG=$(echo $SIGMA | tr '.' 'p')  # 0.1 -> 0p1

    RECON_DIR="./results_sigma_sweep/sigma_${SIGMA_TAG}/recons"
    EVAL_PMF="./results_sigma_sweep/sigma_${SIGMA_TAG}/eval_pmf"
    EVAL_PMF_NOPCA="./results_sigma_sweep/sigma_${SIGMA_TAG}/eval_pmf_nopca"

    log ""
    log "────────────────────────────────────────"
    log "sigma=${SIGMA} (tag=${SIGMA_TAG})"
    log "────────────────────────────────────────"

    # ── Phase 1: DPS sampling ──
    log "  Phase 1: sampling (sigma=${SIGMA}, scale=0.5, 20 seeds)"
    python -u tools/run_ddad_dps_sampling.py \
        --ckpt "$CKPT" \
        --image_dir "$IMAGE_DIR" \
        --samples $SAMPLES_IDS \
        --sigma "$SIGMA" --scale 0.5 --num_seeds 20 --skip 25 \
        --out_root "$RECON_DIR" \
        --test_origin "dps_sigma_${SIGMA_TAG}" \
        2>&1 | tee -a "$LOG"
    log "  Phase 1 exit=${PIPESTATUS[0]}"

    # ── Build per-sigma eval config on the fly ──
    CFG="./results_sigma_sweep/sigma_${SIGMA_TAG}/eval_config.yaml"
    cat > "$CFG" <<YAML
sampling:
  enabled: false
  num_patches: 1
  model: {config: configs/model_config.yaml, checkpoint: null}
  diffusion: {sampler: ddpm, steps: 1000, noise_schedule: linear, model_mean_type: epsilon, model_var_type: learned_range, timestep_respacing: "1000"}
  conditioning: {method: ps, scale: 0.5}
  measurement:
    operator: inpainting
    noise: {type: gaussian, sigma: ${SIGMA}}
    mask: {type: refined_box, image_size: 256, mask_len_range: [128, 129], mask_prob: 0.9}
  patch_size: 64
  gpu_ids: [0]
  seed: 0

data:
  image_dir: ${IMAGE_DIR}
  figures_dir: ./figures
  results_dir: ${RECON_DIR}
  sample_name: samples_000
  test_origin: dps_sigma_${SIGMA_TAG}
  bottom_suffix: "4"
  gt_mask:
    path: /data/akheirandish3/mvtec_ad/cable/ground_truth/combined/{sample}_mask.png
    downsample_factor: 4

embeddings:
  backbone: resnet18
  layers: [layer1, layer2, layer3]
  use_patch_context: true
  patchify_size: 3
  proj_dim_per_layer: null
  device: cuda

pca:
  n_components: 5

superpixels:
  var_threshold: 0.0
  min_pixels: 20
  max_sub: 6
  max_depth: 4
  compactness: 12.0
  alpha_grad: 10.0
  target_size: 10

scoring:
  bins_rgb: 32
  bins_pca: 16
  smooth_sigma: 0.1
  min_pixels: 2
  use_label_as_target: true
  eps: 1.0e-12

eval:
  delta_smooth_sigmas: [null, 5.0]
  sp_anomaly_threshold: 0.5
  save_plots: false
  output_dir: ${EVAL_PMF}

baselines:
  enabled: false
  simplenet: {enabled: false, repo_dir: ".", results_dir: ".", category: cable}
  ddad: {enabled: false, repo_dir: ".", config: ".", category: cable, checkpoint_epoch: 2000}
YAML

    # ── Phase 2: eval with PCA ──
    log "  Phase 2: typical_set with PCA"
    python -u evaluate.py \
        --config "$CFG" \
        --skip_sampling --no_plots --scorer typical_set \
        --n_pca 5 --bins_pca 16 \
        --output_dir "$EVAL_PMF" \
        --sample_names $SAMPLES_NAMES \
        2>&1 | tee -a "$LOG"
    log "  Phase 2 exit=${PIPESTATUS[0]}"

    # ── Phase 3: eval without PCA ──
    log "  Phase 3: typical_set no PCA"
    python -u evaluate.py \
        --config "$CFG" \
        --skip_sampling --no_plots --scorer typical_set \
        --bins_pca 1 \
        --output_dir "$EVAL_PMF_NOPCA" \
        --sample_names $SAMPLES_NAMES \
        2>&1 | tee -a "$LOG"
    log "  Phase 3 exit=${PIPESTATUS[0]}"

    log "  sigma=${SIGMA} complete"
done

log ""
log "=========================================="
log "Sigma sweep done — $(date)"
log "=========================================="

# ── Summary table ──
log ""
log "SUMMARY (grep from eval JSONs):"
for SIGMA in $SIGMAS; do
    SIGMA_TAG=$(echo $SIGMA | tr '.' 'p')
    PMF_JSON="./results_sigma_sweep/sigma_${SIGMA_TAG}/eval_pmf/sweep_resnet18_pca5_rgb32_pca16.json"
    NOPCA_JSON="./results_sigma_sweep/sigma_${SIGMA_TAG}/eval_pmf_nopca/sweep_resnet18_pca5_rgb32_pca1.json"
    if [ -f "$PMF_JSON" ]; then
        PMF_SP=$(python -c "import json; d=json.load(open('$PMF_JSON')); print(f\"{d['averaged']['raw']['sp_roc_auc']:.4f}\")" 2>/dev/null || echo "N/A")
        PMF_PX=$(python -c "import json; d=json.load(open('$PMF_JSON')); print(f\"{d['averaged']['sigma_5.0']['px_roc_auc']:.4f}\")" 2>/dev/null || echo "N/A")
    else
        PMF_SP="MISSING"; PMF_PX="MISSING"
    fi
    if [ -f "$NOPCA_JSON" ]; then
        NP_SP=$(python -c "import json; d=json.load(open('$NOPCA_JSON')); print(f\"{d['averaged']['raw']['sp_roc_auc']:.4f}\")" 2>/dev/null || echo "N/A")
        NP_PX=$(python -c "import json; d=json.load(open('$NOPCA_JSON')); print(f\"{d['averaged']['sigma_5.0']['px_roc_auc']:.4f}\")" 2>/dev/null || echo "N/A")
    else
        NP_SP="MISSING"; NP_PX="MISSING"
    fi
    log "  sigma=${SIGMA}  PMF+PCA: SP=${PMF_SP} Px(s5)=${PMF_PX}  |  PMF-noPCA: SP=${NP_SP} Px(s5)=${NP_PX}"
done
