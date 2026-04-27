#!/bin/bash
# Full faces experiment: mirrors cable R1/R3/R3np/R6a/R6a-np/R6b.
# Usage: screen -dmS faces bash run_faces_experiments.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate ood
export PYTHONPATH=/home/rohan/ood/dps:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=6

mkdir -p logs figures_faces
LOG=logs/faces_experiments.log
: > "$LOG"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# Faces sample IDs (non-sequential)
FACE_IDS="40 44 49 61 65 80 92 98 107 129"
FACE_SAMPLES="samples_40 samples_44 samples_49 samples_61 samples_65 samples_80 samples_92 samples_98 samples_107 samples_129"

log "=========================================="
log "Faces experiments — $(date)"
log "=========================================="

# ══════════════════════════════════════════════
# R1-faces: DDAD native baseline
# ══════════════════════════════════════════════
log ""
log "R1-faces: DDAD native baseline"
cd /home/rohan/ood/Statistical_OOD_detection/DDAD
python -u main.py --config config_faces.yaml --detection True 2>&1 | tee -a "$LOG"
log "  R1-faces exit=${PIPESTATUS[0]}"
cd "$(dirname "$0")/.."

# ══════════════════════════════════════════════
# Generate superpixel masks for each face image
# ══════════════════════════════════════════════
log ""
log "Generating superpixel masks for faces"
for sid in $FACE_IDS; do
    sp_dir="figures_faces/samples_${sid}"
    if [ ! -f "$sp_dir/mask.png" ]; then
        img="/data/akheirandish3/mvtec_ad/faces/test/random/${sid}.png"
        if [ -f "$img" ]; then
            mkdir -p "$sp_dir"
            python -u super_pixel_generation.py --input_image="$img" --output_dir="$sp_dir" 2>&1 | tee -a "$LOG"
            log "  Generated SP mask for $sid"
        else
            log "  MISSING image: $img"
        fi
    else
        log "  SP mask exists for $sid"
    fi
done

# ══════════════════════════════════════════════
# R2-faces: Generate DDAD-native recons (20 seeds)
# ══════════════════════════════════════════════
log ""
log "R2-faces: DDAD native recons (20 seeds)"
python -u tools/run_ddad_reconstruction.py \
    --ckpt /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/faces/2000 \
    --image_dir /data/akheirandish3/mvtec_ad/faces/test/random \
    --samples $FACE_IDS \
    --num_seeds 40 \
    --out_root ./results_patches_ddad_native_faces \
    2>&1 | tee -a "$LOG"
log "  R2-faces exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R3-faces: typical_set PMF (with PCA) on DDAD-native recons
# ══════════════════════════════════════════════
log ""
log "R3-faces: typical_set with PCA"
python -u evaluate.py \
    --config configs/experiment_ddad_native_faces.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_faces_ddad_native_pmf \
    --sample_names $FACE_SAMPLES  2>&1 | tee -a "$LOG"
log "  R3-faces exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R3np-faces: typical_set PMF (no PCA) on DDAD-native recons
# ══════════════════════════════════════════════
log ""
log "R3np-faces: typical_set no PCA"
python -u evaluate.py \
    --config configs/experiment_ddad_native_faces.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_faces_ddad_native_pmf_nopca \
    --sample_names $FACE_SAMPLES  2>&1 | tee -a "$LOG"
log "  R3np-faces exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R6-faces Phase 1: additive-noise DPS sampling (sigma=0.1, scale=0.5)
# ══════════════════════════════════════════════
log ""
log "R6-faces Phase 1: additive-noise DPS sampling"
python -u tools/run_ddad_dps_sampling.py \
    --ckpt /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/faces/2000 \
    --image_dir /data/akheirandish3/mvtec_ad/faces/test/random \
    --samples $FACE_IDS \
    --sigma 0.1 --scale 0.5 --num_seeds 20 --skip 25 \
    --out_root ./results_patches_ddad_dps_faces \
    2>&1 | tee -a "$LOG"
log "  R6-faces Phase 1 exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R6a-faces: typical_set PMF (with PCA) on DPS recons
# ══════════════════════════════════════════════
log ""
log "R6a-faces: typical_set with PCA on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_faces.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_faces_ddad_dps_pmf \
    --sample_names $FACE_SAMPLES  2>&1 | tee -a "$LOG"
log "  R6a-faces exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R6a-np-faces: typical_set PMF (no PCA) on DPS recons
# ══════════════════════════════════════════════
log ""
log "R6a-np-faces: typical_set no PCA on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_faces.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_faces_ddad_dps_pmf_nopca \
    --sample_names $FACE_SAMPLES  2>&1 | tee -a "$LOG"
log "  R6a-np-faces exit=${PIPESTATUS[0]}"

# ══════════════════════════════════════════════
# R6b-faces: local_gaussian on DPS recons (sigma=0.1, base SP mask)
# ══════════════════════════════════════════════
log ""
log "R6b-faces: local_gaussian on DPS recons"
python -u evaluate.py \
    --config configs/experiment_ddad_dps_faces.yaml \
    --skip_sampling --no_plots --scorer local_gaussian \
    --sigma_rohan 0.1 \
    --output_dir ./results_eval_faces_ddad_dps_lg \
    --sample_names $FACE_SAMPLES  2>&1 | tee -a "$LOG"
log "  R6b-faces exit=${PIPESTATUS[0]}"

log ""
log "=========================================="
log "All faces experiments done — $(date)"
log "=========================================="
