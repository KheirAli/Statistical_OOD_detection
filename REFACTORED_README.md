# Refactored Repo — Run Commands Reference

This file documents the main commands for the cleaned-up repo at
`cleanup/push-ready`. Short-form companion to [README.md](README.md).

---

## 0. Environment setup (once)

```bash
# Clone the DPS repo somewhere (only needed for guided_diffusion that DDAD falls back to)
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git /path/to/dps

# Python env
conda create -n ood python=3.10 -y
conda activate ood
pip install -r requirements.txt

# Export paths used by every script
export PYTHONPATH=/path/to/dps:${PYTHONPATH:-}
```

## 1. Tests (sanity check)

```bash
# Fast suite (< 3 s, no data, no GPU) — always run this first
pytest tests/ -v -m "not requires_ckpt and not requires_gpu and not slow"

# Full suite incl. UNet load + one sampling trajectory (~5 s, needs GPU + DDAD ckpt)
CUDA_VISIBLE_DEVICES=0 pytest tests/ -v
```

## 2. Reconstruction (samples → recons)

Reconstructions are the expensive part of the pipeline; generate them once per
(dataset × sampler × σ) combination and then run cheap scoring on top.

```bash
# ── DDAD-native conditioned denoising (starts from x_{t*}, w=2 conditioning) ──
CUDA_VISIBLE_DEVICES=0 python tools/run_ddad_reconstruction.py \
    --ckpt /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000 \
    --image_dir /data/akheirandish3/mvtec_ad/cable/test/combined \
    --samples 000 001 002 003 004 005 006 007 008 009 010 \
    --num_seeds 20 \
    --out_root ./results_patches_ddad_native

# ── Additive-noise DPS (y = x + σε, autograd guidance from pure noise) ──
CUDA_VISIBLE_DEVICES=0 python tools/run_ddad_dps_sampling.py \
    --ckpt /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000 \
    --image_dir /data/akheirandish3/mvtec_ad/cable/test/combined \
    --samples 000 001 002 003 004 005 006 007 008 009 010 \
    --sigma 0.1 --scale 0.5 --num_seeds 20 --skip 25 \
    --out_root ./results_patches_ddad
```

Recons land at `{out_root}/samples_<id>/<test_origin>_0_4/inpainting/recon/<seed>_0_00000.png`.
Labels and (noisy) inputs are stored alongside, and a `sigma.txt` file records
the effective σ for the local-Gaussian scorer to consume.

## 3. Scoring (recons → AUCs)

All three scorers go through the same entry point `evaluate.py`, selected by
`--scorer`. Point `--config` at an experiment YAML for your dataset; pass the
list of sample IDs via `--sample_names`.

```bash
SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 \
         samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

# (a) Typical-set PMF (ours) with PCA features
python evaluate.py --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_native_pmf \
    --sample_names $SAMPLES

# (b) Typical-set PMF — RGB only, no PCA (ablation)
python evaluate.py --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --bins_pca 1 \
    --output_dir ./results_eval_ddad_native_pmf_nopca \
    --sample_names $SAMPLES

# (c) Local-Gaussian (theory-based)
python evaluate.py --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer local_gaussian \
    --sigma_rohan 0.1 \
    --output_dir ./results_eval_ddad_native_lg \
    --sample_names $SAMPLES

# (d) DDAD's own heat_map scorer (full paper baseline)
cd DDAD && python main.py --config config_cable.yaml --detection True
```

Each eval run produces `{output_dir}/sweep_resnet18_pca5_rgb32_pca16.json` with
per-sample AUCs + swept means.

## 4. Full pipelines (one-liners)

```bash
# Cable: DDAD baseline + all 6 scorer × sampler cells (edit CUDA_VISIBLE_DEVICES at top)
bash scripts/run_cable.sh         # ~2 hours end-to-end

# Faces: mirror of cable pipeline
bash scripts/run_faces.sh         # ~2 hours

# σ sweep: 8 sigmas × (sampling + 2 scorers) sequential on one GPU
bash scripts/run_sigma_sweep.sh   # ~6 hours
```

## 5. Running on a new dataset

1. Drop your data into MVTec layout under `/your/path/<category>/{train/good, test/<defect>, ground_truth/<defect>}/*.png`.
2. Train a DDAD UNet: `python DDAD/main.py --config DDAD/config_<cat>.yaml --train True` (or reuse an existing ckpt at your own risk).
3. Write two config files — copy `configs/experiment_ddad_native.yaml` and `configs/experiment_ddad_dps.yaml`, set paths + category.
4. Copy `scripts/run_cable.sh` → `scripts/run_<cat>.sh`, swap sample IDs and config paths.
5. Run.

See [README.md § Run on a new MVTec category](README.md#run-on-a-new-mvtec-category) for the detailed steps.

## 6. Repo reference card

```
.
├── evaluate.py                            # scoring entry (--scorer, --sigma_rohan)
├── super_pixel_generation.py              # called by evaluate.py to produce SP masks
├── ood/                                   # core eval package
│   ├── data.py                            # recon / GT mask loaders
│   ├── superpixels.py                     # recursive SLIC
│   ├── embeddings.py                      # ResNet + PCA
│   ├── scoring.py                         # typical-set PMF (ours)
│   ├── scoring_local_gaussian.py          # local-Gaussian (rohan)
│   ├── metrics.py                         # AUROC / AP
│   └── visualize.py
├── tools/
│   ├── run_ddad_reconstruction.py         # DDAD native recons
│   └── run_ddad_dps_sampling.py           # additive-noise DPS recons
├── scripts/                               # top-level orchestrators
│   ├── run_cable.sh
│   ├── run_faces.sh
│   └── run_sigma_sweep.sh
├── tests/                                 # pytest suite (21 fast + 3 ckpt/GPU)
├── configs/                               # per-experiment YAMLs
├── DDAD/                                  # upstream DDAD code + config_{cable,faces}.yaml
├── DDAD_DPS/                              # alt samplers for the DDAD UNet
├── rohan/                                 # theory derivation + reference notebook
└── docs/EXPERIMENTS.md                    # lab journal (detailed methodology)
```

## 7. Reproduction verification

We verified that the cleaned repo reproduces RESULTS.md R3 (our PMF scorer + PCA
on DDAD-native reconstructions across all 11 cable images):

| run | Mean SP AUC | Mean Px AUC (raw) | Mean Px AUC (σ=5) |
|---|---:|---:|---:|
| Original (pre-cleanup) | **0.8990** | 0.8967 | **0.9621** |
| Cleaned (`cleanup/push-ready` branch) | **0.8990** | 0.8967 | **0.9621** |

Numbers match to 4 decimal places — no pipeline regression. Command used to
reproduce:

```bash
SAMPLES="samples_000 samples_001 samples_002 samples_003 samples_004 \
         samples_005 samples_006 samples_007 samples_008 samples_009 samples_010"

python evaluate.py \
    --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --scorer typical_set \
    --n_pca 5 --bins_pca 16 \
    --output_dir ./results_eval_ddad_native_pmf \
    --sample_names $SAMPLES
```
