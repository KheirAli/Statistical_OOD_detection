# Statistical OOD Detection via Diffusion Posterior Sampling

Detect out-of-distribution (OOD) anomalies in images by reconstructing them with diffusion models, then measuring how much the reconstruction deviates from the original using information-theoretic scoring over superpixel regions.

## How It Works

Given a test image:
1. **Segment** it into superpixel regions (SLIC), then recursively subdivide until color-homogeneous
2. **Reconstruct** the image multiple times via diffusion posterior sampling (DPS) with inpainting masks focused on each region
3. **Embed** both the original and all reconstructions using a pretrained ResNet, then project to a shared PCA basis
4. **Score** each superpixel by building a joint RGB+PCA probability distribution from the reconstructions and measuring how far the original falls from the typical set: `delta = |avg_neg_logp - H|`
5. **Evaluate** against ground-truth anomaly masks using AUROC and AP at both superpixel and pixel level

In-distribution regions reconstruct consistently (low delta); anomalous regions produce high delta because the diffusion model hasn't learned to reconstruct them.

## Setup

### 1. Clone this repo and the DPS dependency

```bash
git clone https://github.com/KheirAli/Statistical_OOD_detection.git
cd Statistical_OOD_detection

# Clone the Diffusion Posterior Sampling framework (provides guided_diffusion module)
git clone https://github.com/DPS2022/diffusion-posterior-sampling.git ../dps

# motionblur is needed by DPS but not on PyPI — install from source
git clone https://github.com/LeviBorodenko/motionblur.git /tmp/motionblur
mkdir -p ../dps/motionblur
cp /tmp/motionblur/motionblur.py /tmp/motionblur/__init__.py ../dps/motionblur/
```

### 2. Create the conda environment

```bash
conda create -n ood python=3.10 -y
conda activate ood
pip install numpy torch torchvision scikit-image scipy matplotlib pyyaml pillow tqdm
```

### 3. Get model checkpoints

You need a trained diffusion model checkpoint. Available checkpoints:
- `CT_scan/model050000.pt` — medical CT images
- `mvtec_cable_256/model010000.pt` — MVTec cable anomaly detection
- `ffhq_10m.pt` — face images (FFHQ)

Set the checkpoint path in `configs/model_config.yaml` under `model_path`.

### 4. Prepare test data

Place test images (256x256 PNG) in a directory and update `configs/experiment.yaml`:
```yaml
data:
  image_dir: /path/to/your/test/images
  gt_mask:
    path: /path/to/ground_truth/{sample}_mask.png
```

## Usage

### Full pipeline (sampling + analysis)

```bash
python evaluate.py --config configs/experiment.yaml
```

This runs DPS reconstruction on each superpixel region, then scores and evaluates. With DDIM (default) and 24 patches x 4 measurements, expect ~10 min per image on a single GPU.

### Analysis only (skip sampling)

If you already have DPS reconstruction results in `results_patches/`:

```bash
python evaluate.py --config configs/experiment.yaml --skip_sampling
```

### Other options

```bash
# Headless — just metrics JSON, no plots (for automated sweeps)
python evaluate.py --config configs/experiment.yaml --skip_sampling --no_plots

# Override sample name
python evaluate.py --config configs/experiment.yaml --skip_sampling --sample_name samples_005

# Run with baseline comparisons (SimpleNet, DDAD)
python evaluate.py --config configs/experiment.yaml --skip_sampling --baselines

# Use a specific GPU
python evaluate.py --config configs/experiment.yaml --device cuda:1
```

### Output

Results are saved to `results_eval/{sample_name}/`:

```
results_eval/samples_010/
  results.json        # Metrics + full config snapshot (reproducible)
  delta_map.npy       # Raw anomaly score map
  plots/
    evaluation_raw.png        # ROC curves + GT overlay + heatmap
    evaluation_sigma_5.0.png  # Same with Gaussian-smoothed delta
    delta_map.png             # Delta heatmap + overlay
    superpixels.png           # Refined superpixel boundaries
    comparison.png            # Baseline comparison (if --baselines)
```

## Configuration

Everything is controlled by `configs/experiment.yaml`. Key sections:

### Sampling
```yaml
sampling:
  diffusion:
    sampler: ddim              # ddim (fast, default) | ddpm (slow, original)
    timestep_respacing: "100"  # 100 steps for DDIM vs 1000 for DDPM
  measurement:
    noise:
      sigma: 0.1              # Gaussian noise added to observations
    mask:
      mask_prob: 0.9           # Masking probability for refined_box
  num_measurements: 4          # Reconstructions per patch
  num_patches: 24              # Superpixel regions to sample
```

### Scoring (the algorithmic core)
```yaml
scoring:
  bins_rgb: 32       # RGB quantization bins (higher = finer PMF)
  bins_pca: 8        # PCA quantization bins
  smooth_sigma: 0.1  # Gaussian smoothing on PMF histograms
```

### Superpixels
```yaml
superpixels:
  var_threshold: 0.0   # Max RGB variance before splitting (0 = always split)
  target_size: 10      # Desired region size in pixels
  max_depth: 4         # Recursion depth limit
```

### Embeddings
```yaml
embeddings:
  backbone: resnet18              # resnet18 | resnet50
  layers: [layer1, layer2, layer3]
  use_patch_context: true         # 3x3 local averaging on features
pca:
  n_components: 9
```

## System Design

```
evaluate.py                         CLI entry point
    |
    v
ood/sampler.py    [Stage 0]        Wraps sample_batch.py as subprocess
    |                               Generates temp YAML configs from experiment.yaml
    |                               Parallelizes across GPUs
    v
ood/data.py       [Stage 1]        Loads reconstructions, label image, masks
    |
    v
ood/superpixels.py [Stage 2]       Recursive SLIC subdivision until color-homogeneous
    |
    v
ood/embeddings.py  [Stage 3]       ResNet feature extraction + PCA projection
    |                               ResNetPixelEmbedder: multi-layer concat at full resolution
    |                               PCA basis computed from label image, applied to all
    v
ood/scoring.py     [Stage 4]       Typical set scoring (the algorithmic core)
    |                               Per-superpixel: factorized PMF (RGB * PCA) from recons
    |                               Delta = |avg_neg_logp(label) - H(PMF)|
    v
ood/metrics.py     [Stage 5]       ROC/PR/AUC at superpixel + pixel level
    |
    v
ood/visualize.py   [Stage 6]       Heatmaps, ROC curves, GT overlays
    |
    v
ood/baselines.py   [Stage 7]       Wraps SimpleNet/DDAD for comparison
```

**Design principle:** one file per pipeline stage. To test a new scoring algorithm, edit only `ood/scoring.py`. To try new embeddings, edit only `ood/embeddings.py`. All hyperparameters live in the YAML config.

### DPS Dependency

The sampling stage depends on [diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling) for:
- `guided_diffusion.unet` — UNet diffusion model
- `guided_diffusion.gaussian_diffusion` — DDPM/DDIM samplers
- `guided_diffusion.condition_methods` — Posterior sampling conditioning
- `guided_diffusion.measurements` — Inpainting operator + noise models
- `data.dataloader` — Image dataset loading

The sampler injects this via `PYTHONPATH` automatically (expects `../dps/` relative to this repo).

### Legacy Code

- `New_dataset_clean.ipynb` — the original notebook all `ood/` code was extracted from. Kept as reference.
- `run_with_mask.sh` — legacy batch sampling script. Replaced by `ood/sampler.py`.
- `super_pixel_generation.py` — standalone SLIC mask generation. Still called by the sampler.
- `util/` — utilities used by `sample_batch.py`. Not used by the eval harness.

## Known Issues and Areas for Improvement

### Performance

- **Superpixel subdivision is aggressive.** With `var_threshold: 0.0`, a 38-region mask subdivides to 5000+ regions. The scoring loop then iterates over each, and the ResNet embedding step processes each image sequentially on GPU. For quick iteration, raise `var_threshold` (e.g. 90) and `target_size` (e.g. 150).

- **Embedding is the bottleneck.** `embed_images()` processes one image at a time to avoid GPU OOM. Batching (e.g. 4-8 at a time) would be significantly faster for GPUs with enough memory.

- **PCA SVD on full-resolution features is expensive.** The label image produces a `(65536, 448)` matrix for SVD. Downsampling or using fewer ResNet layers would help.

### Correctness / Robustness

- **`sample_batch.py` processes only 1 image per run** (line 159: `break`). This is by design for the current patch-based workflow, but means the sampler must launch one subprocess per patch. Removing the `break` and batching patches would be more efficient.

- **`box_coords` in `sample_batch.py` are not pixel coordinates.** They are `(superpixel_index, mode_flag, unused, unused)` where mode `4` triggers random probability masking. This is confusing and undocumented in the original code. The eval harness handles it correctly but the interface is fragile.

- **Hardcoded paths throughout `util/img_utils.py` and `super_pixel_generation.py`.** Default `mask_path` and `output_dir` point to `/home/akheirandish3/...`. The eval harness overrides these, but running the legacy scripts directly requires manual path editing.

- **No error handling for failed sampling subprocesses.** If a patch fails (e.g. GPU OOM), the sampler continues and the eval stage gets fewer reconstructions without warning.

### Algorithm

- **The scoring function uses a factorized PMF** (`P_rgb * P_pca`) to avoid the curse of dimensionality, but this independence assumption is strong. Joint modeling or copula-based approaches could capture RGB-PCA correlations.

- **PCA basis is computed only from the label (test) image.** This means the projection is optimized for the test image rather than the training distribution. Computing PCA from the reconstruction ensemble might give a more informative basis.

- **Only one scoring method is implemented** (typical set divergence). The modular design makes it easy to add alternatives in `ood/scoring.py` — e.g., Mahalanobis distance, likelihood ratio, or learned scoring.

- **Gaussian smoothing on the delta map** (`delta_smooth_sigmas`) is applied as a post-hoc step. Spatially-aware scoring that uses superpixel adjacency would be more principled.

### Infrastructure

- **No automated tests.** The pipeline was validated end-to-end manually. Unit tests for `metrics.py`, `scoring.py`, and `superpixels.py` (all pure numpy) would catch regressions when modifying the algorithmic core.

- **Baseline runners (`ood/baselines.py`) parse AUROC from stdout** via regex, which is brittle. A more robust approach would be to import the baseline code directly or have them write results to a standard JSON format.

- **DDIM is the default sampler** (`timestep_respacing: "100"`), which is ~10x faster than DDPM. However, the quality/speed tradeoff hasn't been characterized for OOD detection specifically — DDPM may produce better-calibrated reconstructions.

- **No multi-image evaluation.** The pipeline runs on one sample at a time. A sweep mode that processes a directory of images and aggregates metrics would enable proper dataset-level benchmarking.

## Datasets

Tested with:
- **MVTec AD** — industrial anomaly detection (cable, bottle, etc.)
- **Warped face images** — synthetic OOD with controlled anomalies (tumors, stars, geometric warps)
- **CT scans** — medical imaging anomaly detection

## Acknowledgments

- Diffusion Posterior Sampling: [DPS2022/diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling)
- SimpleNet baseline: [DonaldRR/SimpleNet](https://github.com/DonaldRR/SimpleNet)
- DDAD baseline: anomaly detection via denoising diffusion
