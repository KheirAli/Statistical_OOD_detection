# Statistical OOD Detection — Experiments & Results

End-to-end log of what we ran, why, and what we found. Cable category, MVTec-AD,
11 images (samples_000..010 / test/combined). All numbers are single-run, no
seed averaging across runs.

---

## TL;DR

| # | Pipeline | Mean Px AUC (σ=5) | Notes |
|---|---|---:|---|
| **R1** | DDAD native (full paper pipeline) | **0.969** | Anchor. Uses DDAD's conditioned denoising + heat_map scorer. |
| R3 | DDAD recons + **our typical-set PMF scorer** | **0.962** | Within 0.7 pt of the anchor — our scorer is competitive. |
| R4 | DDAD recons + **rohan local-Gaussian scorer** | 0.669 | Near-random (SP AUC 0.541). Theory's sample-distribution assumption is violated by DDAD's conditioned sampler. |

**Headlines:**
1. Our PMF scorer, applied to DDAD's reconstructions, essentially matches DDAD's full pipeline. The sampler, not the scorer, is where most of DDAD's signal lives on this dataset.
2. Rohan's theory-based scorer **does not work on DDAD reconstructions.** The theory assumes samples from `q_σ`; DDAD's sampler produces tightly-clustered conditioned reconstructions that don't satisfy that assumption.

---

## Active code

### Our eval harness ([ood/](ood/))

| file | role |
|---|---|
| [ood/data.py](ood/data.py) | Loads recons, labels, SP masks, GT masks |
| [ood/superpixels.py](ood/superpixels.py) | Recursive SLIC subdivision |
| [ood/embeddings.py](ood/embeddings.py) | ResNet pixel embeddings + PCA |
| [ood/scoring.py](ood/scoring.py) | **Typical-set PMF scorer** (RGB × PCA factorized) |
| [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py) | **Local-Gaussian scorer** (theory-based, ported from the reference notebook) |
| [ood/metrics.py](ood/metrics.py) | AUROC / AP |
| [ood/visualize.py](ood/visualize.py) | Plots |
| [evaluate.py](evaluate.py) | CLI entry, with `--scorer {typical_set, local_gaussian}` dispatch |

> Historical note: the original pipeline included `ood/sampler.py` and
> `sample_batch.py` (DPS + inpainting via `guided_diffusion`). These were
> dropped in the push-ready cleanup — sampling is now handled by the scripts
> in `tools/` directly, and evaluate.py is scoring-only. See earlier git
> history for the removed code.

### DDAD ([DDAD/](DDAD/), pulled from origin/main)

| file | role |
|---|---|
| [DDAD/main.py](DDAD/main.py) | Entry point — `--detection True` |
| [DDAD/ddad.py](DDAD/ddad.py) | Detection orchestrator |
| [DDAD/reconstruction.py](DDAD/reconstruction.py) | **DDAD's native sampler** (conditioned denoising from `x_{t*}`) |
| [DDAD/anomaly_map.py](DDAD/anomaly_map.py) | **DDAD's scorer** — pixel + cosine-feature heat map |
| [DDAD/unet.py](DDAD/unet.py) | DDAD UNet (33M params) |
| [DDAD/feature_extractor.py](DDAD/feature_extractor.py) | `wide_resnet101_2` loader (`feat0` verified = ImageNet V1) |
| [DDAD/dataset.py](DDAD/dataset.py) | MVTec loader — flipped `test/random`→`test/combined` for cable |
| [DDAD/config_cable.yaml](DDAD/config_cable.yaml) | Cable-targeted config (checkpoint_dir, load_chp=3000, use_frozen_fe=True) |

### Alt samplers ([DDAD_DPS/](DDAD_DPS/))

| file | role |
|---|---|
| [DDAD_DPS/samplers.py](DDAD_DPS/samplers.py) | Two samplers for the DDAD UNet: Mode 1 = DPS (from noise, autograd), Mode 2 = DDAD-style |

### Reference material for the theory-based scorer

The derivation of the local-Gaussian scorer, plus the reference notebook it was
ported from, lives outside the tracked repo (`rohan/` is in `.gitignore`). The
production port is [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py).

### Tools

| file | role |
|---|---|
| [tools/run_ddad_reconstruction.py](tools/run_ddad_reconstruction.py) | Generates N DDAD recons per image, saves in eval layout, records effective σ |
| [tools/run_ddad_dps_sampling.py](tools/run_ddad_dps_sampling.py) | Generates N additive-noise DPS recons per image |
| [tools/prepare_xray_dataset.py](tools/prepare_xray_dataset.py) | Converts SIXray → MVTec layout for the X-ray pilot |

### Configs

| file | role |
|---|---|
| [configs/experiment.yaml](configs/experiment.yaml) | Default pipeline config (DPS+inpainting, σ=0.1) |
| [configs/experiment_first_sigma.yaml](configs/experiment_first_sigma.yaml) | Eval against grad student's recons |
| [configs/experiment_reproduce_000.yaml](configs/experiment_reproduce_000.yaml) | Reproduce samples_000 end-to-end with our DPS |
| [configs/experiment_ours_filtered.yaml](configs/experiment_ours_filtered.yaml) | Our recons filtered to the same 7 patches |
| [configs/experiment_ddad_native.yaml](configs/experiment_ddad_native.yaml) | Eval DDAD-native recons (R3, R4) |

---

## Runs

### Phase A — sanity: harness, sampler, scoring (samples_000 only)

Before switching to DDAD, we validated the eval harness against the grad student's
"correct" reconstructions (`first_sigma_batched`) to confirm the refactor is sound.

| run | recons | # | Scorer | SP AUC | Px AUC (σ=5) |
|---|---|---:|---|---:|---:|
| Grad student's recons | `first_sigma`, 7 patches | 28 | Our PMF | 0.957 | **0.982** |
| Our DPS+inpainting | `half_sigma σ=0.1`, 24 patches | 96 | Our PMF | 0.946 | 0.969 |
| Ours filtered to same 7 patches | `half_sigma σ=0.1`, 7 patches | 28 | Our PMF | 0.942 | 0.972 |

Recon pixel-diff (our DPS vs their recons, matched patches):
mean |diff| ≈ **10.5/255** with std 0.09 across patches — highly uniform, suggests
systematic (not random-seed) difference. Either way the eval pipeline is working.

### Phase B — DDAD baseline (E1 = R1)

Ran `python DDAD/main.py --detection True` unmodified on cable.

- **Configuration**: [DDAD/config_cable.yaml](DDAD/config_cable.yaml) — `load_chp=3000`, `w=2`, `v=1`, `test_trajectoy_steps=250`, `use_frozen_fe=True` (feat0 is byte-identical to torchvision's `IMAGENET1K_V1` weights — verified).
- **Command**:
  ```bash
  cd DDAD && PYTHONPATH=/home/rohan/ood/dps python main.py --config config_cable.yaml --detection True
  ```
- **Result**: **Pixel AUROC 96.9%** averaged across 11 cable/combined images. Image AUROC `nan` (test/combined has no "good" images — undefined).
- Caveats: Single run, no seed averaging. DDAD paper reports 98.1% for cable; we're 1.2 pt below, likely because we used `feat0` (ImageNet) not domain-adapted (`feat1..4` don't exist in the checkpoint dir).

### Phase C — three-way scorer A/B/C on identical DDAD recons

**Priority from advisor**: run (1) rohan's theory-based scorer and (2) our previous
PMF scorer on the same model as the DDAD baseline. Approach: generate one set of
DDAD reconstructions, then score three ways.

#### R2 — generate DDAD reconstructions

- **Script**: [tools/run_ddad_reconstruction.py](tools/run_ddad_reconstruction.py)
- **Parameters**: `w=2`, `t*=250`, `skip=25`, N=20 seeds per image, 11 images → **220 reconstructions**
- **Effective σ at t\***: `sqrt(1 − α̅_{250})` = **0.6899** (saved to `results_patches_ddad_native/sigma.txt`)
- **Output**: [results_patches_ddad_native/samples_XXX/ddad_native_0_4/inpainting/recon/](results_patches_ddad_native/)
- **Runtime**: ~15 min total on one GPU
- **Recon sanity** (samples_000 at seed 0): recon mean 99.7 / label 99.7, recon std 44.7 / label 48.3, |label−recon| ≈ 10.7/255, seed variability |recon0−recon1| ≈ 3.5/255. Much less variance than our DPS+inpainting recons (≈ 11/255 variability) — DDAD's sampler is more deterministic, more tightly coupled to the input image.

#### R3 — our typical-set PMF scorer on DDAD recons

- Reuses [ood/scoring.py](ood/scoring.py) via `--scorer typical_set`
- `n_pca=5, bins_rgb=32, bins_pca=16` (same params as Phase A)
- **Per-sample SP AUC (raw)**: 0.887, 0.948, 0.918, 0.904, 0.859, 0.872, 0.878, **0.960**, 0.920, 0.859, 0.885
- **Mean SP AUC 0.899, Mean Px AUC (σ=5) 0.962**
- Versus DDAD anchor 0.969: **−0.7 points**.

#### R4 — rohan local-Gaussian scorer on DDAD recons

- New scorer [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py) via `--scorer local_gaussian`
- σ auto-loaded from `sigma.txt` (0.6899)
- `n_realizations=1000, metric=typicality_unsigned`, RGB-only (no PCA — faithful to theory)
- **Per-sample SP AUC (raw)**: 0.525, 0.618, 0.563, 0.542, 0.570, 0.506, 0.509, 0.501, 0.561, 0.515, 0.542
- **Mean SP AUC 0.541, Mean Px AUC (σ=5) 0.669**
- Near-random across every sample. On samples_007, pixel AUC drops below 0.5 (0.497) — the scorer is momentarily anti-discriminative.

---

## Full three-way summary (11 cable samples averaged)

| Scorer | Mean SP AUC (raw) | Mean Px AUC (raw) | Mean Px AUC (σ=5) |
|---|---:|---:|---:|
| **DDAD heat_map** (R1, the anchor) | — | — | **0.969** |
| **Our typical-set PMF** (R3) | 0.899 | 0.897 | 0.962 |
| **Rohan local-Gaussian** (R4) | 0.541 | 0.537 | 0.669 |

Per-sample detail: [results_eval_ddad_native_pmf/sweep_resnet18_pca5_rgb32_pca16.json](results_eval_ddad_native_pmf/sweep_resnet18_pca5_rgb32_pca16.json)
and [results_eval_ddad_native_lg/sweep_resnet18_pca5_rgb32_pca16.json](results_eval_ddad_native_lg/sweep_resnet18_pca5_rgb32_pca16.json).

---

## What we learned

1. **Our PMF scorer is competitive with DDAD's full pipeline on its own reconstructions.** 0.962 vs 0.969 means the scoring algorithm is essentially a non-factor at this level of recon quality — the bottleneck is the reconstructions themselves. If we can produce DDAD-quality reconstructions with a cheaper / more theoretically grounded sampler, our scorer picks up the same signal.

2. **Rohan's theory-based scorer fails on DDAD reconstructions** — structurally, not by a tuning margin. Reading why:
   - Rohan's algorithm assumes reconstructions are samples from `q_σ = N(μ_denoised, Σ_denoised)`, where `Σ_denoised = (Λ+αI)^-1 + α(Λ+αI)^-2`. It inverts this relation to recover the precision Λ via eigendecomposition of the sample covariance.
   - DDAD's sampler is *not* drawing from `q_σ`. It starts at a fixed `x_{t*}` from the forward process of the test image, conditions the reverse trajectory toward the input via `w=2`, and outputs a reconstruction that stays near the input. Seed-to-seed variability on samples_000 is ~3.5/255 — much too small a sample covariance for the precision recovery to be meaningful.
   - Concretely: `(x* − μ)` under the theory should be near zero for in-distribution pixels and nonzero for anomalies. When DDAD's conditioning forces every seed to land near `x*`, the empirical `μ` learned from the samples ≈ `x*` everywhere, and the deviation signal collapses to noise.

3. **Neither scorer dominates universally.** The pertinent comparison for rohan is on reconstructions that *do* resemble q_σ samples — i.e., our DPS+inpainting recons (seed variability ≈ 11/255) or a true DPS-from-noise trajectory. That test is the natural follow-up.

4. **Small operational findings worth remembering**:
   - `feat0` in the checkpoint dir is byte-equal to torchvision's `IMAGENET1K_V1` for `wide_resnet101_2`. Domain-adapted `feat1..4` do not exist in `/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/`. Using `feat0` is therefore equivalent to using the ImageNet pretrained model.
   - Effective σ at DDAD's `t*=250` is 0.6899, not 0.1. This matters for rohan's algorithm, which is σ-sensitive.
   - `test/combined` has no "good" images → image-level AUROC is undefined. Pixel AUROC is the only well-defined number on this split.

---

## Phase D — R6 corrected: additive-noise DPS + ablations (Apr 15)

### What changed and why

**R6 first attempt (inpainting mask) was wrong.** The original R6 used a center
128×128 inpainting mask (`y = M·x`) with DPS scale=5. But the intended method is
additive-noise DPS (`y = x + σε`, identity forward model, scale=0.5) — this is
what [DDAD_DPS/samplers.py](DDAD_DPS/samplers.py) Mode 1 implements and what the
grad student designed.

**Eigenvalue truncation fix.** [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py)
was using `threshold = 1e-10 × γ_max` to select eigenvectors — retaining all ~999
rank-limited eigenvectors including noise. Changed to **cumulative variance
explained ≥ 95%** (`np.searchsorted(cumvar, 0.95)`). This drops noise eigenvectors
whose gamma→lambda inversion would inject garbage into the precision matrix. Now
parameterized as `variance_explained=0.95`.

**No-PCA ablation added.** Grad student asked: what are the numbers without PCA
features? Running our PMF scorer with `--bins_pca 1` makes the PCA PMF trivially
`[1.0]` → H=0, NLL=0 → score is pure RGB. No code changes needed.

### Runs queued (screen `r6_corrected`)

| Phase | Run | What | Expected output |
|---|---|---|---|
| 0 | R3 no-PCA | Our PMF scorer, bins_pca=1, on DDAD-native recons | `results_eval_ddad_native_pmf_nopca/` |
| 1 | R6 sampling | DDAD UNet + additive-noise DPS, σ=0.1, scale=0.5, 11×20 seeds | `results_patches_ddad/` |
| 2 | R6 PMF | Our PMF on R6 recons, n_pca=5, bins_pca=16 | `results_eval_ddad_dps_pmf/` |
| 3 | R6 PMF no-PCA | Our PMF on R6 recons, bins_pca=1 | `results_eval_ddad_dps_pmf_nopca/` |
| 4 | R6 rohan | Local-Gaussian on R6 recons, σ=0.1 (the true value), 95% var truncation | `results_eval_ddad_dps_lg/` |

### Key differences from old R6

| | Old R6 (wrong) | Corrected R6 |
|---|---|---|
| Forward model | Inpainting (M·x, center 128×128 box) | Identity (y = x + 0.1ε) |
| DPS scale | 5.0 (cranked up for mask-pinning) | 0.5 (standard DPS) |
| σ for rohan | 1.0 (placeholder) | **0.1** (the actual noise level) |
| Eigenvalue truncation | 1e-10 × γ_max (retains ~999 noise eigvecs) | **95% variance explained** (retains ~5-15) |

### Feature extractor note (from grad student)

R1 (DDAD heat_map) uses `wide_resnet101_2` (ImageNet V1, 127M params). R3/R6
(our PMF) use `ResNet-18` (ImageNet, 11M params). These are NOT the same feature
extractor. A fairer R1 vs R3/R6 comparison would use the same backbone. Not done
yet — adding `wide_resnet101_2` to our [ood/embeddings.py](ood/embeddings.py) is
straightforward but expands scope.

### Results (to be filled)

| # | Sampler | Scorer | PCA | Mean SP AUC | Mean Px AUC (σ=5) |
|---|---|---|---|---:|---:|
| R1 | DDAD native | DDAD heat_map | — | — | **0.969** |
| R3 | DDAD native | Typical-set PMF | yes | 0.899 | 0.962 |
| R3np | DDAD native | Typical-set PMF | **no** | TBD | TBD |
| R4 | DDAD native | Local-Gaussian | — | 0.541 | 0.669 |
| R6a | Additive-noise DPS | Typical-set PMF | yes | TBD | TBD |
| R6a-np | Additive-noise DPS | Typical-set PMF | **no** | TBD | TBD |
| R6b | Additive-noise DPS | Local-Gaussian (fixed) | — | TBD | TBD |

---

## Planned but not done

| | rationale |
|---|---|
| **Rohan metric sweep** (32 variants instead of just `typicality_unsigned`) | If R6b looks promising, grid-sweep the other 31 metric combinations from the notebook. |
| **Fair feature-extractor comparison** (wide_resnet101_2 for our PMF scorer) | Isolates the scoring algorithm from the feature-backbone gap between R1 and R3/R6a. |
| **R5**: rohan scorer on our DPS+inpainting recons (guided_diffusion UNet) | Tests rohan on recons from our original pipeline. |

---

## Reproduction commands

```bash
# R1 — DDAD baseline
cd DDAD && PYTHONPATH=/home/rohan/ood/dps CUDA_VISIBLE_DEVICES=0 \
    python main.py --config config_cable.yaml --detection True

# R2 — generate DDAD recons
cd /home/rohan/ood/Statistical_OOD_detection
CUDA_VISIBLE_DEVICES=0 python tools/run_ddad_reconstruction.py --num_seeds 20

# R3 — our PMF on DDAD recons (11 samples)
python evaluate.py --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots --n_pca 5 --bins_pca 16 \
    --scorer typical_set --output_dir ./results_eval_ddad_native_pmf \
    --sample_names samples_000 samples_001 samples_002 samples_003 samples_004 \
                   samples_005 samples_006 samples_007 samples_008 samples_009 samples_010

# R4 — rohan on DDAD recons (same 11 samples)
python evaluate.py --config configs/experiment_ddad_native.yaml \
    --skip_sampling --no_plots \
    --scorer local_gaussian --output_dir ./results_eval_ddad_native_lg \
    --sample_names samples_000 samples_001 samples_002 samples_003 samples_004 \
                   samples_005 samples_006 samples_007 samples_008 samples_009 samples_010
```

Or run [run_three_way.sh](run_three_way.sh) for R3+R4 in one go.
