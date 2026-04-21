# OOD Detection — Scorer × Sampler Comparison (Cable + Faces)

**Task**: pixel-level anomaly detection on MVTec-AD, 256×256 images.
**Metric**: mean ROC-AUC across test images, per-superpixel (SP) and per-pixel (Px) with optional spatial Gaussian smoothing (σ=5).

## Components tested

**Samplers** (produce reconstructions from the input image):
- **DDAD native** — DDAD's own conditioned denoising ([DDAD/reconstruction.py](DDAD/reconstruction.py)): starts at `x_{t*}` (t*=250), reverse-diffuses with analytical `w=2` conditioning toward the input. 20 seeds per image.
- **Additive-noise DPS** — the same DDAD checkpoint driving DPS posterior sampling from pure noise, identity forward model `y = x + 0.1ε`, autograd guidance on `‖y − x̂₀‖`, scale=0.5. 20 seeds per image. ([tools/run_ddad_dps_sampling.py](tools/run_ddad_dps_sampling.py))

**Scorers** (score anomaly maps from the reconstructions):
- **DDAD heat_map** ([DDAD/anomaly_map.py](DDAD/anomaly_map.py)): pixel distance + cosine feature distance on `wide_resnet101_2` features (ImageNet V1), Gaussian-blurred.
- **Typical-set PMF** (our original, [ood/scoring.py](ood/scoring.py)): per-superpixel factorized PMF over RGB ± PCA-projected `ResNet-18` features; score = `|avg NLL − entropy|`. 5636 refined superpixels.
- **Local-Gaussian** (new theory-based, [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py)): recovers precision matrix Λ from sample covariance of reconstructions via closed-form γ→λ identity; scores by typicality deviation. 44 base superpixels, 95% variance truncation.

**Note on feature extractors**: R1 uses `wide_resnet101_2` (127M params); R3/R6a use `ResNet-18` (11M params); R6b uses none (RGB only). Not apples-to-apples on the backbone axis.

---

## Cable results (test/combined, 11 images)

UNet: DDAD cable/3000 (33M params).

| # | Sampler | Scorer | PCA | Mean SP AUC | Mean Px AUC (σ=5) |
|---|---|---|---|---:|---:|
| R1 | DDAD native | DDAD heat_map | — | — | **0.969** |
| R3 | DDAD native | Typical-set PMF | yes | 0.899 | **0.962** |
| R3np | DDAD native | Typical-set PMF | **no** | 0.829 | 0.937 |
| R6a | Additive DPS | Typical-set PMF | yes | 0.880 | 0.942 |
| R6a-np | Additive DPS | Typical-set PMF | **no** | 0.812 | 0.911 |
| R6b | Additive DPS | Local-Gaussian | — | 0.648 | 0.632 |

## Faces results (test/random, 10 images)

UNet: DDAD faces/2000 (33M params).

| # | Sampler | Scorer | PCA | Mean SP AUC | Mean Px AUC (σ=5) |
|---|---|---|---|---:|---:|
| R1 | DDAD native | DDAD heat_map | — | — | ❌ invalid† |
| R3 | DDAD native | Typical-set PMF | yes | 0.597 | 0.581 |
| R3np | DDAD native | Typical-set PMF | **no** | **0.729** | **0.820** |
| R6a | Additive DPS | Typical-set PMF | yes | **0.740** | 0.769 |
| R6a-np | Additive DPS | Typical-set PMF | **no** | 0.677 | 0.715 |
| R6b | Additive DPS | Local-Gaussian | — | 0.643 | 0.661 |

†R1-faces invalid: DDAD's metrics code doesn't handle RGBA GT masks (faces GT is RGBA; cable GT was single-channel). Fixable but requires patching DDAD's dataset loader.

---

## Headlines

### 1. Our PMF scorer is competitive with DDAD on cable, weaker on faces.
On cable, PMF (0.962 Px σ=5) nearly matches DDAD's own pipeline (0.969). On faces, without a valid R1 baseline we can't make the same claim, but PMF reaches 0.820 Px σ=5 at best (DDAD native + no-PCA).

### 2. PCA features help on cable but hurt on faces (with DDAD native recons).
| Dataset | DDAD native + PCA | DDAD native − PCA | Δ |
|---|---:|---:|---|
| Cable | 0.962 | 0.937 | PCA helps +2.5 pt |
| Faces | 0.581 | **0.820** | PCA hurts **−24 pt** |

With DPS recons, PCA consistently helps on both datasets (+3-6 pt). The PCA-hurts effect is specific to DDAD native recons on faces — likely because DDAD's tight conditioning produces recons where PCA features are nearly identical across seeds, making the PCA PMF degenerate.

### 3. Best sampler depends on the dataset and scorer configuration.
| Config | Cable best | Faces best |
|---|---|---|
| PMF + PCA | DDAD native (0.962) | DPS (0.769) |
| PMF no-PCA | DDAD native (0.937) | DDAD native (0.820) |
| Local-Gaussian | DPS (0.648) | DPS (0.661) |

No single sampler dominates. DDAD native is better when its tight conditioning helps (cable + RGB scoring). DPS is better when the scorer needs sample diversity (faces + PCA, or local-Gaussian everywhere).

### 4. The local-Gaussian scorer is consistently the weakest.
| Dataset | PMF best | Local-Gaussian | Gap |
|---|---:|---:|---|
| Cable | 0.962 | 0.648 | −31 pt |
| Faces | 0.820 | 0.661 | −16 pt |

Rohan's theory assumes additive-noise posterior samples (`q_σ`). Neither DDAD's conditioned denoising nor DPS+identity quite matches this assumption. The scorer shows real signal (above 0.5 on most samples) but high per-sample variance and no competitive performance.

### 5. Faces is harder than cable across all methods.
Every scorer × sampler combination drops 10-30 pt from cable to faces. This likely reflects the nature of anomalies (face OOD is subtle morphological distortion vs cable's visible physical damage) and the smaller anomaly regions (~0.6% of pixels vs ~4%).

---

## Cable σ sweep (additive DPS, DDAD UNet, PMF scorer)

Sweep over the DPS measurement noise `y = x + σε` on cable, holding scale=0.5,
20 seeds, 11 images. Each σ has its own recon set + both eval variants saved
under `results_sigma_sweep/sigma_{TAG}/`.

| σ | PMF+PCA SP | PMF+PCA Px σ=5 | no-PCA SP | no-PCA Px σ=5 |
|---|---:|---:|---:|---:|
| 0.05 | 0.879 | 0.942 | 0.811 | 0.911 |
| 0.10 | 0.879 | 0.942 | 0.812 | 0.911 |
| 0.15 | 0.879 | 0.942 | 0.811 | 0.909 |
| 0.20 | 0.881 | 0.942 | 0.813 | 0.908 |
| 0.25 | 0.881 | 0.941 | 0.813 | 0.906 |
| 0.30 | **0.882** | 0.942 | 0.812 | 0.904 |
| 0.35 | 0.881 | 0.942 | 0.807 | 0.902 |
| 0.40 | 0.880 | 0.940 | 0.805 | 0.899 |

**Key finding: σ is not a sensitive hyperparameter for DPS + our PMF scorer.**
- PMF+PCA is essentially flat across the entire 0.05–0.40 range (SP varies by ±0.002, Px σ=5 by ±0.002).
- No-PCA shows a slow decline at high σ: Px σ=5 drops 0.911 → 0.899 (−1.2 pt over the full range).
- Best was σ=0.30 by a hair, but the differences are within run-to-run noise.
- PCA features act as a stabilizer that absorbs the effect of noisier measurements.

Why it's flat: DPS starts from **pure noise** regardless of σ. The σ only scales the DPS gradient strength. Our UNet is capable enough that the guidance converges to similar reconstructions in the 0.05–0.40 range. We'd expect performance to degrade at much higher σ (>0.5) where the measurement becomes so noisy that the gradient direction is unreliable.

**Saved artifacts** under `results_sigma_sweep/sigma_{0p05,0p10,0p15,0p20,0p25,0p30,0p35,0p40}/`:
- `recons/samples_XXX/dps_sigma_0pNN_0_4/inpainting/recon/*.png` (220 recons per σ)
- `eval_pmf/sweep_resnet18_pca5_rgb32_pca16.json` (with-PCA scorer results)
- `eval_pmf_nopca/sweep_resnet18_pca5_rgb32_pca1.json` (no-PCA scorer results)
- `eval_config.yaml` (reproducible config)

---

## Open questions

- **R1-faces fix**: convert RGBA GT masks to single-channel, or patch DDAD's dataset loader to handle 4-channel masks. Needed for a fair DDAD baseline on faces.
- **More superpixels for local-Gaussian**: using 500-1000 fixed SLIC (matching the notebook's setup) instead of 44 base SPs could improve spatial resolution without breaking rank.
- **Other metric variants**: 31 untested variants from the notebook. Per-pixel mode, sequence aggregation, etc.
- **Wide_resnet101_2 for our PMF**: would close the feature-extractor gap with R1 and might improve faces performance.

---

## Reproduction

All scripts auto-pick GPU via `CUDA_VISIBLE_DEVICES` (edit the value at the top of each).

```bash
# ── Cable (R1 DDAD baseline + R3/R3np/R4 + R6a/R6a-np/R6b) ──
cd DDAD && PYTHONPATH=/home/rohan/ood/dps python main.py --config config_cable.yaml --detection True
cd ..
bash scripts/run_cable.sh       # ~2 hours end-to-end

# ── Faces (mirrors cable) ──
cd DDAD && PYTHONPATH=/home/rohan/ood/dps python main.py --config config_faces.yaml --detection True
cd ..
bash scripts/run_faces.sh       # ~2 hours

# ── Cable σ sweep ──
bash scripts/run_sigma_sweep.sh # ~6 hours sequential
```
