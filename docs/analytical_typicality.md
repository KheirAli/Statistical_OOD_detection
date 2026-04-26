# Analytical Typicality: Single-Pass DOOD via the Score Function

A research note on collapsing the multi-seed DOOD/PMF pipeline into a
single-forward-pass anomaly detector that's still grounded in typicality
theory. Compute cost: one UNet evaluation per test image, matching the
single-pass baselines (SimpleNet, CutPaste, DRAEM, DDAD heat_map at N=1).

The idea: DOOD's empirical typicality estimator (sample N=20, build a
histogram, measure NLL deviation) can be replaced with an analytical
estimate computed directly from the diffusion model's noise prediction,
because **the UNet's noise prediction is, up to a known constant, the score
function of the noised data distribution**. Score magnitude is a
sufficient statistic for typicality at fixed noise level, which means
sampling is unnecessary.

---

## 1. Motivation

DOOD/PMF (the paper's home method) currently estimates per-superpixel
typicality empirically:

1. Run a posterior sampler (DDAD-native or DPS) `N=20` times to draw
   reconstructions $\hat{x}^{(1)}, \ldots, \hat{x}^{(N)} \sim p_\theta(x \mid y)$.
2. For each superpixel $S$, build a histogram $\hat{p}_S$ over pixel features
   (RGB × PCA channels) across the $N$ reconstructions.
3. Score:
   $$
   T_S = \left| -\frac{1}{N}\sum_{i=1}^N \log \hat{p}_S\!\left(\hat{x}^{(i)}\right)
                 - H\!\left(\hat{p}_S\right) \right|
   $$
   the absolute deviation of the empirical NLL from the histogram's
   entropy. By the asymptotic equipartition property, this is small for
   typical samples and large for atypical ones.

The expensive part is step 1. Each $\hat{x}^{(i)}$ requires running the
reverse diffusion process — at the very least 10–40 UNet evaluations per
sample. Total: $N \cdot 40 \approx 800$ UNet calls per test image.
Single-pass baselines (SimpleNet et al.) need just 1.

**Comparing them on AUROC at their natural budgets is unfair compute-wise**.
Reviewers do notice. We want a version of DOOD that operates at single-pass
cost while remaining methodologically connected to the typicality framework.

---

## 2. Background: diffusion models and the score function

### 2.1 Forward / reverse process

A standard DDPM defines a forward noising process

$$
q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\; \sqrt{\bar{\alpha}_t}\, x_0,\; (1-\bar{\alpha}_t)\, I\right),
\qquad
\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)
$$

with linear $\beta$ schedule and $T = 1000$ steps. Equivalently:

$$
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I).
$$

The neural network $\varepsilon_\theta(x_t, t)$ is trained to predict the
noise component:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{x_0, \varepsilon, t}\!\left[\,\| \varepsilon - \varepsilon_\theta(x_t, t)\|^2\,\right].
$$

This is exactly what our DDAD UNet was trained on (`MVTec/cable/3000`,
`MVTec/faces/2000`, etc.).

### 2.2 The Tweedie / score connection

The score function of the noised distribution $p_t$ is, by Tweedie's formula:

$$
\nabla_x \log p_t(x_t) \;=\; -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}
$$

This is an exact identity at convergence: training the UNet on
$\mathcal{L}_{\text{simple}}$ asymptotically learns the score function of
$p_t$. Empirically, modern DDPM checkpoints satisfy this with low error.

So the UNet output gives us the gradient of the *log-density* at any
$(x_t, t)$, for **free**. One forward pass = one full per-pixel score map
of how the data distribution behaves in the neighborhood of $x_t$.

### 2.3 What "typical" means in the noised distribution

The noised distribution $p_t$ smooths the data distribution by gaussian
convolution with width $\sigma_t = \sqrt{1-\bar{\alpha}_t}$. For
in-distribution clean samples $x_0 \sim p_{\text{data}}$, the noised
$x_t$ has

$$
\| \nabla_x \log p_t(x_t) \|^2 \;=\;
\frac{\| \varepsilon_\theta(x_t, t)\|^2}{1-\bar{\alpha}_t}.
$$

Under perfect denoising, $\varepsilon_\theta(x_t, t) \approx \varepsilon$, so

$$
\mathbb{E}_{\varepsilon}\!\left[\|\varepsilon_\theta(x_t, t)\|^2\right] \;=\;
\mathbb{E}_{\varepsilon}\!\left[\|\varepsilon\|^2\right] \;=\; C \cdot H \cdot W
$$

for $C$ channels, $H \times W$ spatial dimensions, and unit-variance noise.
For 3-channel pixels, **the per-pixel expectation of squared noise prediction
norm is exactly 3** when $x_0$ is in-distribution.

For out-of-distribution pixels, the UNet's prediction *can't* match the
sampled noise — it's been trained only on clean MVTec patches, so when the
clean signal beneath the noise isn't a typical cable / capsule / etc.
patch, $\varepsilon_\theta$ deviates from the true $\varepsilon$. Both
directions of deviation count: pixels that are too "easy" to denoise
(e.g., uniform regions where the model expects more variation) and pixels
that are too "hard" to denoise (e.g., genuinely anomalous content).

---

## 3. From empirical typicality to analytical typicality

The empirical DOOD/PMF score (Section 1) measures how surprising a
collection of reconstructions $\hat{x}^{(1)}, \ldots, \hat{x}^{(N)}$ is
under their own empirical distribution. The analytical alternative
measures how surprising the test image $x_0$ itself is under the model's
**implicit** distribution at noise level $t$, computed from a single noise
prediction.

### 3.1 Analytical typicality at a single noise level

Define the per-pixel **score-magnitude** signal:

$$
s(x_0, t)_i \;=\; \| \varepsilon_\theta(x_t, t)_i \|^2,
\qquad
x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon
$$

with $\varepsilon$ a fixed, deterministic, seed-controlled draw. The
typicality signal at pixel $i$ is

$$
T_i(x_0; t) \;=\; \big| s(x_0, t)_i \;-\; \mu_t \big|
$$

where $\mu_t = \mathbb{E}_{x_0 \sim p_{\text{data}}}\!\left[\,s(x_0, t)\,\right]$
is the expected score-magnitude on **clean training data**, computed once
offline. For unit-variance gaussian noise and a perfectly-trained UNet,
$\mu_t \approx C = 3$ (channel-summed) regardless of $t$ — but for real
checkpoints it's slightly biased and per-pixel non-uniform, so we estimate
it empirically.

This $T_i$ is mathematically the **squared deviation of the predicted
noise from its expected magnitude**, which by AEP is the leading-order
typicality signal in the asymptotic regime.

### 3.2 Why this is the analytical analog of empirical typicality

Multi-seed PMF score:
$$
T_{\text{empirical}}(S) \;=\; \big|\, \mathbb{E}_{i}[\,-\log \hat{p}_S(\hat{x}^{(i)})\,] \;-\; H(\hat{p}_S)\,\big|.
$$

Single-pass analytical score:
$$
T_{\text{analytical}}(x_0; t)_i \;=\; \big|\, s(x_0, t)_i \;-\; \mu_{t,i} \,\big|.
$$

Both measure deviation of an NLL-like quantity from its expected value
under the corresponding null distribution. The empirical version uses the
Monte Carlo draws to estimate the null distribution from samples; the
analytical version uses the score function to compute it directly. As
$N \to \infty$, the empirical estimator should converge to the analytical
one *for the same UNet*, so the two are theoretically connected, not
arbitrary alternatives.

---

## 4. Multi-resolution extension

A single timestep $t^*$ gives a single noise scale. Different anomalies
are visible at different scales:

- **Small $t$** (mild noise, $\bar{\alpha}_t \approx 1$): score is sensitive
  to fine-grained pixel-level deviations (texture defects, small artifacts).
- **Large $t$** (heavy noise, $\bar{\alpha}_t \approx 0$): score sees only
  coarse signal and reflects deviations in global structure (missing
  components, swapped layouts).

Multi-resolution typicality aggregates across a small set of timesteps:

$$
T_i(x_0) \;=\; \sum_{t \in \mathcal{T}} w_t \cdot \big|\, s(x_0, t)_i \;-\; \mu_{t,i} \,\big|
$$

with e.g. $\mathcal{T} = \{50, 100, 250, 500\}$ and equal weights $w_t = 1/|\mathcal{T}|$.

**Cost: $|\mathcal{T}|$ UNet evaluations per test image** — still ~5-10× cheaper
than the 20-seed sampler at $N=20$, since each sampler draw costs ~10-40
UNet evals.

Pixel-level interpretation: a pixel is anomalous if it deviates from
expected score magnitude *at any noise level*. An anomaly that's visible
only at low resolution (mild noise) and one visible only at high
resolution (heavy noise) both contribute to $T$.

---

## 5. Calibration: estimating $\mu_t$

The expected score magnitude $\mu_t$ depends on:
- The category (clean cable patches have different statistics than clean
  faces patches).
- The pixel location (central vs edge regions can differ for
  position-sensitive datasets).
- The noise scale $t$.

Three estimation strategies, in increasing modeling ambition:

1. **Global scalar** $\mu_t \in \mathbb{R}$ (per timestep, per category).
   Compute on a held-out subset of `train/good/`:
   $$
   \mu_t = \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{x_0 \in \mathcal{D}_{\text{cal}}}
   \frac{1}{HW} \sum_i s(x_0, t)_i.
   $$
   Simplest, ignores spatial structure. Most robust to overfitting on the
   calibration set.

2. **Per-pixel $\mu_{t,i} \in \mathbb{R}^{H \times W}$** (per timestep,
   per category). Computed analogously but kept spatially varying. Captures
   position-dependent statistics (e.g., "cable images always have the cable
   running through the middle, so center pixels have lower expected $s$").
   Risk: overfits the calibration set.

3. **Theoretical $\mu_t \approx C$**. By the score-matching loss, a
   well-trained UNet has $\mathbb{E}[\|\varepsilon_\theta - \varepsilon\|^2] \to 0$,
   so $\mathbb{E}[\|\varepsilon_\theta\|^2] \to \mathbb{E}[\|\varepsilon\|^2] = C$
   (for $C$ channels, summed). Use the constant $C = 3$ with no
   calibration. Most theoretically motivated, but assumes the UNet is
   well-trained at the target $t$ — empirically check before relying on it.

The recommended default is **(1) for the simplest version**; ablate against
(2) and (3) to see whether spatial structure or learned bias matters in
practice.

Calibration set size: ~200 clean training images is plenty for a stable
$\mu_t$ estimate on cable. One-time cost: ~30 seconds per category.

---

## 6. Algorithm

### 6.1 Single-timestep variant (DOOD-1)

```
Inputs:
    UNet f_theta              # diffusion UNet checkpoint (cable / faces / ...)
    image x0                  # (3, H, W), float in [-1, 1]
    t_star                    # int, fixed noise timestep (e.g. 250)
    mu_t                      # scalar or (H, W) — expected score magnitude
    seed                      # int, fixed for determinism

Procedure:
    1. eps ~ N(0, I)          # seeded RNG, deterministic per (image, seed)
    2. a_bar = compute_alpha_bar(t_star)
    3. x_t = sqrt(a_bar) * x0 + sqrt(1 - a_bar) * eps
    4. eps_pred = f_theta(x_t, t_star)            # ONE UNet forward pass
    5. s = sum(eps_pred ** 2, axis=channels)      # (H, W) score magnitude
    6. T = abs(s - mu_t)                          # (H, W) typicality map
    return T
```

Cost: 1 UNet forward pass + a few elementwise ops. ~50 ms on a 4090.

### 6.2 Multi-timestep variant (DOOD-K)

```
Inputs:
    UNet f_theta
    image x0
    timesteps T_set = [t_1, ..., t_K]
    mu_t for each t in T_set
    seed

Procedure:
    T_total = zeros(H, W)
    for t in T_set:
        eps ~ N(0, I)                              # fresh seeded draw per t
        a_bar = compute_alpha_bar(t)
        x_t = sqrt(a_bar) * x0 + sqrt(1 - a_bar) * eps
        eps_pred = f_theta(x_t, t)                 # ONE forward pass per t
        s = sum(eps_pred ** 2, axis=channels)
        T_total += abs(s - mu_t)
    return T_total / len(T_set)                    # (H, W) averaged typicality
```

Cost: $K$ UNet forward passes. Recommended $K \in \{4, 5\}$ for
multi-scale coverage at $\sim 5\times$ single-pass cost.

### 6.3 Calibration (one-time, offline)

```
Inputs:
    UNet f_theta
    calibration set D_cal           # ~200 clean images from train/good/
    t (single value or list)

Procedure:
    accum = zeros(H, W)             # for per-pixel mu
    n = 0
    for x0 in D_cal:
        eps ~ N(0, I)               # fresh draw per image
        x_t = sqrt(a_bar(t)) * x0 + sqrt(1 - a_bar(t)) * eps
        eps_pred = f_theta(x_t, t)
        s = sum(eps_pred ** 2, axis=channels)
        accum += s
        n += 1
    mu_t = accum / n                # (H, W) per-pixel expected score
    return mu_t
```

Done once per (UNet, t), cached on disk. Cost: $|D_{\text{cal}}|$ UNet
evals — about 30 seconds on a 4090.

### 6.4 Determinism

For per-image deterministic output (important for reproducibility and
fair comparison across runs), seed the noise draw `eps` from
`hash(image_path + t)`. Same image, same timestep → same $T$ map.
Different runs → bitwise identical numbers.

---

## 7. Implementation skeleton (PyTorch)

```python
import torch
from pathlib import Path

@torch.no_grad()
def calibrate_mu_t(
    unet: torch.nn.Module,
    cal_loader,                 # yields (B, 3, H, W) clean images in [-1,1]
    t_star: int,
    n_calibration: int = 200,
    device: str = "cuda:0",
    seed: int = 0,
) -> torch.Tensor:
    """Estimate per-pixel expected score magnitude on clean training data.

    Returns: (H, W) tensor of E[‖eps_θ(x_t*, t*)‖²] over the calibration set.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a_bar_t = compute_alpha_bar(t_star).to(device)
    accum, count = None, 0
    for x0 in cal_loader:
        x0 = x0.to(device)
        eps = torch.randn(x0.shape, generator=g, device=device)
        x_t = a_bar_t.sqrt() * x0 + (1 - a_bar_t).sqrt() * eps
        t = torch.full((x0.shape[0],), t_star, device=device, dtype=torch.long)
        eps_pred = unet(x_t, t)
        s = (eps_pred ** 2).sum(dim=1)            # (B, H, W)
        accum = s.sum(dim=0) if accum is None else accum + s.sum(dim=0)
        count += x0.shape[0]
        if count >= n_calibration:
            break
    return accum / count                          # (H, W)


@torch.no_grad()
def analytical_typicality_score(
    unet: torch.nn.Module,
    image: torch.Tensor,         # (1, 3, H, W) float in [-1, 1]
    t_star: int,
    mu_t: torch.Tensor,          # (H, W) precomputed
    seed: int = 0,
) -> torch.Tensor:
    """Single-pass analytical typicality at fixed noise level.

    Returns: (H, W) anomaly map. Higher = more anomalous.
    """
    g = torch.Generator(device=image.device).manual_seed(seed)
    eps = torch.randn(image.shape, generator=g, device=image.device)
    a_bar_t = compute_alpha_bar(t_star).to(image.device)
    x_t = a_bar_t.sqrt() * image + (1 - a_bar_t).sqrt() * eps
    t = torch.tensor([t_star], device=image.device, dtype=torch.long)
    eps_pred = unet(x_t, t)                       # ONE forward pass
    s = (eps_pred ** 2).sum(dim=1).squeeze(0)     # (H, W)
    return (s - mu_t).abs()                       # (H, W) anomaly map


@torch.no_grad()
def multi_resolution_typicality(
    unet, image, t_set, mu_t_dict, seed=0,
) -> torch.Tensor:
    """Sum analytical typicality across a fixed set of timesteps.

    t_set:        e.g. [50, 100, 250, 500]
    mu_t_dict:    {t: precomputed mu_t tensor for that t}
    """
    out = torch.zeros(image.shape[2:], device=image.device)
    for t in t_set:
        out = out + analytical_typicality_score(unet, image, t, mu_t_dict[t], seed)
    return out / len(t_set)
```

Total: ~70 lines including the calibration helper. Drops cleanly into
`ood/baselines/dood_analytical.py` next to the multi-seed `MDPSBaseline`
and `SimpleNetBaseline` wrappers, registered via the `Baseline` ABC.
`evaluate.py --scorer baseline` runs against it without changes.

---

## 8. Connection to existing single-pass methods

| Method                    | Single pass | Uses score function? | Connected to typicality? |
|---------------------------|-------------|----------------------|--------------------------|
| DDAD heat_map (N=1)       | ✓           | implicitly (via x̂₀ reconstruction) | no — pixel + feature distance |
| SimpleNet                 | ✓           | no — discriminator on patch features | no |
| CutPaste                  | ✓           | no — KDE on projection features | no |
| **DOOD-1 (analytical)**   | ✓           | yes — Tweedie identity | **yes — direct** |

DDAD heat_map at $N=1$ uses the diffusion model but only through the
single reconstruction $\hat{x}_0$, which it then compares to the input via
pixel + ResNet feature distance. It doesn't use the score function or
typicality structure — it's a hybrid of "reconstruction-based" and
"feature-based" anomaly detection.

DOOD-1 analytical is genuinely new in this list because it computes a
typicality signal *directly from the score function* without ever
materializing a reconstruction. That's the methodological novelty.

---

## 9. Evaluation protocol

### 9.1 Sanity checks (do first)

1. **Empirical–analytical correlation.** On cable test images, compute
   per-pixel $T_{\text{analytical}}(x_0; t^*=250)$ alongside the existing
   DOOD-20 $T_{\text{empirical}}$. Plot scatter. Goal: Pearson $r > 0.6$
   and visually similar anomaly regions. This validates that the analytical
   estimator is in the same family as the empirical one, not a different
   metric in disguise.

2. **Calibration sanity.** Verify that on `train/good/` images,
   $\mathbb{E}[s(x_0, t)] \approx C \cdot H \cdot W = 3 \cdot H \cdot W$
   (approximately, after summing channels). Large deviation → UNet has a
   training bias at this $t$, and per-pixel calibration matters.

3. **Spatial coherence.** $T_{\text{analytical}}$ should produce
   spatially coherent anomaly regions, not pixel-wise noise. Inspect
   anomaly maps visually. If the maps are noisy, gaussian-smooth before
   AUROC computation (we already do this, $\sigma=5$).

### 9.2 Headline experiments (paper)

Cable + faces (current scope), reporting pixel AUROC + SNR with bootstrap
CIs, evaluated identically to the existing DOOD-20:

| Method                       | Cost (UNet evals/img) | Cable | Faces |
|-----------------------------|-----------------------|-------|-------|
| SimpleNet                   | 1                     | x.xx  | x.xx  |
| CutPaste                    | 1                     | x.xx  | x.xx  |
| DDAD heat_map (N=1)         | ~10                   | 0.969 | 0.558 |
| **DOOD-1 (analytical)**     | **1**                 | **?** | **?** |
| **DOOD-5 (multi-res)**      | **5**                 | **?** | **?** |
| MDPS (N=20)                 | ~800                  | 0.977 | 0.659 |
| DOOD (N=20, empirical)      | ~800                  | 0.962 | 0.581 |

The paper's pitch is whichever of the following the data supports:

a) DOOD-1 ≈ DOOD-20 → **multi-seeding is unnecessary; the analytical
estimator is a strict improvement**. Rewrite the paper around this.

b) DOOD-5 ≈ DOOD-20 at 1/4 the cost → **multi-resolution is the right
budget; analytical-1 is a fast approximation**. Paper has both as a
two-axis story.

c) DOOD-20 >> DOOD-1 by >5 pt → **multi-seeding captures distributional
structure that score-magnitude misses**. Analytical version becomes a
useful baseline (still single-pass-fair) but DOOD-20 stays the paper's
main contribution.

All three are interesting outcomes. (c) is somewhat the "least exciting"
but still a clean negative result on a principled question.

### 9.3 Ablations

- **Choice of $t^*$**: sweep $t^* \in \{50, 100, 250, 500, 750, 999\}$. Plot
  AUROC as a function of $t^*$. Expect a unimodal curve with peak near
  $t^* \approx 250$ (matching DDAD-native's choice empirically).
- **Calibration choice**: per-pixel vs global vs theoretical $C=3$.
- **Multi-resolution timestep set**: $|\mathcal{T}| \in \{1, 2, 4, 8\}$. Plot
  AUROC vs cost.
- **Score-norm vs score-difference**: instead of $|s - \mu|$, try
  $\|\varepsilon_\theta - \varepsilon\|^2$ (literal denoising error).
  Theoretically equivalent for unit-noise $\varepsilon$, but worth checking.

---

## 10. Open research questions

1. **Sharper bound on $T_{\text{analytical}}$.** The construction uses
   the score-norm as a typicality proxy, but the actual NLL of $x_0$ under
   $p_\theta$ requires the full ELBO chain. Is there a closed-form
   intermediate that's tighter than score-norm but cheaper than the full
   chain? (Hutchinson's trace trick for the divergence of the score
   field is one candidate.)

2. **Connection to score-based OOD detection literature.** Score-norm-based
   OOD has been explored for image-level detection (Mahmood et al., Liu
   et al.). Their methods are image-level only; we're proposing a per-pixel
   localization version. Worth a literature review section.

3. **Does multi-resolution recover the full DOOD-20 quality?** If
   $|\mathcal{T}| = 8$ closes the gap to DOOD-20, the multi-seed sampler
   has no per-image AUROC advantage and just adds variance over many
   redundant draws. That would be a strong result.

4. **Adversarial robustness.** Empirical PMF aggregates across 20
   stochastic samples — has natural averaging. Analytical typicality is
   single-pass and might be more brittle to small input perturbations
   that move $\varepsilon_\theta$ by chance. Worth measuring.

5. **Distillation as an alternative.** Train a feedforward CNN to predict
   $T_{\text{empirical}}$ from $x_0$ directly. Even cheaper than analytical
   typicality (no UNet forward pass), and might capture richer signal.
   Different methodological story (no theoretical guarantee), but a useful
   compute-quality point on the Pareto frontier.

---

## 11. Risks and gotchas

- **UNet calibration mismatch.** If our UNet was trained on a slightly
  non-standard noise schedule, $\mathbb{E}[\|\varepsilon_\theta\|^2]$ might
  not equal $C$ even on clean data. Per-pixel calibration $\mu_{t,i}$
  handles this, but verify before publishing.
- **Calibration set drift.** $\mu_t$ is estimated from `train/good/`. If
  the test set has different lighting, JPEG compression, etc., the
  baseline expectation shifts and you over-flag clean test images. Mitigate
  with per-image normalization or a held-out subset of test/good for
  calibration.
- **Per-pixel noise is high.** Single-pass score gives a single
  noise-corrupted observation per pixel. The analytical signal is
  high-variance even on clean data. Multi-resolution averaging and
  post-hoc gaussian smoothing ($\sigma=5$, what we already do) mitigate
  this. May still be worse than DOOD-20 on per-image AUROC for visually
  uniform classes (carpet, leather) where the score is genuinely flat.
- **Doesn't handle multi-modal posteriors well.** A class like
  metal_nut has two equally-valid orientations. The score function at a
  rotated-but-clean test image has high norm (the model's score points
  toward the more common orientation), but it's not actually anomalous.
  DOOD-20 captures both modes in the histogram and isn't fooled. DOOD-1
  analytical might be.

---

## 12. Implementation milestones (if we decide to build it)

| Step | Effort | Outcome |
|------|--------|---------|
| 1. Write `ood/baselines/dood_analytical.py` (~100 lines) | 1 hr | Wrapper implementing the algorithm |
| 2. Write `tools/calibrate_dood_mu.py` (~30 lines) | 30 min | Precomputes and caches $\mu_t$ tensors |
| 3. Configs `configs/baselines/dood_analytical_{cable,faces}.yaml` | 15 min | Match the existing wrapper layout |
| 4. Run sanity check on cable (correlation with DOOD-20) | 30 min | Validates the connection |
| 5. Run full sweep cable + faces, all $t^*$ ablations | 2-3 hrs | Numbers for the paper |
| 6. Write up results in `docs/dood_analytical_results.md` | 1 hr | Paper-ready table + plots |

Total: ~5-6 hours of work. Mostly wrappers + sweeps; the math is in this
document.

---

## References (to add)

- Tweedie's formula and score-based generative models: Song & Ermon
  (NeurIPS 2019), Song et al. (ICLR 2021).
- Empirical typicality via AEP: Cover & Thomas, *Elements of Information
  Theory*, Ch. 3.
- Score-norm-based OOD detection: Mahmood et al. (ICLR 2021), Liu et al.
  (NeurIPS 2020) — image-level only.
- DDAD multi-seed and DOOD/PMF: see [docs/results_organized.md](results_organized.md)
  and the project's results consolidation.
