# Sampling: DPS and DDAD-style Denoising

Two sampling modes for reconstructing a clean image from Gaussian noise, implemented in `samplers.py` and driven by `run_sampling.py`.

## Problem Setup

Given a clean image `x` and noise level `sigma`, both modes aim to recover `x`. They differ in how the reverse diffusion process is initialized and guided.

## Mode 1: DPS (Diffusion Posterior Sampling)

**Reference:** Chung et al., "Diffusion Posterior Sampling for General Noisy Inverse Problems", ICLR 2023.

**Setup:** A noisy measurement is created as `y = x + sigma * n` (identity forward model, no signal scaling).

**Algorithm:** The reverse process starts from pure Gaussian noise `x_T ~ N(0, I)`. At each reverse step:

1. Predict noise: `eps = unet(x_t, t)` (with gradient tracking)
2. Tweedie estimate: `x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)`
3. Compute guidance gradient: `grad = d/d(x_t) ||y - x_0_hat||`
4. DDIM reverse step: `x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_hat + c1 * noise + c2 * eps`
5. DPS correction: `x_{t-1} -= scale * grad`

The `scale` parameter controls how strongly the sample is steered toward consistency with `y`.

## Mode 2: DDAD (Conditioned Denoising)

**Reference:** Mousakhan et al., "Anomaly Detection with Conditioned Denoising Diffusion Models", WACV 2024.

**Setup:** The starting timestep `t*` is computed so that the diffusion noise level matches `sigma`:

```
find t* such that sqrt(1 - alpha_bar_{t*}) ≈ sigma
```

Then `x_{t*}` is constructed via the proper forward process:

```
x_{t*} = sqrt(alpha_bar_{t*}) * x_clean + sqrt(1 - alpha_bar_{t*}) * noise
```

**Algorithm:** The reverse process starts from `x_{t*}` and runs with DDAD conditioning (no gradients). At each reverse step:

1. Predict noise: `eps = unet(x_t, t)`
2. Stiffened prediction: `y_t = sqrt(alpha_bar_t) * x_clean + sqrt(1 - alpha_bar_t) * eps`
3. Conditioned noise: `eps_hat = eps - sqrt(1 - alpha_bar_t) * w * (y_t - x_t)`
4. Predict x0: `x_0_t = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)`
5. DDIM reverse step using `eps_hat` and `x_0_t`

The `w` parameter controls conditioning strength (how much the reconstruction is pulled toward `x_clean`).

## Key Differences

| | Mode 1 (DPS) | Mode 2 (DDAD) |
|---|---|---|
| Initialization | Pure noise `x_T ~ N(0, I)` at `t=T-1` | Forward-process `x_clean` to `x_{t*}` |
| Guidance | Autograd through `\|\|y - x_0_hat\|\|` | Analytical conditioning via `w` |
| Requires grad | Yes | No |
| Input to sampler | Noisy `y` (doesn't need `x_clean`) | Clean `x_clean` (constructs `x_{t*}` internally) |
| Starting timestep | `t=T-1` (full trajectory) | Computed from `sigma` |

## Usage

```bash
# Mode 1: DPS
python run_sampling.py --image path/to/image.png --sigma 0.1 --mode 1 --scale 0.5

# Mode 2: DDAD
python run_sampling.py --image path/to/image.png --sigma 0.1 --mode 2 --w 2
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--image` | (required) | Path to clean input image |
| `--sigma` | `0.1` | Gaussian noise std in [-1, 1] image scale |
| `--mode` | (required) | `1` for DPS, `2` for DDAD |
| `--scale` | `0.5` | DPS guidance step size (mode 1 only) |
| `--w` | `2.0` | DDAD conditioning strength (mode 2 only) |
| `--save_dir` | `results_dps` | Output directory |
| `--cfg` | `config.yaml` | Path to config file |
| `--gpu` | `0` | GPU index |

### Outputs

Saved to `--save_dir`:
- `clean.png` — original image
- `noisy.png` — noisy measurement `y` (mode 1) or `x_{t*}` (mode 2)
- `recon.png` — reconstructed image
- `comparison.png` — side-by-side of all three

PSNR between `x_clean` and reconstruction is printed to stdout.

## Config

Both modes read the diffusion schedule and sampler settings from `config.yaml`:

```yaml
trajectory_steps: 1000    # total diffusion steps (T)
skip: 25                  # sampling stride (evaluates model every 25th timestep)
eta: 1                    # 1 = DDPM stochastic, 0 = DDIM deterministic
beta_start: 0.0001        # linear beta schedule start
beta_end: 0.02            # linear beta schedule end
```

- **Mode 1** runs the full trajectory (`T=1000`) with stride `skip`, giving `1000/25 = 40` reverse steps.
- **Mode 2** runs from `t*` (computed from `sigma`) with stride `skip`. For small `sigma`, `t*` is small and the reverse process is short.
- `test_trajectoy_steps` is not used by either mode (it is used by the original DDAD detection pipeline).
