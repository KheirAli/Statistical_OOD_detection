# Running the DDAD Baseline

How we ran the official [DDAD](https://arxiv.org/abs/2305.15956) implementation (this `DDAD/` folder) to produce the baseline numbers, and which checkpoints were used.

## Environment

```bash
conda activate ddad_env
cd DDAD/
```

All commands below are run from inside `DDAD/`.

## Basic Detection Command

```bash
CUDA_VISIBLE_DEVICES=<gpu> python main.py \
    --config config.yaml \
    --detection True \
    --category <category> \
    --load_chp <epoch> \
    --checkpoint_dir <checkpoint_base> \
    --seed 42
```

The checkpoint is resolved as `<checkpoint_base>/<category>/<epoch>`. The reported metric is parsed from the line `Per-image Pixel AUROC (mean):` in the output.

Reference eval recipe (as in `config.yaml`, matching the official DDAD settings): `w=2`, `v=1`, `test_trajectoy_steps=250`, `skip=25`, `eta=1`, frozen feature extractor (`use_frozen_fe: True`).

## Checkpoints

### MVTec (per-category diffusion UNets)

Base directory: `/data/akherandish3/MVTec` — one subfolder per category containing the best epoch checkpoint (epochs follow the official DDAD release):

| Category | Epoch | Category | Epoch |
|---|---|---|---|
| bottle | 1000 | pill | 1000 |
| cable | 3000 | screw | 2000 |
| capsule | 1500 | tile | 1000 |
| carpet | 2500 | toothbrush | 2000 |
| grid | 2000 | transistor | 2000 |
| hazelnut | 2000 | wood | 2000 |
| leather | 2000 | zipper | 1000 |
| metal_nut | 3000 | | |

Example:

```bash
python main.py --config config.yaml --detection True \
    --category bottle --load_chp 1000 \
    --checkpoint_dir /data/akherandish3/MVTec --seed 42
```

### Faces (FFHQ)

- **Checkpoint used:** `/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/faces/2000` (UNet trained for 2000 epochs on the faces `train/good` split; intermediate checkpoints every 250 epochs, 250–2000, are in the same folder).
- **Config:** `config_faces.yaml` (dataset `/data/akheirandish3/mvtec_ad/faces`, anomalous test images under `test/random/`, feature extractor `wide_resnet101_2`, frozen).

```bash
python main.py --config config_faces.yaml --detection True \
    --category faces --load_chp 2000 \
    --checkpoint_dir /data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec --seed 42
```

### Other datasets (for reference)

| Dataset | Checkpoint | Config |
|---|---|---|
| CT | `/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/combined/CT/2000` | `config.yaml` |
| X-ray | `/data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/500` | `config.yaml` |

## Batch Runs

- `load_chp.sh` — loops over all MVTec categories with the per-category epochs above, runs detection per seed, and writes `results_full_<timestamp>.txt` (raw logs) and `results_averaged_<timestamp>.txt` (mean ± std of pixel AUROC over seeds).
- `hparam_sweep.sh` / `run_offmvtec_variant.sh` — re-run detection with eval-time hyperparameter overrides (e.g. `model.v=7`); logs land in `hparam_sweep/<variant>/<category>.log`.

The reported MVTec baseline numbers come from `results_averaged_20260506_203015.txt` (reference recipe, per-category checkpoints above).
