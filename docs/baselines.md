# Anomaly-detection baselines

How to run and (eventually) train each baseline implemented under
`ood/baselines/`. The driver is the same `evaluate.py` used for our PMF
and Local-Gaussian scorers — pass `--scorer baseline` plus a YAML under
`configs/baselines/<name>_<dataset>.yaml`. Output is a `sweep_*.json`
with per-image pixel AUROC, AP, SNR and bootstrap 95% CIs, identical in
shape to the rest of the eval pipeline.

## How baselines plug in

Each baseline implements the `Baseline` ABC in `ood/baselines/base.py`:

```python
score(image: (H,W,3) uint8, gt_mask: (H,W) | None) -> (H,W) float
metadata -> BaselineMetadata
```

`evaluate.py` checks for `cfg["scoring"]["algorithm"] == "baseline"` and
dispatches to `_run_baseline_sample`, which loads the test image
directly from `data.image_dir`, calls `baseline.score()`, and computes
pixel-level metrics (skipping superpixel aggregation — baselines emit
dense maps natively). SNR uses the formula in `ood/metrics/snr.py`.

To register a new baseline:

1. Create `ood/baselines/<name>.py` with a `Baseline` subclass.
2. Add `<name>` to the `BASELINES` tuple in `ood/baselines/__init__.py`
   and the `if name == ...` dispatch in `build_baseline()`.
3. Add a `configs/baselines/<name>_<dataset>.yaml` with `data.image_dir`,
   `data.gt_mask`, `baseline.name`, and `baseline.params`.

## Status (snapshot at branch tip)

| Baseline | Cable | Faces | Notes |
|---|---|---|---|
| **MDPS** | ✅ implemented | ✅ implemented | Reuses our DDAD UNet ckpt — no extra training |
| **SimpleNet** | ✅ implemented | ✅ implemented | Loads cluster ckpt at `/data/akheirandish3/SimpleNet/results/...` |
| **DRAEM** | scaffolded — needs ckpt download | ❌ needs training | Pretrained MVTec weights at https://drive.google.com/uc?id=1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1 |
| **SuperSimpleNet** | scaffolded — needs ckpt download | ❌ needs training | Pretrained MVTec weights on HuggingFace at https://huggingface.co/papers/2508.19060 |
| **CutPaste** | needs training | needs training | No pretrained weights released; train via upstream `pytorch-cutpaste/run_training.py` |

DDAD's own scorer (`heat_map`) is run via `tools/ddad_multiseed_eval.py`,
not through this package — it's a per-recon metric, not a single-image
baseline.

## MDPS — Wu et al., IJCAI'24

Two-stage masked DPS: stage 1 produces an initial anomaly map via
standard DPS reconstruction; stage 2 thresholds that map and re-runs DPS
with the mask as a side-conditioning. We reuse our existing DDAD UNet
checkpoint (the `UNetModel` class in MDPS is byte-identical to DDAD's),
so no extra training or downloads are required.

```bash
python evaluate.py \
  --config configs/baselines/mdps_cable.yaml \
  --skip_sampling --no_plots --scorer baseline \
  --sample_names samples_000 samples_001 samples_002 samples_003 samples_004 \
                 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010
# faces:
python evaluate.py --config configs/baselines/mdps_faces.yaml \
  --skip_sampling --no_plots --scorer baseline \
  --sample_names samples_40 samples_44 samples_49 samples_61 samples_65 \
                 samples_80 samples_92 samples_98 samples_107 samples_129
```

Hyperparameters (`baseline.params`) match the upstream
`MDPS/config/mvtec/config_cable.yaml`. The `mask_repeat` and
`test_repeat` knobs are the paper's `N` (number of seeds averaged); we
default both to 1.

Repo path (read-only): `/home/rohan/ood/baseline-algos-clone/MDPS`.

## SimpleNet — Liu et al., CVPR'23

WideResNet-50 features → MLP projection → discriminator. The
discriminator is the anomaly scorer; its negated per-patch output is
upsampled to the input resolution.

We load the cluster's per-class checkpoint (`ckpt.pth` containing
`discriminator` + `pre_projection` state dicts):

- Cable: `/data/akheirandish3/SimpleNet/results/MVTecAD_Results/simplenet_mvtec/run/models/0/mvtec_cable/ckpt.pth`
- Faces: `/data/akheirandish3/SimpleNet/results/MVTecAD_Results_faces/simplenet_mvtec_faces/run/models/0/mvtec_faces/ckpt.pth`

```bash
python evaluate.py --config configs/baselines/simplenet_cable.yaml \
  --skip_sampling --no_plots --scorer baseline \
  --sample_names samples_000 samples_001 samples_002 samples_003 samples_004 \
                 samples_005 samples_006 samples_007 samples_008 samples_009 samples_010
```

Inputs are resized to `imagesize` (default 288) to match the training
resolution; the output anomaly map is bilinearly resized back to the
input/GT resolution.

To retrain on a new dataset, run upstream's `bash run.sh` after editing
the dataset paths, then point `baseline.params.ckpt` at the resulting
`ckpt.pth`. We don't yet drive training from this repo.

Repo path: `/home/rohan/ood/baseline-algos-clone/SimpleNet`.

## DRAEM — Zavrtanik et al., ICCV'21

Reconstructive subnetwork (autoencoder) + Discriminative subnetwork
(segmentation U-Net). Trains on the train/good split with synthetic
anomalies pasted from the Describable Textures Dataset (DTD); inference
runs both subnetworks on the test image and uses the segmentation mask
as the anomaly map.

**Pretrained MVTec weights**: download from
https://drive.google.com/uc?id=1eOE8wXNihjsiDvDANHFbg_mQkLesDrs1
(use `gdown` or run the upstream `scripts/download_pretrained.sh`).
Each class has two `.pckl` files — the reconstructive and the
discriminative subnet (e.g. `DRAEM_seg_large_ae_large_0.0001_800_bs8_cable_.pckl`
and `..._cable_seg.pckl`).

**Training** requires the DTD anomaly source (`./scripts/download_dataset.sh`
in the upstream repo, ~600 MB). For faces we'll need to train from
scratch — there's no DRAEM-faces checkpoint anywhere.

The wrapper at `ood/baselines/draem.py` (TODO) will load both sub-nets
and call `model_seg(...).softmax(dim=1)[:,1]` as the anomaly map.

Repo path: `/home/rohan/ood/baseline-algos-clone/DRAEM`.

## SuperSimpleNet — Rolih et al., ICPR'24 / JIMS'25

WideResNet-50 + segmentation/classification heads. Same training stack
as SimpleNet but with a unified supervised/unsupervised regime.

**Pretrained MVTec weights**: HuggingFace at
https://huggingface.co/papers/2508.19060 (or
https://drive.google.com/drive/folders/1bBKL7-xFgNrzOZVnED0jBgqT5poeYf0d).
Extract under `weights/0/mvtec/<class>/<ratio>/weights.pt`.

For faces we need to train. The upstream `train.py` is straightforward
(`python train.py mvtec`).

Wrapper at `ood/baselines/supersimplenet.py` (TODO).

Repo path: `/home/rohan/ood/baseline-algos-clone/SuperSimpleNet`.

## CutPaste — Li et al., CVPR'21 (unofficial impl)

ResNet-18 + projection head + KDE/Mahalanobis density on patch
embeddings. The released `pytorch-cutpaste` repo is unofficial and
ships **no pretrained weights** — we'll need to train from scratch on
both cable and faces.

Training is short (~256-step "epochs", default 256 epochs, ~30 min on
one GPU). Anomaly maps come from running the projection head on
sliding patches; the upstream `eval.py` writes per-class image-level
AUC by default — the wrapper at `ood/baselines/cutpaste.py` (TODO) will
need to expose pixel-level scores via the patch path.

Repo path: `/home/rohan/ood/baseline-algos-clone/pytorch-cutpaste`.

## Adding a new dataset

A new dataset is a YAML, not new code. Drop a
`configs/baselines/<baseline>_<newdataset>.yaml` with:

```yaml
data:
  image_dir: /path/to/<newdataset>/test
  gt_mask:
    path: /path/to/<newdataset>/ground_truth/{sample}_mask.png
    downsample_factor: 1
scoring: {algorithm: baseline}
baseline:
  name: <baseline>
  params: {...}            # ckpt path + hyperparams
eval:
  delta_smooth_sigmas: [null, 5.0]
  output_dir: ./results_eval/baseline_<baseline>_<newdataset>
```

Pass it to `evaluate.py` with `--sample_names <list>`. No code changes
to `evaluate.py`, the baseline wrapper, or the metrics module.
