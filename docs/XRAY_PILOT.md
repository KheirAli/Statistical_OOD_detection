# X-ray OOD Detection Pilot

**Goal**: Apply our existing pipeline (DDAD UNet + DPS/native sampling + typical-set
PMF scorer) to X-ray baggage screening. Detect guns/knives as OOD anomalies
without supervision on those classes.

**Dataset**: [SIXray](https://github.com/MeioJane/SIXray) (start here). PIDray
is a fallback if SIXray proves too noisy.

**Method**: Train a DDAD diffusion model on SIXray "Negative" (clean bags), run
our eval pipeline on "Positive" (bags with prohibited items).

---

## Phase A — infrastructure (done in this repo, see commits)

| artifact | role |
|---|---|
| [tools/prepare_xray_dataset.py](../tools/prepare_xray_dataset.py) | Convert SIXray → MVTec layout; bbox → binary mask |
| [DDAD/config_xray.yaml](../DDAD/config_xray.yaml) | DDAD training config (categories, checkpoint dir, w/v) |
| [configs/experiment_ddad_native_xray.yaml](../configs/experiment_ddad_native_xray.yaml) | Scoring config for DDAD-native recons |
| [configs/experiment_ddad_dps_xray.yaml](../configs/experiment_ddad_dps_xray.yaml) | Scoring config for additive-noise DPS recons |
| [scripts/run_xray_pilot.sh](../scripts/run_xray_pilot.sh) | Orchestrator: recons + eval, reuses existing `evaluate.py` |

Everything in Phase A is ready. Phases B–E are what you do tomorrow.

---

## Phase B — dataset download & prep (day 1, ~4–8h wall time)

### B1. Clone SIXray repo + download data
```bash
cd /data/<your-user>/
git clone https://github.com/MeioJane/SIXray.git SIXray_repo
# SIXray images aren't on GitHub — follow SIXray_repo/README for Baidu/Dropbox links.
# Expected structure after download (example):
#   SIXray_raw/
#     annotation/        # XML per image with bboxes
#     JPEGImages/        # all images (positive + negative)
#     ImageSets/
#       positive.txt     # list of positive image IDs
#       negative.txt     # list of negative image IDs
```

### B2. Ingest to MVTec layout
```bash
cd /home/rohan/ood/Statistical_OOD_detection  # or wherever you checked out
python tools/prepare_xray_dataset.py \
    --sixray_root /data/<your-user>/SIXray_raw \
    --output_root /data/<your-user>/xray_mvtec \
    --n_train 10000 \
    --n_test 500 \
    --image_size 256
```

This produces:
```
/data/<your-user>/xray_mvtec/
├── train/good/           # 10K clean X-rays at 256×256, grayscale→3ch
├── test/
│   ├── good/             # ~50 held-out clean images (for image-AUROC)
│   └── prohibited/       # 500 positive X-rays
└── ground_truth/
    └── prohibited/       # 500 bbox-derived binary masks
```

### B3. Sanity check
```bash
# Count
ls /data/<your-user>/xray_mvtec/train/good/ | wc -l           # expect 10000
ls /data/<your-user>/xray_mvtec/ground_truth/prohibited/ | wc -l   # expect 500

# Eyeball a few
python -c "
from PIL import Image
import glob, random
for p in random.sample(glob.glob('/data/<your-user>/xray_mvtec/train/good/*.png'), 3):
    img = Image.open(p); print(p, img.size, img.mode)
"
```

Both images and masks must be 256×256. Masks must be single-channel with values in {0, 255}.

---

## Phase C — pilot training (day 2–4, ~2–3 days GPU time)

### C1. Edit config
Edit [DDAD/config_xray.yaml](../DDAD/config_xray.yaml):
- `data.data_dir`: point at `/data/<your-user>/xray_mvtec`
- `model.checkpoint_dir`: absolute path where ckpts land (e.g. `/data/<your-user>/ddad_xray_ckpts/MVTec`)
- `model.epochs`: **500** for pilot (not 3000 — we gate first)

### C2. Train
```bash
cd DDAD
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/dps \
    python main.py --config config_xray.yaml --train True \
    2>&1 | tee /home/rohan/ood/ood-cleanup/logs/xray_train.log
```

Expected checkpoints: every 250 epochs → `checkpoint_dir/xray/250`, `.../500`.

### C3. Sanity check mid-training
Every ~100 epochs, sample 1 image and eyeball the recon. Should look:
- **Early (100 ep)**: blurry X-ray-ish output
- **Mid (250 ep)**: recognizable baggage outlines
- **End (500 ep)**: sharp recon, barely distinguishable from input

If mid-training recons are pure noise → LR too high or data normalization wrong.
If end-of-training is blurry → not enough data or epochs, or UNet capacity too small.

---

## Phase D — pilot evaluation + go/no-go gate (day 5, ~4 hours)

### D1. Run pilot
```bash
cd /home/rohan/ood/Statistical_OOD_detection
bash scripts/run_xray_pilot.sh
```

Produces:
- `./results_patches_xray_native/` — 20 DDAD recons per test image
- `./results_patches_xray_dps/` — 20 additive-noise DPS recons per test image
- `./results_eval_xray_*/` — 4 scorer variants × their JSONs

### D2. Decision gate
Look at mean Pixel AUROC σ=5 across the 4 cells:

| Sampler | Scorer | Target Px AUC |
|---|---|---|
| DDAD native | PMF + PCA | — |
| DDAD native | PMF no-PCA | — |
| DPS additive | PMF + PCA | — |
| DPS additive | PMF no-PCA | — |

**Go criteria (proceed to full scale)**: best cell ≥ **0.65** Px AUROC σ=5.
**Gray zone (investigate)**: 0.55–0.65. Likely means the feature extractor is
miscalibrated for X-rays. Options: swap `wide_resnet101_2` → radiology backbone
(RadImageNet), try 512×512, train longer.
**No-go**: < 0.55. The unsupervised paradigm may not transfer to SIXray.
Options: try PIDray (smaller, more controlled), or pivot.

---

## Phase E — full scale (day 6+, if D passes)

If pilot clears the gate:
1. Scale training data: 10K → 100K (or full 1M) training negatives.
2. Train to full 3000 epochs.
3. Run eval on full SIXray positive test set (not just 500 samples).
4. Ablations: sampler choice, PCA on/off, resolution 256 vs 512, feature extractor swap.
5. Compare numbers against SIXray's published supervised baselines (~70% mAP) and unsupervised baselines (~55%).

---

## Risks to keep watching

1. **Feature extractor mismatch** (most likely): ImageNet ResNet's PCA features don't transfer to X-rays. Mitigation: have the no-PCA scorer ready; have RadImageNet as a backup plan.
2. **Bbox-mask over-estimates pixel AUROC**: every number should also be reported at image level (no mask needed) as a cross-check.
3. **Label noise in SIXray Negative set**: some "negative" images may contain prohibited items. Plan: sample 100 random negatives and manually eyeball 5% of them before committing to full training.
4. **GPU compute**: 500 epochs on 10K images ≈ 2 days on 1 GPU. Full scale 3000 × 100K ≈ weeks. Budget.

## Out of scope for the pilot

- Local-Gaussian scorer: was weak even on cable; not worth testing on X-rays until PMF works.
- Multi-class anomaly detection (gun vs knife): we treat all prohibited items as one anomaly class.
- Supervised comparison: we're positioning this as zero-annotation alternative.
