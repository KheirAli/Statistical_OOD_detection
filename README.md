# Statistical OOD Detection via Diffusion Reconstruction

Pixel-level anomaly detection on MVTec-AD using diffusion-based reconstruction
and information-theoretic scoring. We compare two **samplers** (DDAD-style
conditioned denoising vs additive-noise DPS) and three **scorers** (DDAD's
pixel+feature heat map, our typical-set PMF over RGB×PCA features, and a
theory-based local-Gaussian scorer) on identical UNet weights.

## Headline results (cable / test/combined, mean Px AUROC σ=5 over 11 images)

| scorer                          | DDAD native recons | additive-noise DPS recons |
|---------------------------------|-------------------:|--------------------------:|
| DDAD heat_map (baseline)        |          **0.969** | —                         |
| Typical-set PMF (ours)          |              0.962 |                     0.942 |
| Local-Gaussian (rohan, theory)  |              0.669 |                     0.661 |

Full comparison including faces, σ sweep, per-sample tables in [RESULTS.md](RESULTS.md).

## Quickstart

### Environment
```bash
conda create -n ood python=3.10 -y
conda activate ood
pip install -r requirements.txt
```

### Reproduce the headline on cable
```bash
# 1. DDAD baseline (their full pipeline)
cd DDAD && PYTHONPATH=/path/to/dps python main.py --config config_cable.yaml --detection True

# 2. Generate recons (DDAD native) + score with our PMF + local-Gaussian
cd ..
python tools/run_ddad_reconstruction.py --num_seeds 20
bash scripts/run_cable.sh
```

### Run on a new MVTec category
1. Copy `DDAD/config_cable.yaml` → `DDAD/config_<cat>.yaml`, set `category: <cat>` and `load_chp: <epoch>`.
2. Copy `configs/experiment_ddad_native.yaml` → `configs/experiment_ddad_native_<cat>.yaml`, set `data.image_dir`, `data.gt_mask.path`, `data.test_origin`, `data.figures_dir` accordingly.
3. Adapt `scripts/run_cable.sh` — swap sample IDs and config paths.

## Repository layout

```
.
├── README.md          # this file
├── RESULTS.md         # detailed results tables
├── docs/
│   ├── EXPERIMENTS.md # lab journal / detailed methodology
│   └── XRAY_PILOT.md  # X-ray pilot plan (SIXray OOD detection)
├── requirements.txt
├── evaluate.py        # main eval CLI — scorer × config dispatch
├── super_pixel_generation.py  # SLIC mask generator (called by evaluate.py)
├── ood/               # core eval package
│   ├── data.py              # recon / label / GT mask loaders
│   ├── superpixels.py       # recursive SLIC refinement
│   ├── embeddings.py        # ResNet pixel embeddings + PCA
│   ├── scoring.py           # typical-set PMF scorer (ours)
│   ├── scoring_local_gaussian.py  # local-Gaussian scorer (rohan)
│   ├── metrics.py           # AUROC / AP
│   └── visualize.py
├── tools/
│   ├── run_ddad_reconstruction.py  # DDAD native recons (w=2 conditioning)
│   └── run_ddad_dps_sampling.py    # additive-noise DPS recons
├── scripts/           # orchestrators (call into evaluate.py + tools/)
│   ├── run_cable.sh
│   ├── run_faces.sh
│   └── run_sigma_sweep.sh
├── configs/           # experiment YAMLs (data paths, scorer params)
├── DDAD/              # grad student's DDAD implementation (from upstream)
├── DDAD_DPS/          # alternative samplers for DDAD UNet
└── tests/             # pytest smoke tests
```

## Tests

```bash
pytest tests/ -v
```

Covers UNet loading, reconstruction output shapes, scoring end-to-end, metrics.
Tests that require external checkpoints are marked `@pytest.mark.requires_ckpt`
and can be skipped with `pytest -m "not requires_ckpt"`.

## Dependencies on external paths

- DDAD checkpoints: `/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/<category>/<epoch>`
- MVTec-AD data: `/data/akheirandish3/mvtec_ad/<category>/`
- The `guided_diffusion` module (from [DPS](https://github.com/DPS2022/diffusion-posterior-sampling)) must be on `PYTHONPATH` for `DDAD/main.py` (DDAD imports it as a fallback UNet).

These paths are hardcoded in the configs; override per-invocation via CLI
flags or by editing the YAML.

## Citation / acknowledgements

- **DDAD** (Mousakhan et al., WACV 2024) — reconstruction and scoring baseline.
- **DPS** (Chung et al., ICLR 2023) — posterior-sampling diffusion.
- Local-Gaussian scorer: derivation + reference notebook live outside the tracked repo (see `.gitignore`); production port is in [ood/scoring_local_gaussian.py](ood/scoring_local_gaussian.py).
