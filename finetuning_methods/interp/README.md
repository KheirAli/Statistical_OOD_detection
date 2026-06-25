# Interp (WiSE-FT) — weight-space interpolation

**No training at all.** Take the pretrained anchor (`feat0`) and a fully
fine-tuned checkpoint (`featN`) and **linearly blend their weights**:

```
W_interp = (1 - α) · W_pretrained  +  α · W_fullft
```

This is **WiSE-FT** (Weight-space Ensembling for Fine-Tuning): it recovers much
of the robustness that full fine-tuning destroys, by walking part-way back toward
the pretrained anchor. `α = 0` is pretrained, `α = 1` is Full FT; we use
**α = 0.3** by default.

> Read [`../README.md`](../README.md) first for the shared AE recipe and
> evaluation. This file only covers what is specific to interp.

---

## What it produces

A single plain ResNet-101 state dict that is a convex combination of two
existing checkpoints — no gradient steps, no U-Net, no data.

```python
# _interp_states.py
sd_interp[k] = (1 - alpha) * sd_a[k] + alpha * sd_b[k]   # per-tensor blend
```

**Script:** `all_categories_embedder/_interp_states.py`
```bash
python all_categories_embedder/_interp_states.py \
    --a    experiments/<exp>/<cat>/checkpoints/feat0.pth \
    --b    experiments/<exp>/<cat>/checkpoints/feat8.pth \
    --alpha 0.3 \
    --out  experiments/<exp>/<cat>/checkpoints/_interp/<cat>_a0.3.pth
```
(There is also `interpolate_resnet101_weights.py` in the fine-tuning `scripts/`
dir for the same operation.)

Because it's free, you can sweep α (0.1 … 0.9) and pick the best by the
downstream detector AUC — see `all_categories_embedder/run_interp_alpha_sweep.sh`.

---

## Run it (end to end)

### 1. Build the interpolated backbone (needs `feat0` + `featN` from Full FT)
```bash
python all_categories_embedder/_interp_states.py \
    --a .../checkpoints/feat0.pth --b .../checkpoints/feat8.pth \
    --alpha 0.3 --out .../checkpoints/_interp/pill_a0.3.pth
```

### 2. Train the AE head on the interpolated backbone
```bash
DDAD_FE_WEIGHTS=.../checkpoints/_interp/pill_a0.3.pth \
python all_categories_embedder/train_pixel_ae_interp.py \
    --weights .../checkpoints/_interp/pill_a0.3.pth \
    --data_dir /data/akheirandish3/mvtec_ad/pill/train/good \
    --save_model models/pixel_ae_interp/pill.pth \
    --latent_dim 3 --num_epochs 30 --seed 0 --n_restarts 3 \
    --ae_loss cos --select_loss mse_var
```

### 3. Evaluate
Same `evaluate_ddad_fe.py` call as the others, with
`DDAD_FE_WEIGHTS=.../_interp/pill_a0.3.pth` and
`--autoencoder_path models/pixel_ae_interp/pill.pth`.

---

## Expected behaviour / results

- Interp lands **between pretrained and Full FT** by construction, and at α=0.3
  it usually **keeps most of the pretrained robustness** while picking up a
  little of the adaptation.
- It is the **best method on several MVTec categories** (capsule 0.983, tile
  0.913, toothbrush 0.966) where a touch of adaptation helps but Full FT
  over-shoots.
- **X-ray/scissors:** between full FT and pretrained (use the α sweep for the
  exact value).
- **Average over 15 MVTec categories: interp 0.930 px-AUC** (vs pretrained
  0.942, LoRA 0.936, full FT 0.891).

**Takeaway:** the cheapest way to claw back robustness lost to full fine-tuning.
If you have a Full-FT checkpoint and it over-adapted, interpolate toward `feat0`
instead of throwing it away — and always sweep α against the detector AUC.
