# LoRA fine-tuning

**Inject low-rank adapters into `layer2`/`layer3` conv2 layers and train only
those** (~0.5 M params, 1.1 %). The frozen backbone weights never change; LoRA
learns a small additive `B·A` correction per targeted conv. On save, the adapter
is **folded back into a plain ResNet-101 state dict**, so the eval harness loads
a LoRA backbone exactly like any other.

> Read [`../README.md`](../README.md) first for the shared DDAD loss, AE recipe,
> and evaluation. This file only covers what is specific to LoRA.

---

## What it trains

```python
# ddad_da_finetune.py, --lora_rank r  ( > 0 enables LoRA )
base, lora_params, n_tr, n_tot = D.inject_lora(base, r=8, alpha=1.0)
# inject_lora targets = ("layer2","layer3"), adds rank-r A,B to each conv2
# trainable 512,000 / 45,061,160  (1.136 %)  -> only the A/B adapters
freeze_batchnorm(fe)   # keep pretrained BN running stats; only A/B adapt
```

- Only the LoRA `A`/`B` matrices are trainable; **BatchNorm stats are frozen**.
- Same DDAD domain-adaptation loss as the others.
- On every epoch the adapter is merged into a plain ResNet-101:
  `merge_lora_to_plain_resnet(fe.module)` → `feat{epoch}.pth` (loads identically
  to a fully fine-tuned model — no LoRA code needed at eval time).
- `feat0` is the clean pretrained backbone (adapter `B=0`).

Helpers live in
`experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_diag.py`
(`inject_lora`, `merge_lora_to_plain_resnet`).

---

## Run it

### 1. Fine-tune (rank 8)
```bash
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n ddad_env python \
  experiments/mvtec_resnet101_finetune_diagnosis/20260617_234402/scripts/ddad_da_finetune.py \
    --category   xray \
    --unet_ckpt  /data2/rohan/ckpts/DDAD/DvXray/MVTec/xray/2000 \
    --out_dir    experiments/xray_finetune_mine/lora/checkpoints \
    --da_epochs  4 --seed 0 \
    --lora_rank 8 --lora_alpha 1.0
```

| flag | value | notes |
|------|-------|-------|
| `--lora_rank` | 8 | rank of the `A·B` adapters; **>0 selects LoRA** |
| `--lora_alpha`| 1.0 | LoRA scaling |
| `--train_scope` | (ignored) | LoRA overrides the scope flag |

Common configs used elsewhere in the repo: `r4_50` (rank 4), `r8_50`,
`r8_100` (rank 8, full epochs) — the matched-UNet pipeline defaults to
`r8_100/feat8`.

### 2. Train the AE head + 3. Evaluate
Identical to the other methods (see [`../README.md`](../README.md)), pointing
`DDAD_FE_WEIGHTS` and `--weights` at the merged LoRA checkpoint
`experiments/.../lora/checkpoints/feat<E>.pth`, AE → `models/pixel_ae_lora/<cat>.pth`,
eval `--output_dir results_eval_lora/<cat>`.

---

## Expected behaviour / results

- LoRA changes far fewer weights than Full FT, so it **forgets less** and avoids
  the carpet-style collapse — but it is **not free of over-adaptation**:
- **X-ray/scissors: 0.932 px-AUC — the lowest of the four scopes** (below
  pretrained 0.947 and last-layer 0.953). On X-ray, even a small low-rank
  correction slightly erodes the detector's signal.
- On MVTec, LoRA is the **best px-AP and SNR on average** (px-AP 0.503, SNR 5.11)
  and competitive on AUC (0.936 avg, just under pretrained's 0.942) — it often
  sits between pretrained and Full FT.

**Takeaway:** a strong middle ground — much safer than Full FT, occasionally the
best on precision-recall metrics, but it can still trail pretrained on
already-easy categories. Compare against `feat0` and last-layer before adopting.
