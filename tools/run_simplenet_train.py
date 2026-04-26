"""Run SimpleNet's upstream training with a pandas-2.x compatibility shim.

Upstream `SimpleNet/metrics.py` calls the long-deprecated
`pd.DataFrame.append(...)` (removed in pandas 2.0). With pandas≥2 the
call raises AttributeError after each meta_epoch's eval, which kills
the training loop *before* `torch.save(state_dict, ckpt_path)` runs —
so no checkpoint is written.

Rather than patching upstream, this runner monkey-patches the missing
method onto `pd.DataFrame` at import time, then dispatches to
SimpleNet's `main.py` exactly as `bash run.sh` would (same argv).

Usage matches upstream `main.py` argv (everything after the script name
is forwarded), e.g.:

    python tools/run_simplenet_train.py \
        --gpu 0 --seed 0 --log_group simplenet_mvtec --log_project FOO \
        --results_path results --run_name run \
        net -b wideresnet50 -le layer2 -le layer3 \
        --pretrain_embed_dimension 1536 --target_embed_dimension 1536 \
        --patchsize 3 --meta_epochs 40 --embedding_size 256 \
        --gan_epochs 4 --noise_std 0.015 --dsc_hidden 1024 \
        --dsc_layers 2 --dsc_margin .5 --pre_proj 1 \
        dataset --batch_size 8 --resize 329 --imagesize 288 \
        -d bottle mvtec /data/akheirandish3/mvtec_ad
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pandas as pd


# 1. Monkey-patch `pd.DataFrame.append` — pandas≥2 removed it. Used by
# `SimpleNet/metrics.py:115` inside `compute_pro`. Replicates the
# pre-2.0 semantics for the only call signature SimpleNet actually uses
# (`df.append(dict, ignore_index=True)`).
if not hasattr(pd.DataFrame, "append"):
    def _df_append(self, other, ignore_index=False, **kwargs):  # noqa: ANN001
        if isinstance(other, dict):
            other = pd.DataFrame([other])
        elif isinstance(other, list):
            other = pd.DataFrame(other)
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append   # type: ignore[attr-defined]


# 2. Make SimpleNet's modules importable. Upstream's `main.py` does
# `sys.path.append("src")` relative to its own cwd, so we mimic that.
SIMPLENET_REPO = Path(
    os.environ.get(
        "SIMPLENET_REPO", "/home/rohan/ood/baseline-algos-clone/SimpleNet",
    )
)
if not SIMPLENET_REPO.exists():
    raise FileNotFoundError(f"SIMPLENET_REPO not found: {SIMPLENET_REPO}")

for p in (str(SIMPLENET_REPO), str(SIMPLENET_REPO / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# 3. Switch cwd so relative paths in SimpleNet (results dir, etc.) resolve
# the same way `bash run.sh` would. Keep the original cwd recorded in case
# the caller passed in a relative `--results_path`.
_orig_cwd = os.getcwd()
os.chdir(SIMPLENET_REPO)
try:
    # 4. argv[0] becomes main.py; everything else is forwarded as-is.
    sys.argv = [str(SIMPLENET_REPO / "main.py")] + sys.argv[1:]
    runpy.run_path(str(SIMPLENET_REPO / "main.py"), run_name="__main__")
finally:
    os.chdir(_orig_cwd)
