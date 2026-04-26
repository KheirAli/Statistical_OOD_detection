"""MDPS baseline (Wu et al., IJCAI'24): Masked Diffusion Posterior Sampling.

Two-stage anomaly detection:
  Stage 1.  Run standard DPS reconstruction starting from x_{t*} for
            `test_steps` timesteps. Compare the reconstruction to the input
            via a wide_resnet101_2 feature distance to get an initial
            anomaly map.
  Stage 2.  Threshold that map to a binary mask, then re-run DPS but with
            a *masked* posterior (only the high-anomaly region is updated).
            Take the final feature distance as the output anomaly map.

We reuse our existing DDAD UNet checkpoint — the MDPS UNet class
(`UNetModel(image_size, 64, dropout=0, n_heads=4, in_channels=3)`) is
byte-identical to DDAD's, so we get MDPS for "free" without any extra
training. The MDPS-specific code (sampler, distance, config plumbing) is
imported from the upstream repo at `mdps_repo` (default
`/home/rohan/ood/baseline-algos-clone/MDPS`).

Implements `Baseline`:
  score(image, gt_mask=None) -> (H, W) float anomaly map.
  metadata -> ckpt path + key hyperparams for sweep provenance.

Hyperparameters (passed through `config["params"]` from the YAML):
  ckpt                path to UNet state_dict (DDAD format, "module." prefix
                      OK)
  device              "cuda:0" / etc.
  mdps_repo           path to the MDPS repo on disk (default cluster path)
  image_size          256
  test_steps, skip    DDIM stride params for stage 1 (defaults 250 / 20)
  mask_steps,
    skip_mask         DDIM stride params for stage 2 (defaults 250 / 20)
  diffusion_steps     1000 (DDPM trajectory length)
  beta_start,
    beta_end          0.0001, 0.02 (linear β schedule — matches DDAD)
  eta                 1.0 (DDIM stochasticity)
  w, w_mask           50, 50 (DPS guidance strength — paper defaults)
  mask0_thresholds    0.19 (cutoff for the binary mask between stages)
  mask_repeat,
    test_repeat       1, 1 (number of seeds — paper's "N" knob)
  resnet              "wide_resnet101_2" (feature extractor in distance())
  weight_decay        0.05 (used inside MDPS Resnet wrapper)
  num_workers         8
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import Baseline, BaselineMetadata


_DEFAULT_MDPS_REPO = "/home/rohan/ood/baseline-algos-clone/MDPS"

_DEFAULTS: Dict[str, Any] = dict(
    image_size=256,
    diffusion_steps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    eta=1.0,
    test_steps=250,
    skip=20,
    mask_steps=250,
    skip_mask=20,
    w=50,
    w_mask=50,
    mask0_thresholds=0.19,
    mask_repeat=1,
    test_repeat=1,
    resnet="wide_resnet101_2",
    weight_decay=0.05,
    num_workers=8,
    imput_channel=3,                # MDPS keeps the typo; we mirror it
    save_model=False,
    name="MVTec",                   # MDPS branches on dataset name
    category="cable",
    batch_size=1,
)


def _ensure_mdps_on_path(mdps_repo: str) -> None:
    """Prepend the MDPS repo to sys.path so its imports resolve.

    MDPS uses bare `from src.models.unet import UNetModel` style imports,
    which only work when the repo root is on sys.path.
    """
    if mdps_repo not in sys.path:
        sys.path.insert(0, mdps_repo)


def _make_mdps_config(params: Dict[str, Any], device: str) -> SimpleNamespace:
    """Build the OmegaConf-shaped `cfg.data.*` / `cfg.model.*` object MDPS
    expects, from a flat dict.

    MDPS reads:
      cfg.data.{name, image_size, batch_size, imput_channel, category}
      cfg.model.{device, resnet, weight_decay, eta, w, w_mask,
                 diffusion_steps, mask_steps, test_steps, skip, skip_mask,
                 mask0_thresholds, mask_repeat, test_repeat, beta_start,
                 beta_end, save_model, num_workers}
    """
    merged = {**_DEFAULTS, **params}
    data_keys = ("name", "image_size", "batch_size", "imput_channel", "category")
    model_keys = (
        "device", "resnet", "weight_decay", "eta", "w", "w_mask",
        "diffusion_steps", "mask_steps", "test_steps", "skip", "skip_mask",
        "mask0_thresholds", "mask_repeat", "test_repeat",
        "beta_start", "beta_end", "save_model", "num_workers",
    )
    data_ns = SimpleNamespace(**{k: merged[k] for k in data_keys if k in merged})
    model_ns = SimpleNamespace(
        device=device,
        **{k: merged[k] for k in model_keys if k in merged and k != "device"},
    )
    return SimpleNamespace(data=data_ns, model=model_ns)


class MDPSBaseline(Baseline):
    """Inference-only MDPS, reusing our existing DDAD UNet checkpoint."""

    name = "mdps"

    def __init__(self, config: Dict[str, Any]):
        ckpt = config.get("ckpt")
        if not ckpt:
            raise ValueError("mdps baseline requires `ckpt` in config")
        device = str(config.get("device", "cuda:0"))
        mdps_repo = config.get("mdps_repo", _DEFAULT_MDPS_REPO)
        if not Path(mdps_repo).exists():
            raise FileNotFoundError(f"MDPS repo not found at {mdps_repo}")

        _ensure_mdps_on_path(mdps_repo)
        from src.models.unet import UNetModel                        # type: ignore
        from src.models.resnet import Resnet                         # type: ignore
        from src.diffusion import compute_alpha, sample, sample_mask  # type: ignore
        from src.compare import distance                              # type: ignore

        self._device = device
        self._cfg = _make_mdps_config(config, device)

        unet = UNetModel(
            self._cfg.data.image_size, 64, dropout=0.0,
            n_heads=4, in_channels=self._cfg.data.imput_channel,
        )
        unet = torch.nn.DataParallel(unet)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # Some DDAD checkpoints save the state-dict directly (no module.prefix);
        # MDPS's loader assumes DataParallel-wrapped weights. Both cases work
        # because we wrap the bare model in DataParallel before load_state_dict.
        if state and not next(iter(state)).startswith("module."):
            state = {f"module.{k}": v for k, v in state.items()}
        unet.load_state_dict(state, strict=True)
        unet = unet.to(device).eval()
        self._unet = unet

        self._resnet = Resnet(self._cfg).to(device).eval()
        self._sample = sample
        self._sample_mask = sample_mask
        self._compute_alpha = compute_alpha
        self._distance = distance
        self._ckpt_source = ckpt

    @torch.no_grad()
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run two-stage MDPS on `image` (256x256x3 uint8) and return a
        (256, 256) float anomaly map."""
        # uint8 [0,255] HWC → float [-1,1] (1,3,H,W)
        x = torch.from_numpy(image.astype(np.float32) / 127.5 - 1.0)
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous().to(self._device)

        cfg = self._cfg
        # Stage 1: standard DPS, average over test_repeat seeds
        seq = list(range(0, cfg.model.test_steps, cfg.model.skip))
        anomaly_stage1 = []
        for _ in range(cfg.model.test_repeat):
            t_star = torch.tensor(
                [cfg.model.test_steps], dtype=torch.long, device=self._device,
            )
            at = self._compute_alpha(t_star, cfg)
            noisy = at.sqrt() * x + (1 - at).sqrt() * torch.randn_like(x)
            recon = self._sample(x, noisy, seq, self._unet, cfg, w=cfg.model.w)
            recon_x0 = recon[-1]
            amap = self._distance(recon_x0, x, self._resnet, cfg) / 2
            anomaly_stage1.append(amap.unsqueeze(0))
        anomaly_stage1 = torch.cat(anomaly_stage1, dim=0).mean(dim=0)

        # If mask_steps == 0, paper skips stage 2 — return stage 1 directly.
        if cfg.model.mask_steps == 0:
            out = anomaly_stage1.squeeze().detach().cpu().numpy()
            return out.astype(np.float32)

        # Stage 2: build init mask from stage 1, then masked DPS
        a_min = anomaly_stage1.min()
        a_max = anomaly_stage1.max()
        thr = a_min + cfg.model.mask0_thresholds * (a_max - a_min)
        mask_init = (anomaly_stage1 > thr).float()
        if mask_init.dim() == 3:                  # (1, H, W) -> (1, 1, H, W)
            mask_init = mask_init.unsqueeze(0)
        if mask_init.dim() == 2:                  # (H, W) -> (1, 1, H, W)
            mask_init = mask_init.unsqueeze(0).unsqueeze(0)

        seq2 = list(range(0, cfg.model.test_steps, cfg.model.skip))
        anomaly_stage2 = []
        for _ in range(cfg.model.mask_repeat):
            recon = self._sample_mask(
                x, mask_init, seq2, self._unet, cfg, w=cfg.model.w,
            )
            recon_x0 = recon[-1]
            amap = self._distance(recon_x0, x, self._resnet, cfg) / 2
            anomaly_stage2.append(amap.unsqueeze(0))
        anomaly_stage2 = torch.cat(anomaly_stage2, dim=0).mean(dim=0)

        out = anomaly_stage2.squeeze().detach().cpu().numpy()
        return out.astype(np.float32)

    @property
    def metadata(self) -> BaselineMetadata:
        m = self._cfg.model
        return BaselineMetadata(
            name=self.name,
            ckpt_source=str(self._ckpt_source),
            hyperparams={
                "test_steps": m.test_steps,
                "mask_steps": m.mask_steps,
                "skip": m.skip,
                "skip_mask": m.skip_mask,
                "w": m.w,
                "w_mask": m.w_mask,
                "mask0_thresholds": m.mask0_thresholds,
                "mask_repeat": m.mask_repeat,
                "test_repeat": m.test_repeat,
                "eta": m.eta,
                "diffusion_steps": m.diffusion_steps,
            },
        )
