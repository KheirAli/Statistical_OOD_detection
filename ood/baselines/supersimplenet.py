"""SuperSimpleNet baseline (Rolih et al., ICPR'24 / JIMS'25).

Inference-only wrapper. Loads per-category weights from the HuggingFace
release (`blaz-r/SuperSimpleNet-MVTecAD`) snapshotted under
`/data2/rohan/baseline_ckpts/supersimplenet_hf/mvtec/<class>/1/weights.pt`,
or any equivalent `weights.pt`.

Implements `Baseline`:
  score(image, gt_mask=None) -> (H, W) float anomaly map
  metadata -> name, ckpt_source, hyperparams

Hyperparameters (config["params"]):
  ckpt                    path to weights.pt
  device                  "cuda:0"
  supersimplenet_repo     path to upstream repo (default cluster path)
  image_size              (H, W) — must match training (default (256, 256))
  backbone                "wide_resnet50_2" (paper default)
  layers                  ["layer2", "layer3"]
  patch_size              3
  adapt_cls_feat          False (JIMS extension default; True for ICPR weights)
  noise_std               0.015 (only used during training; carried for ckpt compat)

Training is delegated to upstream `python train.py mvtec` — point `ckpt`
at the resulting weights and re-run; no code changes here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import Baseline, BaselineMetadata


_DEFAULT_REPO = "/home/rohan/ood/baseline-algos-clone/SuperSimpleNet"

_DEFAULTS: Dict[str, Any] = dict(
    image_size=(256, 256),
    backbone="wide_resnet50_2",
    layers=("layer2", "layer3"),
    patch_size=3,
    adapt_cls_feat=False,
    noise_std=0.015,
)


def _ensure_ssn_on_path(repo: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)


class SuperSimpleNetBaseline(Baseline):
    """Inference-only SuperSimpleNet around a per-category checkpoint."""

    name = "supersimplenet"

    def __init__(self, config: Dict[str, Any]):
        ckpt = config.get("ckpt")
        if not ckpt:
            raise ValueError("supersimplenet baseline requires `ckpt` in config")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"SuperSimpleNet ckpt not found: {ckpt}")
        device = str(config.get("device", "cuda:0"))
        repo = config.get("supersimplenet_repo", _DEFAULT_REPO)
        if not Path(repo).exists():
            raise FileNotFoundError(f"SuperSimpleNet repo not found: {repo}")

        _ensure_ssn_on_path(repo)
        from model.supersimplenet import SuperSimpleNet as _SSN          # type: ignore

        merged = {**_DEFAULTS, **{k: v for k, v in config.items() if k in _DEFAULTS}}
        self._device = device
        self._image_size = tuple(merged["image_size"])
        self._merged = merged
        self._ckpt_source = ckpt

        # Upstream config block uses these keys at construction; carry them
        # through so the model sees the same shape it was trained with.
        upstream_cfg = {
            "backbone": merged["backbone"],
            "layers": list(merged["layers"]),
            "patch_size": merged["patch_size"],
            "adapt_cls_feat": merged["adapt_cls_feat"],
            "noise_std": merged["noise_std"],
            # The next two are only consumed in training; safe to set anything.
            "noise": False,
            "perlin": False,
        }
        model = _SSN(image_size=self._image_size, config=upstream_cfg)
        # Upstream uses strict=False, mirror that — HF JIMS weights miss a
        # couple of training-only buffers and would error otherwise.
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False),
                              strict=False)
        model.eval().to(device)
        self._model = model

    @torch.no_grad()
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run SuperSimpleNet inference on `image` (H, W, 3) uint8.

        SuperSimpleNet's eval pipeline applies `transforms.Resize` to the
        configured image_size (256×256 in the JIMS release), then
        ImageNet normalization. We mirror it. Returns a `(out_h, out_w)`
        float anomaly map at the input resolution.
        """
        from PIL import Image as _Image

        out_h, out_w = image.shape[:2]
        # Match training-time preprocessing.
        pil = _Image.fromarray(image).resize(
            (self._image_size[1], self._image_size[0]), _Image.BILINEAR,
        )
        arr = np.asarray(pil).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

        # In eval mode the model returns (anomaly_map, anomaly_score) where
        # anomaly_map is already upsampled to image_size (Gaussian-smoothed
        # by AnomalyMapGenerator at sigma=4).
        anomaly_map, _score = self._model(x)
        amap = anomaly_map.squeeze().detach().cpu().numpy().astype(np.float32)

        if amap.shape != (out_h, out_w):
            amap_pil = _Image.fromarray(amap).resize((out_w, out_h), _Image.BILINEAR)
            amap = np.asarray(amap_pil, dtype=np.float32)
        return amap

    @property
    def metadata(self) -> BaselineMetadata:
        m = self._merged
        return BaselineMetadata(
            name=self.name,
            ckpt_source=str(self._ckpt_source),
            hyperparams={
                "backbone": m["backbone"],
                "layers": list(m["layers"]),
                "image_size": list(m["image_size"]),
                "patch_size": m["patch_size"],
                "adapt_cls_feat": m["adapt_cls_feat"],
            },
        )
