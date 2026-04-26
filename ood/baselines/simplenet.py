"""SimpleNet baseline (Liu et al., CVPR'23).

WideResNet-50 backbone → multi-layer feature aggregation → 1-layer MLP
projection → discriminator. The discriminator is the actual anomaly
detector: it's trained to push real features down and synthetic
("noised") features up; at test time the negated discriminator score
is the per-patch anomaly score, which is upsampled to the input
resolution and used as a dense anomaly map.

We rely on the upstream `simplenet.SimpleNet` class for the model + the
`_predict()` inference path, and load existing per-class checkpoints
from the cluster (`/data/akheirandish3/SimpleNet/results/...`).

Implements `Baseline`:
  score(image, gt_mask=None) -> (H, W) float anomaly map.

Hyperparameters (config["params"]):
  ckpt                  path to `ckpt.pth` (the dict with 'discriminator' +
                        'pre_projection' state dicts, as written by
                        SimpleNet's training loop)
  device                "cuda:0"
  simplenet_repo        path to the SimpleNet repo
                        (default `/home/rohan/ood/baseline-algos-clone/SimpleNet`)
  backbone              "wideresnet50" (registry key in SimpleNet/backbones.py)
  layers_to_extract_from list of layer names, e.g. ["layer2", "layer3"]
  imagesize             input H = W the network was trained on (default 288).
                        Image is resized to this before inference and the
                        anomaly map is resized back to the GT resolution.
  pretrain_embed_dim,
  target_embed_dim      1536, 1536  (paper defaults)
  patchsize             3
  embedding_size        256
  dsc_layers            2
  dsc_hidden            1024
  dsc_margin            0.5
  pre_proj              1            (number of projection layers)

Train hook is not implemented yet — to retrain SimpleNet on a new dataset,
use the upstream `main.py` and point `ckpt` at the resulting `ckpt.pth`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import Baseline, BaselineMetadata


_DEFAULT_REPO = "/home/rohan/ood/baseline-algos-clone/SimpleNet"

_DEFAULTS: Dict[str, Any] = dict(
    backbone="wideresnet50",
    layers_to_extract_from=("layer2", "layer3"),
    # Inference preprocessing — must mirror what `bash run.sh` used during
    # training: Resize(resize) → CenterCrop(imagesize). Default values match
    # the upstream run.sh (`--resize 329 --imagesize 288`); change both
    # together if your ckpt was trained with different shapes.
    resize=329,
    imagesize=288,
    pretrain_embed_dim=1536,
    target_embed_dim=1536,
    patchsize=3,
    embedding_size=256,
    dsc_layers=2,
    dsc_hidden=1024,
    dsc_margin=0.5,
    pre_proj=1,
    proj_layer_type=0,
    # When True, return -anomaly_map. Useful when a checkpoint was trained
    # with the discriminator polarity flipped relative to the upstream
    # convention (suspected for our cluster's faces ckpt: AUC 0.28 → 0.72
    # if the sign is flipped).
    invert_score=False,
)


def _ensure_simplenet_on_path(repo: str) -> None:
    """Prepend the SimpleNet repo to sys.path so its bare imports resolve."""
    if repo not in sys.path:
        sys.path.insert(0, repo)


class SimpleNetBaseline(Baseline):
    """Inference-only SimpleNet wrapper around an existing per-class ckpt."""

    name = "simplenet"

    def __init__(self, config: Dict[str, Any]):
        ckpt = config.get("ckpt")
        if not ckpt:
            raise ValueError("simplenet baseline requires `ckpt` in config")
        if not Path(ckpt).exists():
            raise FileNotFoundError(f"SimpleNet ckpt not found: {ckpt}")
        device = str(config.get("device", "cuda:0"))
        repo = config.get("simplenet_repo", _DEFAULT_REPO)
        if not Path(repo).exists():
            raise FileNotFoundError(f"SimpleNet repo not found: {repo}")

        _ensure_simplenet_on_path(repo)
        # SimpleNet's main.py does `sys.path.append("src")` before importing
        # backbones — we replicate that since the package isn't pip-installable.
        src_path = str(Path(repo) / "src")
        if src_path not in sys.path and Path(src_path).exists():
            sys.path.insert(0, src_path)
        import backbones                                         # type: ignore
        from simplenet import SimpleNet as _SimpleNet            # type: ignore

        merged = {**_DEFAULTS, **{k: v for k, v in config.items() if k in _DEFAULTS}}
        self._device = device
        self._resize = int(merged["resize"])
        self._imagesize = int(merged["imagesize"])
        if self._resize < self._imagesize:
            raise ValueError(
                f"resize ({self._resize}) must be >= imagesize ({self._imagesize}); "
                "the upstream pipeline is Resize(resize) → CenterCrop(imagesize)"
            )
        self._invert_score = bool(merged["invert_score"])
        self._merged = merged
        self._ckpt_source = ckpt

        backbone = backbones.load(merged["backbone"])
        backbone.name = merged["backbone"]
        backbone.seed = 0

        model = _SimpleNet(device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=list(merged["layers_to_extract_from"]),
            device=device,
            input_shape=(3, self._imagesize, self._imagesize),
            pretrain_embed_dimension=merged["pretrain_embed_dim"],
            target_embed_dimension=merged["target_embed_dim"],
            patchsize=merged["patchsize"],
            embedding_size=merged["embedding_size"],
            dsc_layers=merged["dsc_layers"],
            dsc_hidden=merged["dsc_hidden"],
            dsc_margin=merged["dsc_margin"],
            pre_proj=merged["pre_proj"],
            proj_layer_type=merged["proj_layer_type"],
        )

        # Load discriminator + projection from the trained ckpt; backbone
        # stays at its ImageNet weights (SimpleNet doesn't fine-tune it).
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if "discriminator" not in state:
            raise ValueError(
                f"Expected SimpleNet ckpt with 'discriminator' key; got {list(state)}"
            )
        model.discriminator.load_state_dict(state["discriminator"])
        if model.pre_proj > 0 and "pre_projection" in state:
            model.pre_projection.load_state_dict(state["pre_projection"])
        model.discriminator.eval()
        if model.pre_proj > 0:
            model.pre_projection.eval()
        self._model = model

    @torch.no_grad()
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run SimpleNet inference on `image` (HxWx3 uint8) and return a
        dense `(H, W)` anomaly map at the input resolution.

        Preprocessing mirrors `SimpleNet/datasets/mvtec.py`:
            Resize(resize) → CenterCrop(imagesize) → ToTensor → ImageNet norm.
        After inference the anomaly map is bilinearly resized back to the
        input resolution. If `invert_score: true` is set in the YAML, the
        sign of the returned map is flipped — useful when a ckpt was
        trained with the discriminator polarity reversed.
        """
        from PIL import Image as _Image

        out_h, out_w = image.shape[:2]
        # Stage 1: Resize → CenterCrop. PIL.Image.resize uses BICUBIC by
        # default, while torchvision Resize defaults to BILINEAR. We use
        # BILINEAR to match upstream exactly.
        pil = _Image.fromarray(image).resize(
            (self._resize, self._resize), _Image.BILINEAR,
        )
        # CenterCrop equivalent
        left = (self._resize - self._imagesize) // 2
        top = (self._resize - self._imagesize) // 2
        pil = pil.crop((left, top, left + self._imagesize, top + self._imagesize))

        arr = np.asarray(pil).astype(np.float32) / 255.0
        # ImageNet normalization (matches SimpleNet/datasets/mvtec.py constants)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

        _, masks, _ = self._model._predict(x)
        amap = np.asarray(masks[0], dtype=np.float32)            # (imagesize, imagesize)

        if amap.shape != (out_h, out_w):
            amap_pil = _Image.fromarray(amap).resize((out_w, out_h), _Image.BILINEAR)
            amap = np.asarray(amap_pil, dtype=np.float32)
        if self._invert_score:
            amap = -amap
        return amap

    @property
    def metadata(self) -> BaselineMetadata:
        m = self._merged
        return BaselineMetadata(
            name=self.name,
            ckpt_source=str(self._ckpt_source),
            hyperparams={
                "backbone": m["backbone"],
                "layers_to_extract_from": list(m["layers_to_extract_from"]),
                "resize": m["resize"],
                "imagesize": m["imagesize"],
                "invert_score": m["invert_score"],
                "pretrain_embed_dim": m["pretrain_embed_dim"],
                "target_embed_dim": m["target_embed_dim"],
                "patchsize": m["patchsize"],
                "dsc_layers": m["dsc_layers"],
                "dsc_hidden": m["dsc_hidden"],
                "pre_proj": m["pre_proj"],
            },
        )
