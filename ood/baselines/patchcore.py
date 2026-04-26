"""PatchCore baseline (Roth et al., CVPR'22) — official Amazon Science impl.

`PatchCore` is a memory-bank method: a frozen ImageNet-pretrained backbone
extracts patch features from the train/good/ split, a coreset is greedily
subsampled, and inference looks up nearest-neighbor distances against
that bank to produce per-pixel anomaly scores.

"Training" in PatchCore = feature extraction + greedy coreset selection
+ FAISS index build. Cheap (~30-90 sec/category on a single GPU). Saved
artifacts under `<load_path>/`:
    nnscorer_search_index.faiss   # FAISS index of the memory bank
    patchcore_params.pkl          # backbone name + hyperparams

This wrapper:
  - Imports PatchCore's official module (`patchcore.patchcore.PatchCore`).
  - Loads a saved memory bank via `load_from_path(...)`.
  - On `score(image, gt=None)` returns a dense `(H, W)` anomaly map
    upsampled to the input resolution.

Hyperparameters (config["params"]):
  ckpt              path to the saved-bank directory (the one containing
                    `patchcore_params.pkl` + `nnscorer_search_index.faiss`)
  device            "cuda:0"
  patchcore_repo    upstream repo path (default cluster path)
  imagesize         input H = W to forward through the backbone (default 224 to
                    match the WR50/L2-3/IM224 baseline; the saved memory bank
                    encodes the value used at training)

Train new categories via upstream `bin/run_patchcore.py` (see
`scripts/train_patchcore_mvtec.sh`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base import Baseline, BaselineMetadata


_DEFAULT_REPO = "/home/rohan/ood/baseline-algos-clone/patchcore-inspection"

_DEFAULTS: Dict[str, Any] = dict(
    imagesize=224,
    resize=256,
)


def _ensure_patchcore_on_path(repo: str) -> None:
    src = str(Path(repo) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if repo not in sys.path:
        sys.path.insert(0, repo)


class PatchCoreBaseline(Baseline):
    """Inference-only PatchCore around a saved-memory-bank checkpoint."""

    name = "patchcore"

    def __init__(self, config: Dict[str, Any]):
        ckpt = config.get("ckpt")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"PatchCore ckpt dir not found: {ckpt}")
        if not (Path(ckpt) / "patchcore_params.pkl").exists():
            raise FileNotFoundError(
                f"Missing patchcore_params.pkl under {ckpt} — is this a saved bank?"
            )
        device = str(config.get("device", "cuda:0"))
        repo = config.get("patchcore_repo", _DEFAULT_REPO)
        if not Path(repo).exists():
            raise FileNotFoundError(f"PatchCore repo not found: {repo}")

        _ensure_patchcore_on_path(repo)
        import patchcore.patchcore                                       # type: ignore
        import patchcore.common                                          # type: ignore

        merged = {**_DEFAULTS, **{k: v for k, v in config.items() if k in _DEFAULTS}}
        self._device = torch.device(device)
        self._imagesize = int(merged["imagesize"])
        self._resize = int(merged["resize"])
        self._merged = merged
        self._ckpt_source = ckpt

        # NB: faiss-cpu only (faiss-gpu 1.7.2 is incompatible with numpy 2.x);
        # PatchCore's NearestNeighbourScorer accepts a CPU FaissNN with no
        # accuracy loss — only inference latency.
        nn_method = patchcore.common.FaissNN(False, 4)
        model = patchcore.patchcore.PatchCore(self._device)
        model.load_from_path(load_path=ckpt, device=self._device,
                             nn_method=nn_method)
        model.eval()
        self._model = model

    @torch.no_grad()
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run PatchCore inference. Returns `(H, W)` anomaly map at input res.

        Preprocessing mirrors `patchcore.datasets.mvtec.MVTecDataset`:
            Resize(resize) → CenterCrop(imagesize) → ToTensor → ImageNet norm.
        """
        from PIL import Image as _Image

        out_h, out_w = image.shape[:2]
        # Resize → CenterCrop matching upstream's 256→224 pipeline
        pil = _Image.fromarray(image).resize(
            (self._resize, self._resize), _Image.BILINEAR,
        )
        left = (self._resize - self._imagesize) // 2
        top = (self._resize - self._imagesize) // 2
        pil = pil.crop((left, top, left + self._imagesize, top + self._imagesize))

        arr = np.asarray(pil).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

        # _predict returns lists of (image_score, anomaly_map). We use the
        # mask only — image_score is a global max we don't need at the
        # pixel-AUROC level.
        _scores, masks = self._model._predict(x)
        amap = np.asarray(masks[0], dtype=np.float32)

        if amap.shape != (out_h, out_w):
            pil_amap = _Image.fromarray(amap).resize((out_w, out_h), _Image.BILINEAR)
            amap = np.asarray(pil_amap, dtype=np.float32)
        return amap

    @property
    def metadata(self) -> BaselineMetadata:
        m = self._merged
        return BaselineMetadata(
            name=self.name,
            ckpt_source=str(self._ckpt_source),
            hyperparams={
                "resize": m["resize"],
                "imagesize": m["imagesize"],
                # Backbone name is read from the loaded ckpt; pull from model
                "backbone": getattr(getattr(self._model, "backbone", None),
                                    "name", "<unknown>"),
                "layers_to_extract_from": list(
                    getattr(self._model, "layers_to_extract_from", [])
                ),
                "patchsize": getattr(getattr(self._model, "patch_maker", None),
                                     "patchsize", None),
            },
        )
