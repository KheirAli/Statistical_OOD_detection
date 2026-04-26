"""CutPaste baseline (Li et al., CVPR'21; unofficial PyTorch impl).

Loads the per-category checkpoint produced by upstream
`pytorch-cutpaste/run_training.py` (a ProjectionNet `.tch` file) and
exposes a pixel-level anomaly map via sliding-window patch scoring +
Mahalanobis distance from training-set embeddings.

The unofficial repo's `eval.py` only returns image-level AUC. The
original paper (§3.4) describes pixel-level scoring via "patches at
multiple positions" passed through the same projection network, with
each patch's Mahalanobis distance projected back to a dense map. We
implement that here.

Pipeline at __init__:
  1. Load ProjectionNet from `ckpt`.
  2. Compute global L2-normalized embeddings on `train_dir`
     (cached to disk if `embed_cache` is set).
  3. Fit a Gaussian density on the train embeddings (mean + Ledoit-Wolf
     covariance), exactly as upstream `density.GaussianDensityTorch` does.

Pipeline per `score(image)`:
  1. Slide a `patch_size × patch_size` window with `patch_stride` over the
     resized 256×256 image (default 64×8 → 25×25 = 625 patches).
  2. Forward all patches through the ProjectionNet in batches; take the
     post-ResNet, pre-MLP global embedding (the 512-dim vector).
  3. L2-normalize, compute Mahalanobis distance to the train density.
  4. Place each patch's distance at its center pixel; max-pool overlapping
     patches; gaussian-smooth (σ=4) for spatial coherence.
  5. Resize back to the input image's resolution.

Hyperparameters (config["params"]):
  ckpt              path to the .tch produced by upstream training
  train_dir         path to <class>/train/good/ (for fitting density)
  device            "cuda:0"
  cutpaste_repo     upstream repo path (default cluster path)
  size              input size the model was trained at (default 256)
  patch_size        sliding-window edge (default 64)
  patch_stride      sliding-window stride (default 8)
  head_layer        ProjectionNet head depth (default 2 — matches our
                    `--head_layer 2` training command)
  variant           "3way" (default) — affects num_classes in the output
                    head; doesn't change the embedding we use
  embed_cache       optional path to cache train embeddings; if set,
                    avoids recomputing them on subsequent runs
  smooth_sigma      gaussian smoothing σ on the dense anomaly map (default 4)
  batch_size        forward-pass batch size for patch scoring (default 256)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import Baseline, BaselineMetadata


_DEFAULT_REPO = "/home/rohan/ood/baseline-algos-clone/pytorch-cutpaste"

_DEFAULTS: Dict[str, Any] = dict(
    size=256,
    patch_size=64,
    patch_stride=8,
    head_layer=2,
    variant="3way",
    smooth_sigma=4.0,
    batch_size=256,
)


def _ensure_cutpaste_on_path(repo: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)


class CutPasteBaseline(Baseline):
    """Inference-only CutPaste with sliding-window pixel-level scoring."""

    name = "cutpaste"

    def __init__(self, config: Dict[str, Any]):
        ckpt = config.get("ckpt")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"CutPaste ckpt not found: {ckpt}")
        train_dir = config.get("train_dir")
        if not train_dir or not Path(train_dir).exists():
            raise FileNotFoundError(f"CutPaste train_dir not found: {train_dir}")
        device = str(config.get("device", "cuda:0"))
        repo = config.get("cutpaste_repo", _DEFAULT_REPO)
        if not Path(repo).exists():
            raise FileNotFoundError(f"CutPaste repo not found: {repo}")

        _ensure_cutpaste_on_path(repo)
        from model import ProjectionNet                              # type: ignore
        from density import GaussianDensityTorch                     # type: ignore

        merged = {**_DEFAULTS, **{k: v for k, v in config.items() if k in _DEFAULTS}}
        self._device = device
        self._size = int(merged["size"])
        self._patch_size = int(merged["patch_size"])
        self._patch_stride = int(merged["patch_stride"])
        self._smooth_sigma = float(merged["smooth_sigma"])
        self._batch_size = int(merged["batch_size"])
        self._merged = merged
        self._ckpt_source = ckpt
        self._train_dir = train_dir

        head_layers = [512] * int(merged["head_layer"]) + [128]
        weights = torch.load(ckpt, map_location=device, weights_only=False)
        n_classes = weights["out.weight"].shape[0]
        model = ProjectionNet(pretrained=False, head_layers=head_layers,
                              num_classes=n_classes)
        model.load_state_dict(weights)
        model.eval().to(device)
        self._model = model

        # Train embeddings + Mahalanobis density fit. Cached on disk to
        # avoid recomputing on every eval.
        embed_cache = config.get("embed_cache")
        train_embeds = self._load_or_compute_train_embeds(
            embed_cache, train_dir, device,
        )
        train_embeds = F.normalize(train_embeds, p=2, dim=1)
        self._density = GaussianDensityTorch()
        self._density.fit(train_embeds)
        # GaussianDensityTorch keeps `mean` and `inv_cov` on cpu.
        self._density_mean = self._density.mean.to(device)
        self._density_inv_cov = self._density.inv_cov.to(device)

    # ─────────────── train embeddings ────────────────────────────────

    @torch.no_grad()
    def _load_or_compute_train_embeds(
        self, cache_path: Optional[str], train_dir: str, device: str,
    ) -> torch.Tensor:
        if cache_path and Path(cache_path).exists():
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        from PIL import Image as _Image
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        files = sorted(p for p in Path(train_dir).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"No PNGs under {train_dir}")
        embeds = []
        batch = []
        for f in files:
            img = _Image.open(f).convert("RGB").resize(
                (self._size, self._size), _Image.BILINEAR,
            )
            arr = (np.asarray(img).astype(np.float32) / 255.0 - mean) / std
            batch.append(torch.from_numpy(arr).permute(2, 0, 1))
            if len(batch) >= 32:
                e, _ = self._model(torch.stack(batch).to(device))
                embeds.append(e.cpu())
                batch = []
        if batch:
            e, _ = self._model(torch.stack(batch).to(device))
            embeds.append(e.cpu())
        embeds = torch.cat(embeds, dim=0)
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(embeds, cache_path)
        return embeds

    # ─────────────── pixel-level scoring ────────────────────────────

    @torch.no_grad()
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sliding-window patch scoring → dense anomaly map.

        Args:
            image: (H, W, 3) uint8.

        Returns:
            (H, W) float anomaly map at the input resolution.
        """
        from PIL import Image as _Image
        from scipy.ndimage import gaussian_filter as _gf

        out_h, out_w = image.shape[:2]
        # Resize to training-time resolution
        pil = _Image.fromarray(image).resize(
            (self._size, self._size), _Image.BILINEAR,
        )
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (np.asarray(pil).astype(np.float32) / 255.0 - mean) / std
        x = torch.from_numpy(arr).permute(2, 0, 1).to(self._device)        # (3,H,W)

        # Build all (h, w) sliding-window patches
        ps = self._patch_size
        ss = self._patch_stride
        H, W = self._size, self._size
        patches, centers = [], []
        for top in range(0, H - ps + 1, ss):
            for left in range(0, W - ps + 1, ss):
                patches.append(x[:, top:top + ps, left:left + ps])
                centers.append((top + ps // 2, left + ps // 2))

        # If patch_size == size, we'd only get one patch with no spatial info.
        # Force at least one stride sweep: caller should keep patch_size < size.

        # Forward in batches
        patch_tensor = torch.stack(patches, dim=0)                          # (N,3,ps,ps)
        # ProjectionNet expects images at the model's training shape; resize
        # patches to `size×size` to match the ResNet receptive field the
        # model was trained on. (Without this the per-patch features differ
        # from what the density was fit on.)
        patch_tensor = F.interpolate(
            patch_tensor, size=(self._size, self._size),
            mode="bilinear", align_corners=False,
        )
        scores = []
        for i in range(0, patch_tensor.shape[0], self._batch_size):
            batch = patch_tensor[i:i + self._batch_size]
            embed, _ = self._model(batch)
            embed = F.normalize(embed, p=2, dim=1)
            d = self._mahalanobis_batch(embed)
            scores.append(d.cpu())
        patch_scores = torch.cat(scores, dim=0).numpy()                     # (N,)

        # Reassemble to a dense (size, size) map: for each center, max-pool
        # this patch's score over the (ps × ps) window centered there.
        amap = np.full((H, W), -np.inf, dtype=np.float32)
        for s, (cy, cx) in zip(patch_scores, centers):
            top = cy - ps // 2
            left = cx - ps // 2
            block = amap[top:top + ps, left:left + ps]
            np.maximum(block, s, out=block)
        # Patches at edges leave a border of -inf — replace with the min
        # filled value so smoothing doesn't blow up.
        finite = amap[np.isfinite(amap)]
        if finite.size:
            amap[~np.isfinite(amap)] = finite.min()
        else:
            amap = np.zeros_like(amap)

        if self._smooth_sigma > 0:
            amap = _gf(amap, sigma=self._smooth_sigma, mode="nearest")

        if amap.shape != (out_h, out_w):
            pil_amap = _Image.fromarray(amap).resize((out_w, out_h), _Image.BILINEAR)
            amap = np.asarray(pil_amap, dtype=np.float32)
        return amap

    def _mahalanobis_batch(self, embed: torch.Tensor) -> torch.Tensor:
        """Mahalanobis distance from the fitted train density.

        Mirrors `GaussianDensityTorch.mahalanobis_distance` but keeps the
        computation on `self._device` so we don't ping-pong with CPU.
        """
        x_mu = embed - self._density_mean.unsqueeze(0)
        d = torch.einsum("im,mn,in->i", x_mu, self._density_inv_cov, x_mu)
        return d.clamp_min_(0).sqrt()

    @property
    def metadata(self) -> BaselineMetadata:
        m = self._merged
        return BaselineMetadata(
            name=self.name,
            ckpt_source=str(self._ckpt_source),
            hyperparams={
                "size": m["size"],
                "patch_size": m["patch_size"],
                "patch_stride": m["patch_stride"],
                "head_layer": m["head_layer"],
                "variant": m["variant"],
                "smooth_sigma": m["smooth_sigma"],
                "train_dir": str(self._train_dir),
            },
        )
