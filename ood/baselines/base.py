"""Common Baseline interface + metadata container.

Mirrors `ood/samplers/base.py` so the two abstractions feel symmetric:
samplers produce *reconstructions* from a UNet; baselines produce *anomaly
maps* from images + (optional) ground-truth masks.

Each concrete baseline:
  * Takes a config dict and loads its checkpoint (or starts from scratch in
    a future training mode) in `__init__`.
  * Implements `score(image, gt_mask=None) -> anomaly_map` returning a
    `(H, W)` float numpy array — higher = more anomalous.
  * Optionally implements `train(train_dir, **kwargs)` for retraining;
    inference-only baselines can leave it as the default `NotImplementedError`.
  * Exposes `metadata` describing the model + checkpoint provenance.

The `tools/baselines_eval.py` driver consumes only `score()` and `metadata`,
so any new baseline plugs in via two methods + one registration line.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class BaselineMetadata:
    """Provenance metadata for a baseline invocation.

    These fields drive:
      * On-disk output layout (via `name`)
      * The `sweep_*.json` provenance block (via `ckpt_source` + `hyperparams`)
      * Logging
    """
    name: str                                   # registry key, e.g. "mdps"
    ckpt_source: str                            # human-readable origin, e.g.
                                                # "/data/.../cable/3000 (DDAD)"
    hyperparams: Dict[str, Any] = field(default_factory=dict)


class Baseline(ABC):
    """Common interface for all anomaly-detection baselines.

    Contract
    --------
    A concrete Baseline:
      * Takes a config dict in __init__ and loads weights immediately.
      * Implements `score(image, gt_mask=None)`:
          - input  : `image` is `(H, W, 3)` uint8 in [0, 255] — same convention
                     as the rest of the eval pipeline (cf. `ood/data.py`).
          - input  : `gt_mask` is the optional `(H, W)` binary mask. Most
                     baselines ignore it; included for ones that need it
                     (e.g. supervised SuperSimpleNet variants, future train
                     hooks). None during pure inference.
          - output : `(H, W)` float numpy array, anomaly map. Higher = more
                     anomalous. No specific scale is assumed; downstream code
                     (AUROC, SNR) is scale-invariant.
        Determinism: the same `image` must produce a bitwise-identical
        anomaly map across calls. Set seeds inside the implementation if
        the baseline has randomness.
      * Exposes `metadata` as a `BaselineMetadata` instance.

    The ABC deliberately does NOT batch — drivers loop over images. Baselines
    that benefit from batching can override `score_batch()` (default falls
    back to a Python loop over `score()`).
    """

    @abstractmethod
    def score(
        self,
        image: np.ndarray,
        gt_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Produce an anomaly map for one image.

        Args:
            image: `(H, W, 3)` uint8 in [0, 255].
            gt_mask: optional `(H, W)` binary mask. Most baselines ignore it.

        Returns:
            `(H, W)` float anomaly map. Higher = more anomalous.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> BaselineMetadata:
        """Describe this baseline for logging + sweep JSON provenance."""
        ...

    def score_batch(
        self,
        images: np.ndarray,
        gt_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Default batched scoring: loop over `score()`.

        Subclasses with a faster batched path can override.

        Args:
            images: `(N, H, W, 3)` uint8.
            gt_masks: optional `(N, H, W)` binary masks, or None.

        Returns:
            `(N, H, W)` float anomaly maps.
        """
        out = []
        for i in range(images.shape[0]):
            gm = gt_masks[i] if gt_masks is not None else None
            out.append(self.score(images[i], gm))
        return np.stack(out, axis=0)

    def train(self, train_dir: str, **kwargs) -> None:
        """Optional training hook. Default: not implemented.

        Concrete baselines that support retraining (DRAEM, SimpleNet, etc.)
        override this to drive their own training loop. Stays optional so
        inference-only configurations don't need to define it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement training. "
            "Use the upstream repo's training script and point ckpt_path "
            "at the resulting weights."
        )
