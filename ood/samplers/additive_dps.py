"""Additive-noise DPS sampler: y = x + σε, DPS reverse from pure noise.

Thin wrapper around DDAD_DPS/samplers.py's DPSPosteriorSampling class. See the
docstring there for the algorithm (DPS posterior sampling, Chung et al. 2023,
with linear β schedule and DDIM stride).

Hyperparameters (passed as a dict to __init__):
    beta_start, beta_end     linear β schedule; default 0.0001, 0.02
    trajectory_steps         total DDPM steps; default 1000
    skip                     DDIM stride; default 25 (→ 40 reverse steps)
    eta                      DDIM stochasticity; default 1.0
    sigma                    additive measurement noise std (y = x + σε); default 0.1
    scale                    DPS guidance step size; default 0.5
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from .base import Sampler, SamplerMetadata, _as_batched, _unbatched


def _ensure_ddad_dps_on_path() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    p = str(root / "DDAD_DPS")
    if p not in sys.path:
        sys.path.insert(0, p)


def _make_dps_config(params: Dict[str, Any], device: str) -> SimpleNamespace:
    defaults = dict(
        beta_start=0.0001, beta_end=0.02,
        trajectory_steps=1000, skip=25, eta=1.0,
    )
    merged = {**defaults, **params, "device": device}
    return SimpleNamespace(model=SimpleNamespace(**merged))


class AdditiveDPSSampler(Sampler):
    """DPS posterior sampling with additive-noise measurement model."""

    name = "additive_dps"

    def __init__(self, unet: torch.nn.Module, config: Dict[str, Any]):
        _ensure_ddad_dps_on_path()
        from samplers import DPSPosteriorSampling                # type: ignore

        self._device = config.get("device", next(unet.parameters()).device)
        self._device = str(self._device)
        self._sigma = float(config.get("sigma", 0.1))
        self._scale = float(config.get("scale", 0.5))

        self._dps_cfg = _make_dps_config(config, self._device)
        self._sampler = DPSPosteriorSampling(unet, self._dps_cfg)

    def sample(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        x_clean = _as_batched(x).to(self._device)
        noise = torch.randn_like(x_clean)
        y = x_clean + self._sigma * noise
        xs = self._sampler(y, scale=self._scale)
        return _unbatched(xs[-1])                                # final x0

    @property
    def metadata(self) -> SamplerMetadata:
        m = self._dps_cfg.model
        return SamplerMetadata(
            name=self.name,
            effective_sigma=self._sigma,
            test_origin="ddad_dps_whole",                        # preserve legacy dirname for backward compat
            hyperparams={
                "sigma": self._sigma,
                "scale": self._scale,
                "skip": m.skip,
                "eta": m.eta,
                "trajectory_steps": m.trajectory_steps,
                "beta_start": m.beta_start,
                "beta_end": m.beta_end,
            },
        )
