"""Tests for DDAD UNet loading and forward pass.

Requires:
  - DDAD checkpoint on disk (tests auto-skipped otherwise)
  - CUDA GPU (tests auto-skipped otherwise)
"""
import sys
from pathlib import Path

import pytest
import torch

# DDAD's UNet lives in DDAD/unet.py and imports relative names
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DDAD"))


def _strip_module_prefix(sd):
    first_key = next(iter(sd))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


@pytest.mark.requires_ckpt
def test_ddad_unet_loads_strict(ddad_ckpt_path):
    """The cable/3000 checkpoint should strict-load into DDADUNetModel(256, 64, n_heads=4)."""
    from unet import UNetModel
    unet = UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    sd = torch.load(ddad_ckpt_path, map_location="cpu", weights_only=False)
    sd = _strip_module_prefix(sd)
    # strict=True → raises on mismatch
    unet.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in unet.parameters())
    assert n_params == 32_952_707


@pytest.mark.requires_ckpt
@pytest.mark.requires_gpu
def test_ddad_unet_forward_shape(ddad_ckpt_path):
    """Forward on a dummy batch returns noise prediction of the same shape."""
    from unet import UNetModel
    unet = UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    sd = _strip_module_prefix(torch.load(ddad_ckpt_path, map_location="cpu", weights_only=False))
    unet.load_state_dict(sd, strict=True)
    device = "cuda:0"
    unet = unet.to(device).eval()
    x = torch.randn(2, 3, 256, 256, device=device)
    t = torch.tensor([500, 500], device=device, dtype=torch.float)
    with torch.no_grad():
        out = unet(x, t)
    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    # Output should be noise-scaled (rough sanity: std near 1 for a trained model
    # on arbitrary input is not guaranteed, but should at least be finite).
    assert torch.isfinite(out).all()


@pytest.mark.requires_ckpt
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_ddad_native_sampler_one_seed(ddad_ckpt_path):
    """One seed of DDAD's Reconstruction class should return a clean image."""
    from unet import UNetModel
    from reconstruction import Reconstruction

    class Cfg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, Cfg(**v) if isinstance(v, dict) else v)

    unet = UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    sd = _strip_module_prefix(torch.load(ddad_ckpt_path, map_location="cpu", weights_only=False))
    unet.load_state_dict(sd, strict=True)
    device = "cuda:0"
    unet = unet.to(device).eval()

    cfg = Cfg(model={
        "beta_start": 1e-4, "beta_end": 0.02, "trajectory_steps": 1000,
        "test_trajectoy_steps": 250, "skip": 25, "eta": 1.0, "device": device,
    })
    recon = Reconstruction(unet, cfg)
    x_clean = torch.randn(1, 3, 256, 256, device=device).clamp_(-1, 1)
    torch.manual_seed(0)
    xs = recon(x_clean, x_clean, 2.0)
    x0 = xs[-1]
    assert x0.shape == x_clean.shape
    assert torch.isfinite(x0).all()
