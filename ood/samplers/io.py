"""Shared I/O helpers: UNet loading, image ↔ tensor, output layout."""
from pathlib import Path
from typing import Dict

import numpy as np
import sys
import torch
from PIL import Image


# ── DDAD UNet loading ────────────────────────────────────────────────
# The DDAD checkpoints used in this repo are DataParallel state dicts with
# a `module.` prefix. We strip it and instantiate the non-parallel UNet.

def _ensure_ddad_on_path() -> None:
    """Prepend the bundled DDAD/ dir to sys.path so DDAD.unet can be imported."""
    root = Path(__file__).resolve().parent.parent.parent
    ddad = str(root / "DDAD")
    if ddad not in sys.path:
        sys.path.insert(0, ddad)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict))
    if first.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_ddad_unet(ckpt_path: str, device: str = "cuda:0") -> torch.nn.Module:
    """Load the DDAD UNet architecture + weights from `ckpt_path`.

    Architecture follows DDAD's `main.py build_model` for `DDADS=False`:
    `UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)`.

    Raises on strict-load mismatch, which indicates the checkpoint doesn't
    match DDAD's expected architecture.
    """
    _ensure_ddad_on_path()
    from unet import UNetModel                                   # type: ignore

    unet = UNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _strip_module_prefix(state)
    unet.load_state_dict(state, strict=True)
    return unet.to(device).eval()


# ── Image ↔ tensor conversions ──────────────────────────────────────
# The pipeline operates in [-1, 1] float space, (B, C, H, W) layout.

def image_to_tensor(path: Path, device: str = "cuda:0", size: int = 256) -> torch.Tensor:
    """Load a PNG as a `(1, 3, size, size)` tensor in `[-1, 1]`."""
    img = Image.open(str(path)).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert `(3, H, W)` float tensor in `[-1, 1]` to `(H, W, 3)` uint8 in `[0, 255]`."""
    arr = t.detach().cpu().numpy()
    arr = (arr + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def save_image_tensor(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_uint8(t)).save(str(path))


# ── Output layout matching ood/data.py's loader ─────────────────────

def recon_dir(
    out_root: Path, sample_name: str, test_origin: str, bottom_suffix: str = "4"
) -> Path:
    """Return the directory where reconstructions for one sample should land.

    Matches the layout that `ood/data.py:load_reconstructions` reads:
        {out_root}/{sample_name}/{test_origin}_0_{bottom_suffix}/inpainting/{recon,label,input}/
    """
    return out_root / sample_name / f"{test_origin}_0_{bottom_suffix}" / "inpainting"


def prepare_sample_dirs(
    out_root: Path, sample_name: str, test_origin: str, bottom_suffix: str = "4"
) -> Dict[str, Path]:
    """Create the recon/label/input subdirs for one sample; return their paths."""
    base = recon_dir(out_root, sample_name, test_origin, bottom_suffix)
    dirs = {
        "recon": base / "recon",
        "label": base / "label",
        "input": base / "input",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Effective-σ computation (for the local-Gaussian scorer) ─────────

def effective_sigma_at_t_star(
    t_star: int, trajectory_steps: int = 1000,
    beta_start: float = 0.0001, beta_end: float = 0.02,
) -> float:
    """sqrt(1 - α̅_{t*}) with DDAD's linear β schedule.

    Used by samplers that start from `x_{t*}` (e.g. DDAD native); the resulting
    σ is what the local-Gaussian scorer consumes.
    """
    betas = np.linspace(beta_start, beta_end, trajectory_steps, dtype=np.float64)
    alpha_bar = np.cumprod(1 - betas)
    # t_star is 1-indexed in DDAD's convention; protect against off-by-one.
    idx = max(0, min(t_star - 1, trajectory_steps - 1))
    return float(np.sqrt(1.0 - alpha_bar[idx]))
