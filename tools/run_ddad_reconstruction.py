"""R2: Generate DDAD-native reconstructions for cable samples 000..010.

For each test image we run DDAD's own Reconstruction (conditioned denoising from
x_{t*}) N times with different seeds and save to our eval's expected layout.

Output layout:
  results_patches_ddad_native/samples_XXX/ddad_native_0_4/inpainting/
    recon/{seed}_0_00000.png   (N recons per image)
    label/0_00000.png          (original 256x256)
    input/0_00000.png          (same as label — DDAD uses the clean image as input)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "DDAD"))

from unet import UNetModel as DDADUNetModel  # noqa: E402
from reconstruction import Reconstruction  # noqa: E402


class _Cfg:
    """Minimal attribute-access config shim (DDAD uses dot access)."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, _Cfg(**v) if isinstance(v, dict) else v)


def strip_module_prefix(sd):
    first_key = next(iter(sd))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_unet(ckpt_path, device):
    unet = DDADUNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = strip_module_prefix(sd)
    unet.load_state_dict(sd, strict=True)
    return unet.to(device).eval()


def image_to_tensor(path, device):
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_uint8(t):
    arr = t.detach().cpu().numpy()
    arr = (arr + 1.0) * 127.5
    return np.clip(arr, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def effective_sigma(cfg):
    """sqrt(1 - alpha_bar_{t*}) with the same schedule DDAD uses."""
    betas = np.linspace(cfg.model.beta_start, cfg.model.beta_end,
                        cfg.model.trajectory_steps, dtype=np.float64)
    alpha_bar = np.cumprod(1 - betas)
    t_star = cfg.model.test_trajectoy_steps
    return float(np.sqrt(1.0 - alpha_bar[t_star - 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000")
    p.add_argument("--image_dir", default="/data/akheirandish3/mvtec_ad/cable/test/combined")
    p.add_argument("--samples", nargs="+", default=[f"{i:03d}" for i in range(11)])
    p.add_argument("--num_seeds", type=int, default=20)
    p.add_argument("--w", type=float, default=2.0)
    p.add_argument("--test_trajectory_steps", type=int, default=250)
    p.add_argument("--skip", type=int, default=25)
    p.add_argument("--trajectory_steps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_root", default="./results_patches_ddad_native")
    p.add_argument("--test_origin", default="ddad_native")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading DDAD UNet from {args.ckpt}")
    unet = load_unet(args.ckpt, device)

    cfg = _Cfg(model={
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "trajectory_steps": args.trajectory_steps,
        "test_trajectoy_steps": args.test_trajectory_steps,
        "skip": args.skip,
        "eta": args.eta,
        "device": device,
    })
    sigma_eff = effective_sigma(cfg)
    print(f"Effective sigma at t*={args.test_trajectory_steps}: {sigma_eff:.4f}")

    recon = Reconstruction(unet, cfg)

    for sid in args.samples:
        img_path = Path(args.image_dir) / f"{sid}.png"
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        x = image_to_tensor(img_path, device)
        sample_name = f"samples_{sid}"
        out_dir = Path(args.out_root) / sample_name / f"{args.test_origin}_0_4" / "inpainting"
        (out_dir / "recon").mkdir(parents=True, exist_ok=True)
        (out_dir / "label").mkdir(parents=True, exist_ok=True)
        (out_dir / "input").mkdir(parents=True, exist_ok=True)

        label_u8 = tensor_to_uint8(x[0])
        Image.fromarray(label_u8).save(out_dir / "label" / "0_00000.png")
        Image.fromarray(label_u8).save(out_dir / "input" / "0_00000.png")

        print(f"[{sample_name}] w={args.w} t*={args.test_trajectory_steps} N={args.num_seeds}")
        for seed in range(args.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            xs = recon(x, x, args.w)
            x0 = xs[-1][0]
            Image.fromarray(tensor_to_uint8(x0)).save(
                out_dir / "recon" / f"{seed}_0_00000.png"
            )
            if (seed + 1) % 5 == 0 or seed == args.num_seeds - 1:
                print(f"  seed {seed + 1}/{args.num_seeds}")
        print(f"  -> {out_dir / 'recon'}")

    # Save sigma for downstream scorers
    with open(Path(args.out_root) / "sigma.txt", "w") as f:
        f.write(f"{sigma_eff:.6f}\n")
    print(f"\nEffective sigma written to {args.out_root}/sigma.txt")


if __name__ == "__main__":
    main()
