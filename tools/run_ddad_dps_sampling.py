"""Run DPS sampling with the DDAD UNet checkpoint.

Uses DDAD_DPS/samplers.DPSPosteriorSampling (Mode 1: DPS from pure noise with
additive-noise measurement y = x + sigma * n) and writes output into the eval
harness's expected layout so evaluate.py can read it.

Output layout (matches ood/data.py expectations):
  {out_root}/{sample_name}/{test_origin}_0_4/inpainting/
    recon/{seed}_0_00000.png
    label/0_00000.png
    input/0_00000.png    (noisy measurement y)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, _Cfg(**v) if isinstance(v, dict) else v)

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT / "DDAD"))
sys.path.insert(0, str(ROOT / "DDAD_DPS"))

from unet import UNetModel as DDADUNetModel  # noqa: E402
from samplers import DPSPosteriorSampling  # noqa: E402


def strip_module_prefix(sd):
    first_key = next(iter(sd))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_ddad_unet(ckpt_path, device):
    unet = DDADUNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = strip_module_prefix(sd)
    unet.load_state_dict(sd, strict=True)
    return unet.to(device).eval()


def image_to_tensor(path, device):
    """Load PNG as a [-1, 1] tensor (B=1, 3, 256, 256)."""
    img = Image.open(path).convert("RGB").resize((256, 256), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_uint8(t):
    """[-1, 1] float tensor (3, 256, 256) → (256, 256, 3) uint8."""
    arr = t.detach().cpu().numpy()
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000")
    p.add_argument("--image_dir", default="/data/akheirandish3/mvtec_ad/cable/test/combined")
    p.add_argument("--samples", nargs="+", default=["000"], help="sample IDs, e.g. 000 001 ...")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5, help="DPS guidance step size")
    p.add_argument("--num_seeds", type=int, default=32, help="# reconstructions per image")
    p.add_argument("--skip", type=int, default=25, help="DDIM sampling stride")
    p.add_argument("--trajectory_steps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_root", default="./results_patches_ddad")
    p.add_argument("--test_origin", default="ddad_dps_whole")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"Loading DDAD UNet from {args.ckpt}")
    unet = load_ddad_unet(args.ckpt, device)

    config = _Cfg(model={
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,
        "trajectory_steps": args.trajectory_steps,
        "skip": args.skip,
        "eta": args.eta,
        "device": device,
    })
    sampler = DPSPosteriorSampling(unet, config)

    for sample_id in args.samples:
        img_path = Path(args.image_dir) / f"{sample_id}.png"
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        x_clean = image_to_tensor(img_path, device)
        sample_name = f"samples_{sample_id}"
        out_dir = Path(args.out_root) / sample_name / f"{args.test_origin}_0_4" / "inpainting"
        (out_dir / "recon").mkdir(parents=True, exist_ok=True)
        (out_dir / "label").mkdir(parents=True, exist_ok=True)
        (out_dir / "input").mkdir(parents=True, exist_ok=True)

        label_u8 = tensor_to_uint8(x_clean[0])
        Image.fromarray(label_u8).save(out_dir / "label" / "0_00000.png")

        print(f"[{sample_name}] sigma={args.sigma} scale={args.scale} N={args.num_seeds}")
        for seed in range(args.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            noise = torch.randn_like(x_clean)
            y = x_clean + args.sigma * noise

            if seed == 0:
                Image.fromarray(tensor_to_uint8(y[0])).save(out_dir / "input" / "0_00000.png")

            xs = sampler(y, scale=args.scale)
            x_recon = xs[-1][0]
            recon_u8 = tensor_to_uint8(x_recon)
            Image.fromarray(recon_u8).save(out_dir / "recon" / f"{seed}_0_00000.png")

            if (seed + 1) % 8 == 0 or seed == args.num_seeds - 1:
                print(f"  seed {seed + 1}/{args.num_seeds} done")

        print(f"  -> {out_dir / 'recon'} ({args.num_seeds} recons)")


if __name__ == "__main__":
    main()
