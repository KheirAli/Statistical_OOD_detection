"""R6 (overnight): DPS + inpainting with the DDAD UNet.

Pipeline:
  1. Load DDAD UNet + checkpoint.
  2. For each sample, define a binary inpainting mask M (center box).
  3. For each seed:
       - Construct y = M * x   (masked pixels zeroed; unmasked pixels observed)
       - DPS reverse from pure noise, with gradient ||M*(y - x0_hat)||  guiding each step.
  4. Save recons in the eval harness's expected layout.

Defaults: center 128x128 hard mask, N=20 seeds per image, all 11 cable samples,
DDIM stride 25 (40 reverse steps), scale=0.5 (DPS guidance), sigma=0.0 (no
measurement noise — we mask deterministically so the observed pixels are the
clean truth; the posterior is forced to match them exactly on the mask
complement).
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


def build_alpha_bar(beta_start, beta_end, T, device):
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float64)
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    betas = torch.cat([torch.zeros(1, device=device), betas], dim=0)  # [T+1]
    return (1 - betas).cumprod(dim=0)                                # [T+1]


def make_center_mask(size=256, box=128, device="cuda"):
    """Binary mask: 1 on observed pixels (outside box), 0 on masked (inside box)."""
    m = torch.ones(1, 1, size, size, device=device)
    lo = (size - box) // 2
    hi = lo + box
    m[:, :, lo:hi, lo:hi] = 0.0
    return m


def dps_inpaint_sample(unet, x_clean, mask, *,
                       T=1000, skip=25, eta=1.0, scale=0.5,
                       beta_start=1e-4, beta_end=0.02,
                       sigma_obs=0.0, device="cuda"):
    """
    Run DPS from pure noise with inpainting measurement y = M*x_clean + sigma_obs*n,
    guided by grad ||M*(y - x0_hat)||.
    Returns final x_0 estimate of shape (B, 3, H, W).
    """
    B, C, H, W = x_clean.shape
    alpha_bar = build_alpha_bar(beta_start, beta_end, T, device)

    def _alpha(t):
        return alpha_bar.index_select(0, t + 1).view(-1, 1, 1, 1)

    if sigma_obs > 0:
        y = mask * x_clean + sigma_obs * torch.randn_like(x_clean)
    else:
        y = mask * x_clean                                            # observed pixels

    xt = torch.randn_like(x_clean)
    seq = list(range(0, T, skip))
    seq_next = [-1] + seq[:-1]

    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        next_t = torch.full((B,), j, device=device, dtype=torch.long)
        at = _alpha(t)
        at_next = _alpha(next_t)

        xt = xt.detach().requires_grad_(True)
        et = unet(xt, t.float())
        # DDAD UNet returns single-channel eps (no learn_sigma) — nothing to slice.

        x0_hat = (xt - et * (1 - at).sqrt()) / at.sqrt()
        diff = mask * (y - x0_hat)
        norm = torch.linalg.norm(diff)
        grad = torch.autograd.grad(outputs=norm, inputs=xt)[0]

        xt_d = xt.detach()
        et_d = et.detach()
        x0_d = x0_hat.detach()

        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_d + c1 * torch.randn_like(x_clean) + c2 * et_d
        xt_next = xt_next - scale * grad

        xt = xt_next

    # Final step: return the last x0 estimate by one more call without grad
    with torch.no_grad():
        t0 = torch.zeros(B, device=device, dtype=torch.long)
        at0 = _alpha(t0)
        et = unet(xt, t0.float())
        x0 = (xt - et * (1 - at0).sqrt()) / at0.sqrt()
    return x0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000")
    p.add_argument("--image_dir", default="/data/akheirandish3/mvtec_ad/cable/test/combined")
    p.add_argument("--samples", nargs="+", default=[f"{i:03d}" for i in range(11)])
    p.add_argument("--num_seeds", type=int, default=20)
    p.add_argument("--box", type=int, default=128)
    p.add_argument("--skip", type=int, default=25)
    p.add_argument("--trajectory_steps", type=int, default=1000)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--sigma_obs", type=float, default=0.0)
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out_root", default="./results_patches_ddad_dps_inpaint")
    p.add_argument("--test_origin", default="ddad_dps_inpaint")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    unet = load_unet(args.ckpt, device)
    mask = make_center_mask(size=256, box=args.box, device=device)

    # Effective σ parameter for rohan's scorer (DPS from pure noise → samples
    # concentrate around a noise level defined by the DPS gradient scale; since
    # we don't have a clean theoretical σ here, use 1.0 as a placeholder and
    # let --sigma_rohan be overridden manually if needed).
    sigma_rohan_placeholder = 1.0

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

        # label = clean image; input = masked image (observed pixels)
        Image.fromarray(tensor_to_uint8(x[0])).save(out_dir / "label" / "0_00000.png")
        y_vis = mask * x
        Image.fromarray(tensor_to_uint8(y_vis[0])).save(out_dir / "input" / "0_00000.png")

        print(f"[{sample_name}] N={args.num_seeds} box={args.box} scale={args.scale}")
        for seed in range(args.num_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            x0 = dps_inpaint_sample(
                unet=unet, x_clean=x, mask=mask,
                T=args.trajectory_steps, skip=args.skip, eta=args.eta,
                scale=args.scale, beta_start=args.beta_start, beta_end=args.beta_end,
                sigma_obs=args.sigma_obs, device=device,
            )
            Image.fromarray(tensor_to_uint8(x0[0])).save(out_dir / "recon" / f"{seed}_0_00000.png")
            if (seed + 1) % 5 == 0 or seed == args.num_seeds - 1:
                print(f"  seed {seed + 1}/{args.num_seeds}")
        print(f"  -> {out_dir / 'recon'}")

    with open(Path(args.out_root) / "sigma.txt", "w") as f:
        f.write(f"{sigma_rohan_placeholder:.6f}\n")
    print(f"\nPlaceholder sigma written to {args.out_root}/sigma.txt")


if __name__ == "__main__":
    main()
