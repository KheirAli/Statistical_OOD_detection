"""Smoke test: load DDAD UNet checkpoint, verify forward pass shape."""
import os
import sys
import torch

DDAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DDAD")
sys.path.insert(0, DDAD_DIR)

from unet import UNetModel as DDADUNetModel

CKPT = "/data/akheirandish3/DDAD_checkpoints/checkpoints/MVTec/cable/3000"


def strip_module_prefix(sd):
    first_key = next(iter(sd))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def main():
    print(f"Instantiating DDADUNetModel(256, 64, n_heads=4, in_channels=3)...")
    unet = DDADUNetModel(256, 64, dropout=0.0, n_heads=4, in_channels=3)
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"  {n_params:,} params")

    print(f"Loading checkpoint {CKPT} ...")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = strip_module_prefix(sd)
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  first missing: {missing[:3]}")
    if unexpected:
        print(f"  first unexpected: {unexpected[:3]}")
    if not missing and not unexpected:
        print("  strict load would pass.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    unet = unet.to(device).eval()

    print(f"Forward pass on dummy (2, 3, 256, 256) at t=500 on {device}...")
    x = torch.randn(2, 3, 256, 256, device=device)
    t = torch.tensor([500, 500], device=device, dtype=torch.float)
    with torch.no_grad():
        out = unet(x, t)
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  output stats: mean={out.mean().item():.4f} std={out.std().item():.4f}")

    assert out.shape == x.shape, f"expected {x.shape}, got {out.shape}"
    print("PASS")


if __name__ == "__main__":
    main()
