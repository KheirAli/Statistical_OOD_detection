"""Standalone WiSE-FT interpolation of two ResNet-101 state dicts, with no DDAD
dependencies (safe to run in the `ood` env).

    theta_alpha = (1 - alpha) * theta_a + alpha * theta_b

Only float tensors present in BOTH with matching shape are interpolated; every
other key is copied from theta_b (the fine-tuned side). Handles 'state_dict'/
'model' containers and DataParallel 'module.' prefixes. Writes only to --out.
"""
import argparse
import os

import torch


def _unwrap(sd):
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    if isinstance(sd, dict) and len(sd) and next(iter(sd)).startswith("module."):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


def interpolate(a, b, alpha):
    a, b = _unwrap(a), _unwrap(b)
    out = {}
    for k, vb in b.items():
        va = a.get(k)
        if (va is not None and torch.is_floating_point(vb)
                and torch.is_floating_point(va) and va.shape == vb.shape):
            out[k] = (1.0 - alpha) * va.float() + alpha * vb.float()
        else:
            out[k] = vb
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="pretrained / first endpoint (feat0)")
    ap.add_argument("--b", required=True, help="fine-tuned / second endpoint (feat8)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.path.abspath(args.out) not in (
        os.path.abspath(args.a), os.path.abspath(args.b)), "refusing to overwrite an input"
    out = interpolate(torch.load(args.a, map_location="cpu"),
                      torch.load(args.b, map_location="cpu"), args.alpha)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"  [interp] wrote {args.out} (alpha={args.alpha}, {len(out)} tensors)")


if __name__ == "__main__":
    main()
