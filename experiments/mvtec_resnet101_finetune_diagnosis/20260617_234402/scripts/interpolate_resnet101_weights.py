"""
WiSE-FT-style linear interpolation between two ResNet-101 state dicts:

    theta_alpha = (1 - alpha) * theta_a + alpha * theta_b

Rules (per task spec):
  - only interpolate params present in both with same name AND same shape,
  - only interpolate floating-point tensors,
  - non-float tensors copied from theta_b (the fine-tuned side),
  - robust to DataParallel 'module.' nesting,
  - never overwrites the inputs; writes only to --out.

CLI usage:
  python interpolate_resnet101_weights.py \
      --a feat0.pth --b feat8.pth --alpha 0.3 --out interp_a0.3.pth

The interpolation logic lives in ddad_diag.interpolate_state_dicts and is the
same one used by evaluate_all_checkpoints.py.
"""
import os, sys, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
import ddad_diag as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="pretrained / first endpoint")
    ap.add_argument("--b", required=True, help="fine-tuned / second endpoint")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert os.path.abspath(args.out) not in (os.path.abspath(args.a), os.path.abspath(args.b)), \
        "refusing to overwrite an input checkpoint"
    sd_a = torch.load(args.a, map_location="cpu")
    sd_b = torch.load(args.b, map_location="cpu")
    out = D.interpolate_state_dicts(sd_a, sd_b, args.alpha)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}  (alpha={args.alpha}, {len(out)} tensors)")


if __name__ == "__main__":
    main()
