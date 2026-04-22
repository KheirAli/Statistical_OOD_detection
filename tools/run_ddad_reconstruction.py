"""DEPRECATED — use tools/generate_recons.py.

Backward-compat shim. Emits a DeprecationWarning and dispatches to the new
unified driver via an in-process call.

Migration:
    old:  python tools/run_ddad_reconstruction.py --ckpt X --image_dir Y --samples 000 ...
    new:  python tools/generate_recons.py --sampler ddad_native --ckpt X --image_dir Y --samples 000 ...

    or:   python tools/generate_recons.py --recon_config configs/recon/ddad_native_cable.yaml
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.generate_recons import run


def main():
    warnings.warn(
        "tools/run_ddad_reconstruction.py is deprecated. "
        "Use `python tools/generate_recons.py --sampler ddad_native …` or "
        "`--recon_config configs/recon/ddad_native_cable.yaml`.",
        DeprecationWarning, stacklevel=2,
    )

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

    cfg = {
        "sampler": {
            "name": "ddad_native",
            "params": {
                "w": args.w,
                "test_trajectoy_steps": args.test_trajectory_steps,
                "skip": args.skip,
                "trajectory_steps": args.trajectory_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "eta": args.eta,
            },
        },
        "ckpt": args.ckpt,
        "image_dir": args.image_dir,
        "samples": args.samples,
        "num_seeds": args.num_seeds,
        "out_root": args.out_root,
        "device": args.device,
        "test_origin_override": args.test_origin,
        "bottom_suffix": "4",
    }
    run(cfg)


if __name__ == "__main__":
    main()
