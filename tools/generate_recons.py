"""Unified reconstruction driver.

Replaces the old one-off tools (`run_ddad_reconstruction.py`,
`run_ddad_dps_sampling.py`) with a single entry point that dispatches to any
sampler registered in `ood/samplers/__init__.py`.

Two ways to invoke:

(a) Config-driven (preferred):
    python tools/generate_recons.py --recon_config configs/recon/ddad_native_cable.yaml

(b) Flat CLI:
    python tools/generate_recons.py \\
        --sampler ddad_native \\
        --ckpt /path/to/unet/ckpt \\
        --image_dir /path/to/test/images \\
        --samples 000 001 002 003 004 005 006 007 008 009 010 \\
        --num_seeds 20 \\
        --out_root ./results_patches_ddad_native \\
        --sampler_param w=2 --sampler_param test_trajectoy_steps=250

Sampler-specific hyperparameters flow through `--sampler_param KEY=VAL` (flat
CLI) or the `sampler.params` dict (YAML). Unknown keys are silently ignored by
the sampler, so forgetting `--sampler_param` mostly-just uses sensible defaults.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ood.samplers import build_sampler, SAMPLERS
from ood.samplers.io import (
    image_to_tensor, load_ddad_unet, prepare_sample_dirs, save_image_tensor,
)


def _parse_sampler_params(flat_kv_list):
    """Convert `--sampler_param k=v` repeats to a dict; cast numerics."""
    out: Dict[str, Any] = {}
    for kv in flat_kv_list or []:
        if "=" not in kv:
            raise ValueError(f"--sampler_param expects KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        # Cast to int/float if possible; fall back to string.
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _load_recon_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_config(args) -> Dict[str, Any]:
    """Merge YAML config + CLI overrides into one run config dict."""
    if args.recon_config:
        cfg = _load_recon_config(Path(args.recon_config))
    else:
        cfg = {
            "sampler": {"name": args.sampler, "params": _parse_sampler_params(args.sampler_param)},
            "ckpt": args.ckpt, "image_dir": args.image_dir,
            "samples": args.samples, "num_seeds": args.num_seeds,
            "out_root": args.out_root, "device": args.device,
            "test_origin_override": args.test_origin,
            "bottom_suffix": args.bottom_suffix,
        }

    # CLI always wins for these, when provided
    for k, cli_val in [
        ("ckpt", args.ckpt), ("image_dir", args.image_dir),
        ("num_seeds", args.num_seeds), ("out_root", args.out_root),
        ("device", args.device), ("bottom_suffix", args.bottom_suffix),
    ]:
        if cli_val is not None:
            cfg[k] = cli_val
    if args.samples:
        cfg["samples"] = args.samples
    if args.sampler:
        cfg.setdefault("sampler", {})["name"] = args.sampler
    if args.sampler_param:
        cfg.setdefault("sampler", {}).setdefault("params", {}).update(
            _parse_sampler_params(args.sampler_param)
        )
    if args.test_origin:
        cfg["test_origin_override"] = args.test_origin

    # Defaults (robust against explicit None values from YAML/CLI).
    cfg["num_seeds"] = cfg.get("num_seeds") or 20
    cfg["device"] = cfg.get("device") or "cuda:0"
    cfg["bottom_suffix"] = cfg.get("bottom_suffix") or "4"
    return cfg


def run(cfg: Dict[str, Any]) -> None:
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    print(f"[sampler]      {cfg['sampler']['name']}")
    print(f"[ckpt]         {cfg['ckpt']}")
    print(f"[image_dir]    {cfg['image_dir']}")
    print(f"[out_root]     {cfg['out_root']}")
    print(f"[num_seeds]    {cfg['num_seeds']}")

    unet = load_ddad_unet(cfg["ckpt"], device=device)
    sampler_cfg = {**cfg["sampler"].get("params", {}), "device": device}
    sampler = build_sampler(cfg["sampler"]["name"], unet, sampler_cfg)
    meta = sampler.metadata

    test_origin = cfg.get("test_origin_override") or meta.test_origin
    print(f"[test_origin]  {test_origin}")
    print(f"[eff_sigma]    {meta.effective_sigma:.4f}")
    print(f"[hparams]      {meta.hyperparams}")

    out_root = Path(cfg["out_root"])
    image_dir = Path(cfg["image_dir"])
    samples = [str(s) for s in cfg["samples"]]

    for sid in samples:
        img_path = image_dir / f"{sid}.png"
        if not img_path.exists():
            print(f"MISSING: {img_path}")
            continue

        x = image_to_tensor(img_path, device=device)
        sample_name = f"samples_{sid}"
        dirs = prepare_sample_dirs(
            out_root, sample_name, test_origin, bottom_suffix=cfg["bottom_suffix"],
        )

        # label + input are the original image; some scorers read these.
        save_image_tensor(x[0], dirs["label"] / "0_00000.png")
        save_image_tensor(x[0], dirs["input"] / "0_00000.png")

        print(f"[{sample_name}]  N={cfg['num_seeds']}")
        for seed in range(int(cfg["num_seeds"])):
            recon = sampler.sample(x, seed=seed)
            save_image_tensor(recon, dirs["recon"] / f"{seed}_0_00000.png")
            milestone = (seed + 1) % 5 == 0 or seed == int(cfg["num_seeds"]) - 1
            if milestone:
                print(f"  seed {seed + 1}/{cfg['num_seeds']}")

    # Write σ + full metadata for downstream scorers and reproducibility.
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "sigma.txt", "w") as f:
        f.write(f"{meta.effective_sigma:.6f}\n")
    with open(out_root / "recon_metadata.json", "w") as f:
        json.dump({
            "sampler": meta.name,
            "test_origin": test_origin,
            "effective_sigma": meta.effective_sigma,
            "hyperparams": meta.hyperparams,
            "ckpt": cfg["ckpt"],
            "image_dir": cfg["image_dir"],
            "samples": samples,
            "num_seeds": cfg["num_seeds"],
            "bottom_suffix": cfg["bottom_suffix"],
        }, f, indent=2)
    print(f"\nWrote sigma.txt ({meta.effective_sigma:.4f}) and recon_metadata.json under {out_root}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recon_config", type=str, default=None,
                   help="YAML config driving the run (preferred). CLI flags override.")
    # CLI overrides / flat-mode fields
    p.add_argument("--sampler", choices=list(SAMPLERS.keys()), default=None)
    p.add_argument("--sampler_param", action="append", default=None,
                   help="Repeatable KEY=VAL override into sampler.params.")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--image_dir", type=str, default=None)
    p.add_argument("--samples", nargs="+", default=None)
    p.add_argument("--num_seeds", type=int, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--test_origin", type=str, default=None,
                   help="Override sampler's default output dirname segment.")
    p.add_argument("--bottom_suffix", type=str, default=None)
    args = p.parse_args()

    cfg = _resolve_config(args)

    # Validate minimal required fields
    missing = [k for k in ("ckpt", "image_dir", "samples", "sampler") if not cfg.get(k)]
    if missing:
        p.error(f"missing required config fields: {missing}")

    run(cfg)


if __name__ == "__main__":
    main()
