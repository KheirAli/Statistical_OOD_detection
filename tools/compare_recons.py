"""Side-by-side recon comparison: ours vs grad student's, matched patches.

Produces a single PNG per patch showing: [their recon] | [ours] | [|diff|]
Plus a summary JSON with mean abs-diff per patch.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


THEIRS_ROOT = "/data/akheirandish3/mvtec_ad/results_patches/samples_000"
THEIRS_ORIGIN = "Combined_first_sigma_batched"
OURS_ROOT = "./results_patches/samples_000"
OURS_ORIGIN = "Combined_half_sigma_batched"
OUT_DIR = "./figures_validation"
PATCHES = [3, 5, 6, 7, 8, 9, 10]


def load_recons(root, origin, patch):
    d = Path(root) / f"{origin}_{patch}_4" / "inpainting" / "recon"
    if not d.exists():
        return []
    imgs = []
    for p in sorted(d.glob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        imgs.append((p.name, img))
    return imgs


def load_label(root, origin, patch):
    for candidate in [
        Path(root) / f"{origin}_{patch}_4" / "inpainting" / "label" / "0_00000.png",
        Path(root) / f"{origin}_{patch}_4" / "inpainting" / "label" / "00000.png",
    ]:
        if candidate.exists():
            return np.array(Image.open(candidate).convert("RGB"))
    return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {}

    for patch in PATCHES:
        theirs = load_recons(THEIRS_ROOT, THEIRS_ORIGIN, patch)
        ours = load_recons(OURS_ROOT, OURS_ORIGIN, patch)
        label = load_label(THEIRS_ROOT, THEIRS_ORIGIN, patch)

        if not theirs or not ours:
            print(f"Patch {patch}: theirs={len(theirs)} ours={len(ours)} — skipping")
            summary[patch] = {"status": "missing", "theirs": len(theirs), "ours": len(ours)}
            continue

        n = min(len(theirs), len(ours))
        mean_abs_diffs = []
        fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
        if n == 1:
            axes = axes[None, :]

        for i in range(n):
            t_img = theirs[i][1].astype(np.float32)
            o_img = ours[i][1].astype(np.float32)
            diff = np.abs(t_img - o_img).astype(np.uint8)
            mad = float(np.mean(np.abs(t_img - o_img)))
            mean_abs_diffs.append(mad)

            axes[i, 0].imshow(label) if label is not None else axes[i, 0].text(0.5, 0.5, "no label")
            axes[i, 0].set_title(f"label")
            axes[i, 1].imshow(theirs[i][1])
            axes[i, 1].set_title(f"theirs/{theirs[i][0]}")
            axes[i, 2].imshow(ours[i][1])
            axes[i, 2].set_title(f"ours/{ours[i][0]}")
            axes[i, 3].imshow(diff)
            axes[i, 3].set_title(f"|diff| mean={mad:.2f}")
            for a in axes[i]:
                a.axis("off")

        plt.suptitle(f"Patch {patch} — ours vs theirs (σ=0.1)", fontsize=14)
        plt.tight_layout()
        out_path = Path(OUT_DIR) / f"patch_{patch:02d}.png"
        plt.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close()
        print(f"Patch {patch}: n={n}, mean_abs_diff={np.mean(mean_abs_diffs):.2f} -> {out_path}")

        summary[patch] = {
            "n_compared": n,
            "mean_abs_diff_per_pair": mean_abs_diffs,
            "mean_abs_diff_patch": float(np.mean(mean_abs_diffs)),
        }

    with open(Path(OUT_DIR) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {OUT_DIR}/summary.json")
    print(json.dumps({k: v.get("mean_abs_diff_patch", "missing") for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
