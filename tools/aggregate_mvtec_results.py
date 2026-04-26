"""Combine the per-baseline CSVs from `eval_baseline_mvtec.py` into one
master table, in both CSV and Markdown form.

Reads:
    <root>/<baseline>_per_category.csv  for each baseline in --baselines

Writes:
    <root>/mvtec_summary.csv             pivot: rows=category, cols=metric*baseline
    <root>/mvtec_summary.md              human-readable Markdown table

Default metric: pixel AUROC at σ=5 smoothing. Override with --metric.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def _read_csv(path: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["category"]] = row
    return out


def _fmt(s: str) -> str:
    if not s:
        return "—"
    try:
        return f"{float(s):.4f}"
    except ValueError:
        return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./results_eval/mvtec_full")
    p.add_argument("--baselines", nargs="+",
                   default=["simplenet", "supersimplenet", "cutpaste"])
    p.add_argument("--metric", default="px_auc_smooth5",
                   choices=("px_auc_raw", "px_auc_smooth5",
                            "px_ap_smooth5", "px_snr_smooth5"))
    args = p.parse_args()

    root = Path(args.root)
    rows = {b: _read_csv(root / f"{b}_per_category.csv") for b in args.baselines}

    # Master CSV
    out_csv = root / "mvtec_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category"] + [f"{b}_{args.metric}" for b in args.baselines]
                   + [f"{b}_n" for b in args.baselines])
        for cat in MVTEC_CATEGORIES + ["MEAN_ACROSS_CATEGORIES"]:
            line = [cat]
            for b in args.baselines:
                v = rows[b].get(cat, {}).get(args.metric, "")
                line.append(v)
            for b in args.baselines:
                v = rows[b].get(cat, {}).get("n_samples", "")
                line.append(v)
            w.writerow(line)
    print(f"wrote {out_csv}")

    # Markdown table
    md_lines: List[str] = []
    md_lines.append(f"# MVTec-AD baseline pixel AUROC (σ=5, per-image-avg over defective)\n")
    md_lines.append(
        f"Metric: `{args.metric}`. n = number of defective test images per category. "
        f"Source CSVs: `{root}`.\n"
    )
    header = ["category", "n"] + args.baselines
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(":---:" for _ in header) + "|")
    for cat in MVTEC_CATEGORIES:
        # use any baseline's n_samples (they should match)
        ns = ""
        for b in args.baselines:
            n = rows[b].get(cat, {}).get("n_samples", "")
            if n:
                ns = n
                break
        line = [f"**{cat}**", ns]
        for b in args.baselines:
            line.append(_fmt(rows[b].get(cat, {}).get(args.metric, "")))
        md_lines.append("| " + " | ".join(line) + " |")
    # Mean row
    line = ["**MEAN**", ""]
    for b in args.baselines:
        line.append(_fmt(rows[b].get("MEAN_ACROSS_CATEGORIES", {}).get(args.metric, "")))
    md_lines.append("| " + " | ".join(line) + " |")

    out_md = root / "mvtec_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"wrote {out_md}")
    print("\n" + "\n".join(md_lines))


if __name__ == "__main__":
    main()
