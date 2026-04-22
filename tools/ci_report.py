#!/usr/bin/env python3
"""Bootstrap CI report across all cached experiment results.

Scans for result JSONs under results_eval/ and prints per-config mean AUC
with 95% bootstrap CI. Handles three known formats:

  1. evaluate.py sweeps           results_eval/**/sweep_*.json
     → per_sample[sname][metric_key] = {sp_roc_auc, px_roc_auc}
  2. diagnostic_variance_score    results_eval/diagnostic_variance_score/*.json
     → runs[i].per_sample[sname] = {px_auc_raw, px_auc_sigma5}
  3. diagnostic_combined_score    results_eval/diagnostic_combined_score/*.json
     → runs[i].per_sample[j] = {sample, {method}_px_auc_{raw,sigma5}}

Usage:
    python tools/ci_report.py                      # full report
    python tools/ci_report.py --compare            # also run paired comparisons
    python tools/ci_report.py --path <file.json>   # single file
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from tools.ci_stats import bootstrap_mean_ci, format_ci, compare  # noqa: E402


def analyze_sweep(path: Path) -> List[Dict]:
    """evaluate.py sweep output: one metric_key × (SP AUC, Px AUC) per row."""
    data = json.loads(path.read_text())
    per_sample = data.get("per_sample", {})
    sample_names = data.get("sample_names") or list(per_sample.keys())
    if not sample_names:
        return []
    metric_keys = list(per_sample[sample_names[0]].keys())
    out = []
    tag = path.parent.name
    for mk in metric_keys:
        sp_vals = [per_sample[s][mk].get("sp_roc_auc") for s in sample_names
                   if mk in per_sample[s]]
        px_vals = [per_sample[s][mk].get("px_roc_auc") for s in sample_names
                   if mk in per_sample[s]]
        out.append({
            "source": "sweep",
            "tag": tag,
            "metric_key": mk,
            "sp": bootstrap_mean_ci(sp_vals),
            "px": bootstrap_mean_ci(px_vals),
            "sp_raw": sp_vals,
            "px_raw": px_vals,
        })
    return out


def analyze_variance_score(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    out = []
    for run in data.get("runs", []):
        samples = run.get("per_sample", {})
        raws = [v["px_auc_raw"] for v in samples.values()
                if "px_auc_raw" in v]
        s5s = [v["px_auc_sigma5"] for v in samples.values()
               if "px_auc_sigma5" in v]
        out.append({
            "source": "variance_score",
            "tag": run.get("config", "?"),
            "metric_key": "variance_only",
            "px_raw": raws,
            "px_s5_raw": s5s,
            "px_raw_ci": bootstrap_mean_ci(raws),
            "px_s5_ci": bootstrap_mean_ci(s5s),
        })
    return out


def analyze_combined_score(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    out = []
    methods = ["variance", "pmf", "sum", "product", "max", "geometric"]
    for run in data.get("runs", []):
        samples = run.get("per_sample", [])
        for m in methods:
            raws = [s[f"{m}_px_auc_raw"] for s in samples
                    if f"{m}_px_auc_raw" in s]
            s5s = [s[f"{m}_px_auc_sigma5"] for s in samples
                   if f"{m}_px_auc_sigma5" in s]
            out.append({
                "source": "combined_score",
                "tag": run.get("config", "?"),
                "metric_key": m,
                "px_raw": raws,
                "px_s5_raw": s5s,
                "px_raw_ci": bootstrap_mean_ci(raws),
                "px_s5_ci": bootstrap_mean_ci(s5s),
            })
    return out


def print_sweep_table(rows: List[Dict]) -> None:
    if not rows:
        return
    print("\n=== evaluate.py sweep runs (per metric_key) ===")
    print(f"{'tag':<45} {'metric_key':<14} {'N':<4} {'SP AUC (95% CI)':<28} {'Px AUC (95% CI)':<28}")
    print("-" * 123)
    for r in rows:
        sp_mean, sp_lo, sp_hi, n = r["sp"]
        px_mean, px_lo, px_hi, _ = r["px"]
        print(f"{r['tag']:<45} {r['metric_key']:<14} {n:<4} "
              f"{format_ci(sp_mean, sp_lo, sp_hi):<28} "
              f"{format_ci(px_mean, px_lo, px_hi):<28}")


def print_diag_table(rows: List[Dict], source: str) -> None:
    rows = [r for r in rows if r["source"] == source]
    if not rows:
        return
    print(f"\n=== {source} ===")
    print(f"{'config':<24} {'method':<12} {'N':<4} {'Px raw (95% CI)':<28} {'Px σ=5 (95% CI)':<28}")
    print("-" * 100)
    for r in rows:
        px_raw_mean, px_raw_lo, px_raw_hi, n = r["px_raw_ci"]
        px_s5_mean, px_s5_lo, px_s5_hi, _ = r["px_s5_ci"]
        print(f"{r['tag']:<24} {r['metric_key']:<12} {n:<4} "
              f"{format_ci(px_raw_mean, px_raw_lo, px_raw_hi):<28} "
              f"{format_ci(px_s5_mean, px_s5_lo, px_s5_hi):<28}")


def print_paired_comparisons(rows: List[Dict]) -> None:
    """Key paired head-to-heads prompted by today's findings."""
    print("\n=== paired comparisons (same samples, different methods) ===")
    by_key = {(r["tag"], r["metric_key"]): r for r in rows
              if r["source"] in ("variance_score", "combined_score")}

    pairs = []
    # variance_only vs PMF across the 4 configs — when combined-score finishes,
    # both signals are in the combined-score report (methods: variance, pmf).
    for cfg in ["cable_ddad_native", "cable_additive_dps",
                "faces_ddad_native", "faces_additive_dps"]:
        for method in ["sum", "product", "max", "geometric"]:
            r_method = by_key.get((cfg, method))
            r_pmf = by_key.get((cfg, "pmf"))
            r_var = by_key.get((cfg, "variance"))
            if r_method and r_pmf:
                pairs.append((cfg, method, "pmf", r_method, r_pmf))
            if r_method and r_var:
                pairs.append((cfg, method, "variance", r_method, r_var))
        if by_key.get((cfg, "variance")) and by_key.get((cfg, "pmf")):
            pairs.append((cfg, "variance", "pmf",
                          by_key[(cfg, "variance")], by_key[(cfg, "pmf")]))

    if not pairs:
        print("  (not enough data yet — run combined_score to enable)")
        return

    print(f"{'config':<22} {'A':<10} {'vs':<2} {'B':<10} {'Δ mean':<10} "
          f"{'Δ 95% CI':<22} {'straddles 0?'}")
    print("-" * 90)
    for cfg, a_name, b_name, a_row, b_row in pairs:
        result = compare(a_row["px_s5_raw"], b_row["px_s5_raw"], a_name, b_name)
        if "error" in result:
            continue
        lo, hi = result["diff_ci_lo"], result["diff_ci_hi"]
        flag = "yes (noise)" if result["diff_straddles_zero"] else "no"
        print(f"{cfg:<22} {a_name:<10} vs {b_name:<10} "
              f"{result['diff_mean']:+.3f}    "
              f"[{lo:+.3f}, {hi:+.3f}]    {flag}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--path", type=str, default=None,
                   help="Single JSON path. Default: scan results_eval/")
    p.add_argument("--compare", action="store_true",
                   help="Also run paired head-to-head comparisons.")
    args = p.parse_args()

    sweep_rows: List[Dict] = []
    diag_rows: List[Dict] = []

    def ingest(path: Path):
        pn = path.name
        if pn.startswith("sweep_"):
            sweep_rows.extend(analyze_sweep(path))
        elif "variance_score" in str(path):
            diag_rows.extend(analyze_variance_score(path))
        elif "combined_score" in str(path):
            diag_rows.extend(analyze_combined_score(path))

    if args.path:
        ingest(Path(args.path))
    else:
        results_dir = REPO_ROOT / "results_eval"
        if not results_dir.exists():
            print(f"(no results under {results_dir})")
            return
        for path in sorted(results_dir.rglob("*.json")):
            ingest(path)

    print_sweep_table(sweep_rows)
    print_diag_table(diag_rows, "variance_score")
    print_diag_table(diag_rows, "combined_score")

    if args.compare:
        print_paired_comparisons(diag_rows)


if __name__ == "__main__":
    main()
