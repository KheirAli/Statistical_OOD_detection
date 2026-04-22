#!/usr/bin/env python3
"""Bootstrap confidence intervals for per-sample AUC means.

Core helpers used by ci_report.py and callable from notebooks. Treats each
image's AUC as one observation; resamples images with replacement; reports
mean + percentile CI.

For N=10-11 (faces/cable), expect wide CIs — that's honest, not a bug.
"""
from typing import List, Tuple

import numpy as np


def bootstrap_mean_ci(
    xs: List[float],
    B: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """Return (mean, ci_lo, ci_hi, n_valid) for `xs`.

    NaNs are dropped. If n_valid < 2, returns the single value with lo=hi=nan.
    """
    arr = np.asarray([x for x in xs if x is not None and not np.isnan(x)],
                     dtype=np.float64)
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(arr.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), n
    rng = np.random.default_rng(seed=seed)
    idx = rng.integers(0, n, size=(B, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return mean, lo, hi, n


def format_ci(mean: float, lo: float, hi: float) -> str:
    """Standard formatting for tables: `0.869 [0.784, 0.943]`."""
    if np.isnan(mean):
        return "—"
    if np.isnan(lo) or np.isnan(hi):
        return f"{mean:.3f} [N<2]"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def ci_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """Do two CIs overlap? If yes, the difference is plausibly noise."""
    return not (a_hi < b_lo or b_hi < a_lo)


def compare(
    xs_a: List[float], xs_b: List[float], label_a: str, label_b: str,
    B: int = 5000,
) -> dict:
    """Compare two paired-by-index AUC lists. Reports per-run mean + CI and
    the bootstrap CI on the paired difference (xs_a - xs_b).

    Paired bootstrap: for N=N_a=N_b, resample indices and compute mean diff
    on the same resampled subset. Narrower CIs than comparing independent
    bootstraps because image-level variation cancels.
    """
    a = np.asarray(xs_a, dtype=np.float64)
    b = np.asarray(xs_b, dtype=np.float64)
    if a.size != b.size:
        return {"error": f"size mismatch a={a.size} b={b.size}"}
    mask = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[mask], b[mask]
    if a.size < 2:
        return {"error": f"only {a.size} paired samples"}
    d = a - b
    rng = np.random.default_rng(seed=42)
    idx = rng.integers(0, a.size, size=(B, a.size))
    d_means = d[idx].mean(axis=1)
    return {
        "n": int(a.size),
        "label_a": label_a,
        "a_mean": float(a.mean()),
        "label_b": label_b,
        "b_mean": float(b.mean()),
        "diff_mean": float(d.mean()),
        "diff_ci_lo": float(np.percentile(d_means, 2.5)),
        "diff_ci_hi": float(np.percentile(d_means, 97.5)),
        "diff_straddles_zero": bool(
            np.percentile(d_means, 2.5) < 0 < np.percentile(d_means, 97.5)
        ),
    }
