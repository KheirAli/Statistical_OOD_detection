"""Baseline method runners: SimpleNet, DDAD.

Wraps external baseline repos as subprocesses and parses their AUROC output.
"""

import os
import re
import subprocess
import sys
from typing import Dict, Optional


def run_simplenet(cfg: Dict) -> Dict:
    """Run SimpleNet evaluation and extract AUROC.

    Expects cfg keys: repo_dir, results_dir, category.
    """
    if not cfg.get("enabled", False):
        return {}

    repo_dir = cfg["repo_dir"]
    category = cfg["category"]

    cmd = [
        sys.executable, os.path.join(repo_dir, "eval_mvtec_dir.py"),
        "--category", category,
    ]

    print(f"Running SimpleNet for category '{category}'...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=repo_dir, timeout=600,
        )
        output = result.stdout + result.stderr

        # Parse AUROC from output (SimpleNet prints "Image AUROC: X.XX" and "Pixel AUROC: X.XX")
        image_auroc = _extract_float(output, r"[Ii]mage.*?AUROC[:\s]+([0-9.]+)")
        pixel_auroc = _extract_float(output, r"[Pp]ixel.*?AUROC[:\s]+([0-9.]+)")

        return {
            "image_auroc": image_auroc,
            "px_roc_auc": pixel_auroc,
            "raw_output": output[-500:] if len(output) > 500 else output,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"SimpleNet failed: {e}")
        return {"error": str(e)}


def run_ddad(cfg: Dict) -> Dict:
    """Run DDAD detection and extract AUROC.

    Expects cfg keys: repo_dir, config, category, checkpoint_epoch.
    """
    if not cfg.get("enabled", False):
        return {}

    repo_dir = cfg["repo_dir"]
    ddad_config = cfg.get("config", os.path.join(repo_dir, "config.yaml"))
    category = cfg["category"]

    cmd = [
        sys.executable, os.path.join(repo_dir, "main.py"),
        "--config", ddad_config,
        "--mode", "detection",
        "--category", category,
    ]

    print(f"Running DDAD for category '{category}'...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=repo_dir, timeout=600,
        )
        output = result.stdout + result.stderr

        image_auroc = _extract_float(output, r"[Ii]mage.*?AUROC[:\s]+([0-9.]+)")
        pixel_auroc = _extract_float(output, r"[Pp]ixel.*?AUROC[:\s]+([0-9.]+)")

        return {
            "image_auroc": image_auroc,
            "px_roc_auc": pixel_auroc,
            "raw_output": output[-500:] if len(output) > 500 else output,
        }
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"DDAD failed: {e}")
        return {"error": str(e)}


def run_baselines(baselines_cfg: Dict) -> Dict:
    """Dispatch to enabled baselines.

    Returns:
        {"simplenet": {...}, "ddad": {...}} with AUROC results.
    """
    if not baselines_cfg.get("enabled", False):
        return {}

    results = {}

    if baselines_cfg.get("simplenet", {}).get("enabled", False):
        results["simplenet"] = run_simplenet(baselines_cfg["simplenet"])

    if baselines_cfg.get("ddad", {}).get("enabled", False):
        results["ddad"] = run_ddad(baselines_cfg["ddad"])

    return results


def _extract_float(text: str, pattern: str) -> Optional[float]:
    """Extract first float matching regex pattern from text."""
    match = re.search(pattern, text)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            pass
    return None
