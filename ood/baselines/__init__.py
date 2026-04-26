"""Anomaly-detection baselines with a common interface.

All baselines satisfy the `Baseline` ABC in `base.py`. Each is registered
in `BASELINES` and built via `build_baseline(name, config)`.

Adding a new baseline:
  1. Create `ood/baselines/<name>.py` with a subclass of `Baseline`.
  2. Import it here and add to the `BASELINES` dict.
  3. (Optional) Add a smoke test under `tests/test_baselines.py`.

The CLI driver `tools/baselines_eval.py` uses `build_baseline()` to
instantiate the right subclass from a YAML config.
"""
from .base import Baseline, BaselineMetadata

# Concrete baselines are imported lazily inside build_baseline() so that
# their (sometimes heavy / repo-specific) deps don't pollute consumers
# that only need the ABC.

BASELINES = (
    "mdps",
    "simplenet",
    "supersimplenet",
    "patchcore",
    "draem",
)


def build_baseline(name: str, config: dict) -> Baseline:
    """Instantiate a baseline by name with `config`.

    Args:
        name: key in `BASELINES`.
        config: baseline-specific dict; see the concrete subclass for
            accepted keys.

    Returns:
        Configured `Baseline` instance, ready to call `.score()`.
    """
    if name not in BASELINES:
        raise ValueError(
            f"Unknown baseline {name!r}. Available: {sorted(BASELINES)}"
        )
    if name == "mdps":
        from .mdps import MDPSBaseline
        return MDPSBaseline(config)
    if name == "simplenet":
        from .simplenet import SimpleNetBaseline
        return SimpleNetBaseline(config)
    if name == "supersimplenet":
        from .supersimplenet import SuperSimpleNetBaseline
        return SuperSimpleNetBaseline(config)
    if name == "patchcore":
        from .patchcore import PatchCoreBaseline
        return PatchCoreBaseline(config)
    if name == "draem":
        from .draem import DRAEMBaseline
        return DRAEMBaseline(config)
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "Baseline", "BaselineMetadata",
    "BASELINES", "build_baseline",
]
