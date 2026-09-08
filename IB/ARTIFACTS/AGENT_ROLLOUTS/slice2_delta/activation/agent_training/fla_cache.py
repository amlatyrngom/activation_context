"""
fla's Triton kernels autotune at first use, and two of them (the l2norm kernels) key that on a block
count derived from the sequence length, so a training process re-benchmarks them for every new length
bucket: about 14 s each, several times per round, on every process (measured: 51 s of a 75 s step).
fla can instead load tuned configs from `{kernel_name}.json` files with fuzzy numeric matching
(`FLA_CACHE_MODE=fuzzy`), but ships none for this GPU and never writes them. `dump_fla_configs` writes
them from the live autotuners after one tuning pass; `configure_fla_cache` points fla at the files for
the current GPU (under `fla_configs/<sanitized gpu name>/`, checked in per GPU type).
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import torch

CONFIG_ROOT = Path(__file__).parent / "fla_configs"


def gpu_config_dir(device) -> Path | None:
    """The config folder for this device's GPU type (fla's own sanitized name), or None off the GPU."""
    if getattr(device, "type", None) != "cuda":
        return None
    try:
        from fla.ops.utils.cache import get_gpu_info
    except ImportError:
        return None
    return CONFIG_ROOT / get_gpu_info()


def configure_fla_cache(device, config_dir: Path | None = None) -> Path | None:
    """
    Fuzzy config loading from the GPU's folder when it holds files; returns the folder used or None.
    fla reads its mode at import, so the module global is set alongside the environment variables.
    """
    config_dir = config_dir if config_dir is not None else gpu_config_dir(device)
    if config_dir is None or not any(config_dir.glob("*.json")):
        return None
    try:
        import fla.ops.utils.cache as fla_cache
    except ImportError:
        return None
    os.environ["FLA_CONFIG_DIR"] = str(config_dir)
    os.environ["FLA_CACHE_MODE"] = "fuzzy"
    fla_cache.FLA_CACHE_MODE = fla_cache.FlaCacheMode.FUZZY
    return config_dir


def live_autotuners() -> list:
    from fla.ops.utils.cache import CachedAutotuner
    return [obj for obj in gc.get_objects() if isinstance(obj, CachedAutotuner) and getattr(obj, "cache", None)]


def dump_fla_configs(config_dir: Path) -> dict[str, int]:
    """Write `{kernel_name}.json` for every fla autotuner that has tuned something; returns entries per kernel."""
    import triton
    from fla.ops.utils.cache import AutotuneKey
    config_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for tuner in live_autotuners():
        entries = {}
        for key, cfg in tuner.cache.items():
            autotune_key = [value if isinstance(value, (int, float, str, bool)) or value is None else str(value) for value in key]
            entries[AutotuneKey.key_hash(autotune_key)] = {
                "autotune_key": autotune_key,
                "config": {
                    "kwargs": dict(cfg.kwargs), "num_warps": cfg.num_warps, "num_stages": cfg.num_stages,
                    "num_ctas": getattr(cfg, "num_ctas", 1), "maxnreg": getattr(cfg, "maxnreg", None),
                },
            }
        if not entries:
            continue
        path = config_dir / f"{tuner.kernel_name}.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text()).get("autotune_entries") or {}
            except (OSError, ValueError):
                existing = {}
        merged = {**existing, **entries}
        most_common = max(merged.values(), key=lambda entry: sum(1 for other in merged.values() if other["config"] == entry["config"]))
        path.write_text(json.dumps({
            "kernel_name": tuner.kernel_name, "triton_version": triton.__version__,
            "autotune_entries": merged, "default_config": most_common["config"],
        }, indent=1))
        written[tuner.kernel_name] = len(merged)
    return written
