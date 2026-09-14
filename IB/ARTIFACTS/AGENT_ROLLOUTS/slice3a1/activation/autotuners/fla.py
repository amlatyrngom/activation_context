"""Pinned FLA bridge: legal native Config objects, exact semantics, no fuzzy miss."""
from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
import json
from pathlib import Path
import threading
from types import MethodType

from .core import AutotuningUnavailable, digest

KERNELS = {
    "modules.l2norm": ["l2norm_fwd_kernel", "l2norm_bwd_kernel"],
    "ops.utils.cumsum": ["chunk_local_cumsum_scalar_kernel"],
    "ops.gated_delta_rule.chunk_fwd": ["chunk_gated_delta_rule_fwd_kkt_solve_kernel"],
    "ops.gated_delta_rule.wy_fast": ["recompute_w_u_fwd_kernel", "prepare_wy_repr_bwd_kernel"],
    "ops.common.chunk_delta_h": ["chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64"],
    "ops.common.chunk_o": ["chunk_fwd_kernel_o", "chunk_bwd_kernel_dv_local", "chunk_bwd_kernel_dqkwg"],
}
_scope_lock = threading.RLock()
_run_lock = threading.RLock()


def inventory() -> dict:
    from triton.runtime.autotuner import Autotuner
    result = {}
    for module, names in KERNELS.items():
        loaded = import_module("fla." + module)
        for name in names:
            tuner = getattr(loaded, name)
            while not isinstance(tuner, Autotuner):
                tuner = tuner.fn
            result[name] = tuner
    return result


def config_values(config) -> dict:
    return {"kwargs": config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages,
            "num_ctas": config.num_ctas, "maxnreg": config.maxnreg}


def validate_catalog(profile: dict) -> bool:
    tuners = inventory()
    try:
        records = profile["records"]
        if {record["kernel"] for record in records} != set(tuners):
            return False
        return all(record["config"] in [config_values(c) for c in tuners[record["kernel"]].configs]
                   and record["semantic"] and record["key"] == digest(record["semantic"]) for record in records)
    except (KeyError, TypeError, ValueError):
        return False


def call_key(name, tuner, args, kwargs):
    from fla.ops.utils.cache import AutotuneKey
    named = {**dict(zip(tuner.arg_names, args)), **kwargs}
    native = AutotuneKey.build(tuner.arg_names, tuner.keys, args, kwargs).autotune_key
    semantic = {}
    dimensions = {"D", "BD", "H", "HV", "K", "V", "BT", "BC", "BK", "BV", "eps", "scale"}
    for key in tuner.arg_names:
        if key not in named:
            continue
        value = named[key]
        if key == "NB" and name.startswith("l2norm_"):
            continue  # NB is unused by the pinned kernel bodies; T masks independent rows.
        if key == "B" and name == "chunk_local_cumsum_scalar_kernel" and not named["IS_VARLEN"]:
            continue  # launch replication only; all head/layout/flag dimensions stay exact.
        if hasattr(value, "dtype"):
            semantic[key] = {"dtype": str(value.dtype), "last_stride": value.stride(-1), "ndim": value.ndim}
        elif key in tuner.keys or key in dimensions or isinstance(value, bool) or value is None:
            semantic[key] = value
    return tuple(native), semantic, int(named.get("NB", 1)), named


def seed_candidate(name, legal):
    """Old files are preference seeds only; they never constitute a cache hit."""
    import torch
    folder = Path(__file__).parent / "configs" / "legacy"
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    path = folder / gpu / f"{name}.json"
    try:
        preferred = json.loads(path.read_text())["default_config"]
        for config in legal:
            if config_values(config) == preferred:
                return config
    except (OSError, ValueError, KeyError):
        pass
    return min(legal, key=lambda c: (abs(c.num_warps - 4), abs(c.num_stages - 2), abs(c.kwargs.get("BT", 32) - 32)))


class Catalog:
    """Worker may discover/search records; runtime may only consume them."""
    def __init__(self, records=(), *, discover=False, optimize=False, deadline=float("inf")):
        self.records = {(r["kernel"], r["key"], r["bucket"]): dict(r) for r in records}
        self.discover = discover
        self.optimize = optimize
        self.deadline = deadline
        self.visited = set()
        self.optimized = set()
        self.benchmarks = 0

    def run(self, name, tuner, *args, **kwargs):
        from triton.runtime.autotuner import Autotuner
        import time
        native, semantic, bucket, named = call_key(name, tuner, args, kwargs)
        key = digest(semantic)
        self.visited.add(name)
        previous = getattr(tuner, "nargs", None)
        tuner.nargs = dict(zip(tuner.arg_names, args))
        try:
            legal = tuner.prune_configs(kwargs)
            matching = [r for (kernel, signature, _), r in self.records.items() if kernel == name and signature == key]
            record = min(matching, key=lambda r: abs(r["bucket"] - bucket)) if matching else None
            chosen = next((c for c in legal if record and config_values(c) == record["config"]), None)
            if chosen is None:
                if not self.discover or not legal:
                    raise AutotuningUnavailable(f"Uncovered/illegal FLA configuration: {name} {semantic}")
                chosen = seed_candidate(name, legal)
            if self.optimize and (name, key, bucket) not in self.optimized and time.monotonic() < self.deadline:
                from triton.testing import do_bench
                candidates = [chosen] + [c for c in legal if c is not chosen][:3]
                old_bench = tuner.__dict__.get("do_bench")
                had_bench = "do_bench" in tuner.__dict__
                timings = []
                try:
                    tuner.do_bench = lambda fn, quantiles: do_bench(fn, warmup=2, rep=5, quantiles=quantiles)
                    for candidate in candidates:
                        if time.monotonic() >= self.deadline:
                            break
                        timing = tuner._bench(*args, config=candidate, **kwargs)[0]
                        self.benchmarks += 1
                        timings.append((timing, candidate))
                    if timings:
                        best, chosen = min(timings, key=lambda pair: pair[0])
                        if best == float("inf"):
                            raise AutotuningUnavailable(f"No legal candidate launched for {name}")
                finally:
                    tuner.pre_hook({**named, **chosen.all_kwargs()}, reset_only=True)
                    if had_bench:
                        tuner.do_bench = old_bench
                    else:
                        tuner.__dict__.pop("do_bench", None)
                self.optimized.add((name, key, bucket))
            # Native keys omit several flags. Seed every call, bypassing FLA's
            # loader entirely, even when another semantic mode populated cache.
            from triton.runtime.errors import OutOfResources
            from triton.compiler.errors import CompileTimeAssertionFailure
            from triton.backends.nvidia.compiler import PTXASError
            candidates = [chosen] + ([c for c in legal if c is not chosen][:3] if self.discover else [])
            for index, candidate in enumerate(candidates):
                tuner.cache[native] = candidate
                try:
                    result = Autotuner.run(tuner, *args, **kwargs)
                except (OutOfResources, CompileTimeAssertionFailure, PTXASError):
                    if index + 1 == len(candidates):
                        raise
                    tuner.nargs = dict(zip(tuner.arg_names, args))
                    tuner.pre_hook({**named, **candidate.all_kwargs()}, reset_only=True)
                    continue
                if self.discover:
                    self.records[name, key, bucket] = {"kernel": name, "key": key, "semantic": semantic,
                                                      "bucket": bucket, "config": config_values(candidate)}
                return result
        finally:
            tuner.nargs = previous


@contextmanager
def catalog_scope(catalog: Catalog):
    """Per-object bridge; original Config hooks and FLA reset/restore hooks survive."""
    with _scope_lock:
        saved = []
        try:
            for name, tuner in inventory().items():
                saved.append((tuner, tuner.__dict__.get("run"), tuner.cache, getattr(tuner, "nargs", None)))
                tuner.cache = {}
                def run(instance, *args, _name=name, **kwargs):
                    # Autograd can call Python backward on a different thread
                    # while the scope owner waits. Never acquire its scope lock.
                    with _run_lock:
                        return catalog.run(_name, instance, *args, **kwargs)
                tuner.run = MethodType(run, tuner)
            yield catalog
        finally:
            for tuner, run, cache, nargs in reversed(saved):
                if run is None:
                    tuner.__dict__.pop("run", None)
                else:
                    tuner.run = run
                tuner.cache, tuner.nargs = cache, nargs


@contextmanager
def configure_fla_runtime(bundle: dict | None):
    if bundle is None:
        yield
        return
    import torch
    with torch.cuda.device(bundle["device_index"]):
        from .core import environment
        current = environment(torch.device("cuda", bundle["device_index"]))
        if any(profile["identity"]["environment"] != current for profile in bundle["profiles"]):
            raise AutotuningUnavailable("FLA catalog belongs to a different hardware/software environment")
        if not all(validate_catalog(profile) for profile in bundle["profiles"]):
            raise AutotuningUnavailable("FLA catalog no longer matches current legal kernel candidates")
        records = [record for profile in bundle["profiles"] for record in profile["records"]]
        with catalog_scope(Catalog(records)):
            yield
