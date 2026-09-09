"""Disposable CUDA worker. Synthetic tensors only; publishes validated profiles."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

from .core import MODES, TARGET_SECONDS, environment, publish
from ..common.data_syncing import resolve_path
from .fla import Catalog, catalog_scope, inventory


def exercise(geometry, catalog, *, length=129, batch=1, packed=False, final_state=False, reference=True):
    import torch
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    h, k, v = (geometry[key] for key in ("heads", "key_dim", "value_dim"))
    q = torch.randn(batch, length, h, k, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(q)
    value = torch.randn(batch, length, h, v, device="cuda", dtype=torch.bfloat16)
    gate = -torch.rand(batch, length, h, device="cuda", dtype=torch.float32) * .2
    beta = torch.rand(batch, length, h, device="cuda", dtype=torch.bfloat16)
    # Exercise padded rows without changing the recurrence between real rows.
    if batch > 1:
        q[-1, -7:] = 0; key[-1, -7:] = 0; value[-1, -7:] = 0
    tensors = [x.requires_grad_(not final_state) for x in (q, key, value, gate, beta)]
    boundaries = [0, 65, 68, length] if packed else [0, length]
    cu = torch.tensor(boundaries, device="cuda", dtype=torch.int32) if packed else None
    with catalog_scope(catalog):
        out, state = chunk_gated_delta_rule(*tensors[:3], g=gate, beta=beta, cu_seqlens=cu,
                                           output_final_state=final_state, use_qk_l2norm_in_kernel=True)
        cotangent = torch.randn_like(out)
        gradients = torch.autograd.grad(out, tensors, cotangent) if not final_state else []
    assert torch.isfinite(out).all() and all(torch.isfinite(x).all() for x in gradients)
    errors = []
    if reference:
        copies = [x.detach().float().requires_grad_(not final_state) for x in tensors]
        qref, kref, vref, gref, bref = copies
        # Native normalization rounds to BF16 before the delta recurrence.
        qref = (qref * torch.rsqrt(qref.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        kref = (kref * torch.rsqrt(kref.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        pieces, states = [], []
        for start, end in zip(boundaries, boundaries[1:]):
            expected, expected_state = naive_recurrent_gated_delta_rule(qref[:, start:end], kref[:, start:end],
                    vref[:, start:end], bref[:, start:end], gref[:, start:end], output_final_state=final_state)
            pieces.append(expected)
            if expected_state is not None:
                states.append(expected_state)
        expected = torch.cat(pieces, dim=1)
        comparisons = [(out, expected)]
        if final_state:
            comparisons.append((state, torch.cat(states)))
        else:
            expected_gradients = torch.autograd.grad(expected, copies, cotangent.float())
            comparisons.extend(zip(gradients, expected_gradients))
        for actual, expected in comparisons:
            relative = float((actual.float() - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8))
            assert relative < .03, (geometry, batch, packed, final_state, relative)
            errors.append(relative)
    torch.cuda.synchronize()
    return max(errors, default=0.0)


def validate(geometry, catalog):
    errors = [exercise(geometry, catalog, length=129),
              exercise(geometry, catalog, length=129, batch=3),
              exercise(geometry, catalog, length=129, batch=7),
              exercise(geometry, catalog, length=129, packed=True),
              exercise(geometry, catalog, length=65, final_state=True)]
    assert catalog.visited == set(inventory()), (catalog.visited, set(inventory()))
    return max(errors)


def main():
    import torch
    request = json.loads(Path(sys.argv[1]).read_text())
    torch.cuda.set_device(request["device"])
    torch.manual_seed(314159)
    started = time.monotonic()
    deadline = started + request["seconds"] - 5
    env = environment(torch.device("cuda", request["device"]))
    for identity in request["identities"]:
        assert identity["environment"] == env, "Parent and worker kernel environments differ"
        geometry = identity["geometry"]
        catalog = Catalog(discover=True)
        error = validate(geometry, catalog)
        for length in (2048, 8192, 32768):
            exercise(geometry, catalog, length=length, reference=False)
        def save(phase, selected, error):
            seconds = time.monotonic() - started
            profile = {"identity": identity, "coverage": MODES,
                       "records": sorted(selected.records.values(), key=lambda r: (r["kernel"], r["key"], r["bucket"])),
                       "validation": {"passed": True, "max_relative_rms_error": error, "batches": [1, 3, 7],
                                      "measured_lengths": [65, 129, 2048, 8192, 32768],
                                      "reuse": "independent rows/batches; performance beyond measured lengths is unmeasured"},
                       "tuning": {"phase": phase, "seconds": round(seconds, 3), "benchmarks": selected.benchmarks,
                                  "extension_reason": "cold compilation and correctness coverage" if seconds > TARGET_SECONDS else None}}
            path = publish(profile, resolve_path("AUTOTUNE/profiles"))
            print(f"Published {phase}: {geometry} in {seconds:.1f}s ({path.name})", flush=True)
        save("validated_baseline", catalog, error)
        # First finish every requested baseline. Improvements cannot consume the
        # budget needed by another geometry's initial usable result.
    for identity in request["identities"]:
        if time.monotonic() >= min(started + TARGET_SECONDS, deadline - 40):
            break
        from .core import lookup
        baseline = lookup(identity)[0]
        geometry = identity["geometry"]
        trial = Catalog(baseline["records"], discover=True, optimize=True, deadline=min(started + TARGET_SECONDS, deadline - 40))
        exercise(geometry, trial, length=2048, reference=False)
        trial.optimize = False
        error = validate(geometry, trial)
        for length in (8192, 32768):
            exercise(geometry, trial, length=length, reference=False)
        save("bounded_search", trial, error)


if __name__ == "__main__":
    main()
