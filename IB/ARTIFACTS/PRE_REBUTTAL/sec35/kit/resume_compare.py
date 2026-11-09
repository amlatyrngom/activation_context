"""Compare the resume test's runs (jobs/resume_test/resume_test.sh): per-update loss and final pair tensors (reader LoRA, side
LoRA, AC modules), A-B (GPU run-to-run noise) vs A-C (C resumed from the slot of update K). Prints JSON with a verdict.
Usage: python resume_compare.py TEST_DIR"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

LOSS = re.compile(r"global (\d+): sft loss ([0-9.]+)")
STEPS, K = 48, 24


def losses(log: Path) -> dict:
    return {int(m.group(1)): float(m.group(2)) for m in LOSS.finditer(log.read_text())} if log.exists() else {}


def flatten(prefix, value, out):
    if isinstance(value, torch.Tensor):
        out[prefix] = value.float()
    elif isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}", v, out)


def tensors(pair: Path) -> dict:
    out = {}
    for f in sorted(pair.rglob("*")):
        if f.suffix == ".safetensors":
            for k, v in load_file(str(f)).items():
                out[f"{f.relative_to(pair)}:{k}"] = v.float()
        elif f.name == "ac_modules.pt":
            flatten("ac_modules", torch.load(f, map_location="cpu", weights_only=False), out)
    return out


def diff(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    groups = {}
    for k in keys:
        g = "ac_modules" if k.startswith("ac_modules") else ("side_lora" if "side_lora" in k else "reader")
        d = groups.setdefault(g, {"n": 0, "num": 0.0, "den": 0.0, "max_abs": 0.0})
        delta = a[k] - b[k]
        d["n"] += 1; d["num"] += (delta ** 2).sum().item(); d["den"] += (a[k] ** 2).sum().item(); d["max_abs"] = max(d["max_abs"], delta.abs().max().item())
    return {"only_one_side": sorted(set(a) ^ set(b))[:5],
            **{g: {"tensors": d["n"], "max_abs": d["max_abs"], "rel_l2": (d["num"] ** 0.5) / (d["den"] ** 0.5) if d["den"] else None} for g, d in groups.items()}}


def main():
    root = Path(sys.argv[1]); result, problems = {}, []
    for case in ("recon", "task"):
        c = root / case; r = {}
        LA, LB = losses(c / "A" / "train.log"), losses(c / "B" / "train.log")
        a1, a2 = losses(c / "C" / "train_a1.log"), losses(c / "C" / "train_a2.log")
        r["a1_last_update"], r["a2_updates"] = max(a1, default=None), sorted(a2)[:1] + sorted(a2)[-1:]
        LC = {u: v for u, v in a1.items() if u <= K} | a2
        if not a1 or max(a1) >= STEPS: problems.append(f"{case}: C's first attempt was not killed before the end")
        if sorted(a2) != list(range(K + 1, STEPS + 1)): problems.append(f"{case}: C's resumed attempt trained updates {sorted(a2)[:3]}..{sorted(a2)[-3:]} (expected {K + 1}..{STEPS})")
        for x, y, L1, L2 in (("A", "B", LA, LB), ("A", "C", LA, LC)):
            common = sorted(set(L1) & set(L2))
            d = [abs(L1[u] - L2[u]) for u in common]
            post = [abs(L1[u] - L2[u]) for u in common if u > K]
            r[f"loss_{x}{y}"] = {"updates": len(common), "max_abs": max(d, default=None), "after_K_max_abs": max(post, default=None),
                                 "after_K_mean_abs": sum(post) / len(post) if post else None}
        P = {n: tensors(c / n / "sync" / "ckpt" / "final") for n in "ABC" if (c / n / "sync" / "ckpt" / "final").exists()}
        if set(P) != set("ABC"): problems.append(f"{case}: final pair missing for {sorted(set('ABC') - set(P))}")
        else:
            r["pair_AB"], r["pair_AC"] = diff(P["A"], P["B"]), diff(P["A"], P["C"])
            for g in ("ac_modules", "side_lora", "reader"):
                if g not in r["pair_AC"]: problems.append(f"{case}: no {g} tensors compared")
        for n in "ABC":
            st = c / n / "sync" / f"RT_{case}.stats.json"
            r[f"stats_{n}"] = {k: json.loads(st.read_text()).get(k) for k in ("stop_reason", "valid", "resumed_from_update", "resumed_optimizer", "order_matches")} if st.exists() else None
        sc = r["stats_C"] or {}
        if sc.get("resumed_from_update") != K or not sc.get("resumed_optimizer") or not sc.get("valid"):
            problems.append(f"{case}: C stats {sc}")
        ob = c / "opt_bytes.txt"
        r["optimizer_pt_gb"] = round(int(ob.read_text().split()[0]) / 2**30, 2) if ob.exists() else None
        r["optimizer_pt_left_in_C"] = [str(p) for p in (c / "C" / "sync").rglob("optimizer.pt")]
        # the verdict on numbers: A-C after K must be within the A-B noise (2x margin, plus an absolute floor for bit-exact noise)
        ab, ac = r["loss_AB"]["after_K_max_abs"], r["loss_AC"]["after_K_max_abs"]
        if ab is not None and ac is not None and ac > max(2 * ab, 5e-4): problems.append(f"{case}: loss A-C after K {ac:.5f} > noise A-B {ab:.5f}")
        if "pair_AC" in r:
            for g, d in r["pair_AC"].items():
                if g == "only_one_side": continue
                nb = r["pair_AB"].get(g, {}).get("rel_l2") or 0.0
                if d["rel_l2"] is not None and d["rel_l2"] > max(2 * nb, 1e-6): problems.append(f"{case}: {g} weights A-C rel_l2 {d['rel_l2']:.2e} > noise A-B {nb:.2e}")
        result[case] = r
    result["problems"] = problems
    result["verdict"] = "PASS" if not problems else "FAIL"
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
