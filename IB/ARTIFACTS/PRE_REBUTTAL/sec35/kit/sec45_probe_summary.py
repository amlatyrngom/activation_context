"""
P0 probe summary: reads the probe runs of one mode (sec45_train.py stats files + the sec45_eval.py decode probe) and proposes the
run sizes for that mode, keyed by the GPU name the probe ran on. The parent turns the three summaries into the job scripts
(jobs/make_jobs.py --probes ...).

  python sec45_probe_summary.py --mode MODE --dir PROBE_SYNC_ROOT [--budget-hours 5.5] [--plan-fraction 0.85] [--slurm-hours 6]

Plan (review B2/B3): every R/S/T job must END within 5.5 h on this GPU, evals included, so the same job fits CSAIL quanta-h200
and ORCD mit_normal_gpu (6 h limit). The expected end is planned at <= 85 % of the 5.5 h; the trainer's --seconds limit is
only a backstop (a run that hits it is marked invalid).
  per update   u = effective seconds per update of the chosen micro-batch: (training call - optimizer placement - first held panel)
                   / updates of the probe; it includes the no-context reference passes, the step overhead and the probe's
                   first 3 updates on the longest items (pessimistic)
  per item     b = item construction + the trainer's example building (measured per item); held panel P scaled to the full panel
  R            expected(N) = startup + b (8N + 500) + P (1 + N/250) + u N + 6 saves + 300 s       (u includes the no-context reference pass: penalty 1.0 for R too)
  S / T        expected(N) = startup + b (8N + 200) + P (1 + N/250) + u N + 4 saves + 4 evals + 300 s  (T: step-0 eval + 25/50/100 %)
               N rounded down to a multiple of 4; S runs the same N with one eval fewer
  N            the largest N with expected(N) <= plan-fraction x budget, capped by the pool (60,000 / 8; 90,447 / 8)
  ckpt_every   R: about one hour of updates; S/T: N / 4
  seconds      the backstop for the training call: slurm limit - 15 min - startup - (task: 4 evals) - 300 s
  micro-batch  the fastest one that completed its probe, longest items included, with peak memory <= 90 % of the GPU's
"""
import argparse
import json
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def eligible(stats: dict) -> str | None:
    """None when the probe run qualifies, else the reason it does not."""
    t = stats.get("timing", {})
    if stats.get("stop_reason") != "complete":
        return f"stop_reason {stats.get('stop_reason')}"
    if not t.get("longest_first_updates"):
        return "did not run the longest items"
    if not t.get("effective_seconds_per_update"):
        return "no timing"
    if t.get("peak_gb") and t.get("gpu_total_gb") and t["peak_gb"] > 0.9 * t["gpu_total_gb"]:
        return f"peak {t['peak_gb']} GB > 90 % of {t['gpu_total_gb']} GB"
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True); p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--budget-hours", type=float, default=5.5); p.add_argument("--plan-fraction", type=float, default=0.85)
    p.add_argument("--slurm-hours", type=float, default=6.0)
    p.add_argument("--recon-pool", type=int, default=60000); p.add_argument("--task-pool", type=int, default=90447)
    p.add_argument("--recon-held", type=int, default=500); p.add_argument("--task-held", type=int, default=200)
    p.add_argument("--report-every", type=int, default=250); p.add_argument("--epu", type=int, default=8)
    a = p.parse_args()
    found = {path.name.removesuffix(".stats.json"): load(path) for path in sorted(a.dir.rglob(f"P0_{a.mode}_*.stats.json"))}
    expected_runs = [f"P0_{a.mode}_recon_mb{m}" for m in (1, 2)] + [f"P0_{a.mode}_task_mb{m}" for m in (1, 2, 4)]
    evaluation = load(a.dir / "eval" / "probe_decode.summary.json")
    target = a.plan_fraction * a.budget_hours * 3600
    devices = {s["timing"].get("device") for s in found.values() if s}
    out = {"mode": a.mode, "gpu": devices.pop() if len(devices) == 1 else sorted(map(str, devices)), "plan": {"budget_seconds": a.budget_hours * 3600,
           "plan_fraction": a.plan_fraction, "target_seconds": round(target), "slurm_hours": a.slurm_hours},
           "probes": {name: ({"eligible": eligible(found[name]) or "yes"} | found[name]["timing"] if found.get(name) else "missing (failed: see its log; OOM?)")
                      for name in expected_runs},
           "eval_probe": evaluation, "proposal": {}, "problems": []}
    if isinstance(out["gpu"], list):
        out["problems"].append(f"probe runs on different devices {out['gpu']}")
    save_seconds = max([60.0] + [s["timing"].get("checkpoint_seconds", 0) for s in found.values() if s])
    pair_gb = max([0.0] + [s["timing"].get("pair_gb") or 0 for s in found.values() if s]) or None
    for kind, pool, held in (("recon", a.recon_pool, a.recon_held), ("task", a.task_pool, a.task_held)):
        runs = {name: s for name, s in found.items() if s and f"_{kind}_" in name and eligible(s) is None}
        if not runs:
            out["problems"].append(f"{kind}: no eligible probe run")
            continue
        name, best = min(runs.items(), key=lambda kv: kv[1]["timing"]["effective_seconds_per_update"])
        t = best["timing"]
        u = t["effective_seconds_per_update"]
        b = t["item_seconds_per_item"] + t["example_seconds_per_item"]
        panel = t["baseline_panel_seconds"] * held / max(1, t["held_items"])
        startup = t["startup_seconds"]
        if kind == "recon":
            fixed = startup + b * held + panel + 6 * save_seconds + 300
            evals, eval_full = 0, 0.0
        else:
            if not evaluation or not evaluation.get("n"):
                out["problems"].append("task: no decode probe; eval time assumed 900 s each")
                eval_full = 900.0
            else:
                decode = (evaluation.get("decode") or {}).get("decode_seconds") or 0.0
                eval_full = (evaluation["seconds"] - decode) + decode * held / evaluation["n"]
            evals = 4
            fixed = startup + b * held + panel + 4 * save_seconds + evals * eval_full + 300
        per = u + a.epu * b + panel / a.report_every
        steps = min(int((target - fixed) / per), pool // a.epu)
        if kind == "task":
            steps = steps // 4 * 4
        if steps <= 0:
            out["problems"].append(f"{kind}: no update fits the plan")
            continue
        expected = fixed + steps * per
        backstop = a.slurm_hours * 3600 - 900 - startup - evals * eval_full - 300
        entry = {"steps": steps, "items": steps * a.epu, "micro_batch": t["micro_batch"],
                 "ckpt_every": max(1, round(3600 / per)) if kind == "recon" else steps // 4, "seconds": int(backstop),
                 "expected_seconds": round(expected), "expected_fraction_of_budget": round(expected / (a.budget_hours * 3600), 3),
                 "from_probe": name, "effective_seconds_per_update": u, "build_seconds_per_item": round(b, 5),
                 "held_panel_seconds_full": round(panel), "startup_seconds": round(startup), "save_seconds": save_seconds,
                 "eval_seconds_each": round(eval_full) if kind == "task" else None, "peak_gb": t.get("peak_gb"), "gpu_total_gb": t.get("gpu_total_gb"),
                 "pair_gb": pair_gb, "capped_by_pool": steps == pool // a.epu}
        out["proposal"]["R" if kind == "recon" else "S_T"] = entry
    print(json.dumps(out, indent=1))
    (a.dir / f"probe_summary_{a.mode}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
