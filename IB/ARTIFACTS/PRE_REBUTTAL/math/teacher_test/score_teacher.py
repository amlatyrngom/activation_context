"""
Math teacher test (2026-11-15), step 3 (CPU, this host): score the arms and write the result JSON.

Scoring exactly as the paper table (IB/ARTIFACTS/PAPER_TRAJ_QA/scripts/tqa_unified_table_sumres.py):
  - answer = score_synth_lenient.readings("deepmath", text)["lenient"] (strict reported alongside);
  - a length stop (finish "length") or a missing row counts unsolved;
  - right = score_synth1call.score (normalised match, then math-verify, against the gold answer and its aliases) >= 1.
Solver references on the same problems (no generation): single solver (mean of the three 9B runs, 40k rule), majority vote
(the answer two runs share, else a uniformly random run's answer: expected value; the paired tests use one draw seeded by task id)
and any of 3.
Per arm: accuracy = mean over (task, repeat); per-repeat accuracies; a paired bootstrap 95 % CI of the difference to the
reference arm (resampling tasks, each task's mean over its repeats, 10,000 draws, seed 0); McNemar-style counts over task x repeat
pairs (repeat i against the reference's repeat i) with the exact two-sided sign test; the breakdown by k (solvers right of 3);
length stops, unparsed replies and generated-token quantiles. Wall-clock from timing.jsonl.

    .venv/bin/python IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test/score_teacher.py --gen-dir IB/TMP/PRE_REBUTTAL/math_teacher_test/out --out result.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "IB/TMP/PAPER_COMPACTION/kit"))
sys.path.insert(0, str(ROOT / "IB/TMP/PAPER_TRAJ_QA/kit"))
import score_synth_lenient as lenient          # noqa: E402
from score_synth1call import score, sign_test  # noqa: E402

SET = ROOT / "IB/TMP/PRE_REBUTTAL/inventory_math/teacher_set_m1v2_hard200.json"
PROMPTS = ROOT / "IB/TMP/PRE_REBUTTAL/math_teacher_test/prompts.jsonl"


def quantile(values: list[float], q: float) -> float | None:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=Path, required=True)
    parser.add_argument("--arms", default="9b_results,9b_full,27b_full,27b_results")
    parser.add_argument("--reference", default="9b_results")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--boot", type=int, default=10_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    teacher_set = json.loads(SET.read_text())
    tasks = teacher_set["task_ids"]
    meta = {}
    for line in open(PROMPTS):
        row = json.loads(line)
        if row["condition"] == "results":
            meta[row["task_id"]] = row
    # solver references
    refs: dict = {}
    for t in tasks:
        task, answers = meta[t]["dataset_task"], meta[t]["solver_results"]
        solved = [score(task, a) >= 1.0 for a in answers]
        assert solved == teacher_set["per_task"][t]["solver_right"], t
        majority = lenient.majority_answer("deepmath", answers)
        strict = majority is not None and score(task, majority) >= 1.0
        valid = [ok for answer, ok in zip(answers, solved) if isinstance(answer, str)]
        tied = majority is None and bool(valid)
        refs[t] = {"k": sum(solved), "single": sum(solved) / 3, "any": float(any(solved)), "majority": sum(valid) / len(valid) if tied else float(strict),
                   "majority_draw": random.Random(t).choice(valid) if tied else strict}
    by_k = defaultdict(list)
    for t in tasks:
        by_k[refs[t]["k"]].append(t)
    out: dict = {"tasks": len(tasks), "seeds": seeds, "k_tasks": {str(k): len(v) for k, v in sorted(by_k.items())},
                 "solver_references": {name: {"all": sum(refs[t][name] for t in tasks) / len(tasks),
                                              "by_k": {str(k): sum(refs[t][name] for t in v) / len(v) for k, v in sorted(by_k.items())},
                                              "mixed_k12": sum(refs[t][name] for k in (1, 2) for t in by_k[k]) / (len(by_k[1]) + len(by_k[2]))}
                                       for name in ("single", "majority", "any")},
                 "arms": {}}
    marks: dict = {}
    for arm in args.arms.split(","):
        path = args.gen_dir / f"gen_{arm}.jsonl"
        if not path.exists():
            continue
        rows = {}
        for line in open(path):
            if line.strip():
                row = json.loads(line)
                rows[(row["task_id"], row["seed"])] = row
        lenient_ok, strict_ok, finishes, parsed, generated, missing = {}, {}, Counter(), Counter(), [], 0
        for t in tasks:
            for s in seeds:
                row = rows.get((t, s))
                if row is None:
                    missing += 1
                    lenient_ok[(t, s)] = strict_ok[(t, s)] = False
                    continue
                finishes[row["finish"]] += 1
                generated.append(row["generated_tokens"])
                read = lenient.readings("deepmath", row["text"])
                parsed[read["lenient"][1]] += 1
                length = row["finish"] == "length"
                lenient_ok[(t, s)] = (not length) and score(row["dataset_task"], read["lenient"][0]) >= 1.0
                strict_ok[(t, s)] = (not length) and score(row["dataset_task"], read["strict"][0]) >= 1.0
        marks[arm] = lenient_ok
        n = len(tasks) * len(seeds)
        out["arms"][arm] = {
            "rows": n - missing, "missing": missing,
            "lenient": sum(lenient_ok.values()) / n, "strict": sum(strict_ok.values()) / n,
            "lenient_per_seed": {str(s): sum(lenient_ok[(t, s)] for t in tasks) / len(tasks) for s in seeds},
            "lenient_by_k": {str(k): sum(lenient_ok[(t, s)] for t in v for s in seeds) / (len(v) * len(seeds)) for k, v in sorted(by_k.items())},
            "lenient_mixed_k12": sum(lenient_ok[(t, s)] for k in (1, 2) for t in by_k[k] for s in seeds) / ((len(by_k[1]) + len(by_k[2])) * len(seeds)),
            "gap_closed": (sum(lenient_ok.values()) / n - out["solver_references"]["single"]["all"])
                          / (out["solver_references"]["any"]["all"] - out["solver_references"]["single"]["all"]),
            "finish": dict(finishes), "length_stops": finishes.get("length", 0), "parsed_lenient": dict(parsed),
            "generated_tokens": {"mean": sum(generated) / max(1, len(generated)), "median": quantile(generated, 0.5), "p90": quantile(generated, 0.9),
                                 "max": max(generated) if generated else None},
            "prompt_tokens_max": max((r["prompt_tokens"] for r in rows.values()), default=None),
            "prompt_plus_generated_max": max((r["prompt_tokens"] + r["generated_tokens"] for r in rows.values()), default=None)}
    # paired comparisons against the reference arm and the majority vote
    ref = marks.get(args.reference)
    rng = random.Random(0)
    for arm, ok in marks.items():
        entry = out["arms"][arm]
        task_mean = {t: sum(ok[(t, s)] for s in seeds) / len(seeds) for t in tasks}
        w = sum(ok[(t, s)] and not refs[t]["majority_draw"] for t in tasks for s in seeds)
        l = sum(refs[t]["majority_draw"] and not ok[(t, s)] for t in tasks for s in seeds)
        entry["vs_majority"] = {"wins": w, "losses": l, "p": round(sign_test(w, l), 4)}
        if ref is None or arm == args.reference:
            continue
        ref_mean = {t: sum(ref[(t, s)] for s in seeds) / len(seeds) for t in tasks}
        diffs = [task_mean[t] - ref_mean[t] for t in tasks]
        boot = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(args.boot))
        mixed = [task_mean[t] - ref_mean[t] for k in (1, 2) for t in by_k[k]]
        boot_mixed = sorted(sum(mixed[rng.randrange(len(mixed))] for _ in mixed) / len(mixed) for _ in range(args.boot))
        w = sum(ok[(t, s)] and not ref[(t, s)] for t in tasks for s in seeds)
        l = sum(ref[(t, s)] and not ok[(t, s)] for t in tasks for s in seeds)
        entry[f"vs_{args.reference}"] = {"diff": sum(diffs) / len(diffs), "ci95": [boot[int(0.025 * args.boot)], boot[int(0.975 * args.boot) - 1]],
                                         "wins": w, "losses": l, "p": round(sign_test(w, l), 4),
                                         "mixed_k12_diff": sum(mixed) / len(mixed),
                                         "mixed_k12_ci95": [boot_mixed[int(0.025 * args.boot)], boot_mixed[int(0.975 * args.boot) - 1]],
                                         "by_k": {str(k): {"wins": sum(ok[(t, s)] and not ref[(t, s)] for t in v for s in seeds),
                                                           "losses": sum(ref[(t, s)] and not ok[(t, s)] for t in v for s in seeds)}
                                                  for k, v in sorted(by_k.items())}}
    timing = args.gen_dir / "timing.jsonl"
    if timing.exists():
        segments = defaultdict(list)
        for line in open(timing):
            if line.strip():
                row = json.loads(line)
                segments[row["arm"]].append({k: row.get(k) for k in ("segment_seconds", "requests", "done_before", "gpu", "vllm", "engine_load_seconds", "ended_utc")})
        out["timing"] = dict(segments)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "timing"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
