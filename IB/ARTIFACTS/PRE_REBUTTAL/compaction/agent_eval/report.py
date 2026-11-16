"""
Dev-75 compaction agent evaluation: per-arm solved share and continuation tokens, and paired comparisons (2026-11-16).

    python -I report.py --arm NAME:SEED:ROLLOUTS_JSONL [--arm ...] --compare AC_ARM:BASE_ARM [--compare ...] [--budget 50000] [--out OUT.json]

Per run (as run_resume.summarize): solved = score >= 1; continuation tokens = the run's sampled + tool tokens after the prefix,
min(tokens, budget) when it submitted an answer, else the whole budget. A pair is (prefix, seed): the AC arm and the baseline ran
the same prefix with the same seed (sampling still differs, the prompts differ). Differences are AC minus baseline over the pairs
both arms have; 95 % intervals from a bootstrap over prefixes (all seeds of a prefix resampled together; 10,000 draws); McNemar's
exact test on the discordant (prefix, seed) pairs of the solved outcome (it treats the seeds of one prefix as independent pairs).
Runs that ended in `error` are counted and shown; they stay in the tables as unsolved.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict


def task_of(row: dict) -> str:
    task = (row.get("agent_config") or {}).get("dataset_task")
    if isinstance(task, dict) and task.get("task_id"):
        return task["task_id"]
    return row["config_key"].split("/")[1]


def load(path: str, budget: int) -> dict:
    runs = {}
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = sum(len(step.get("token_ids") or []) for step in row.get("trajectory") or [])
            submitted = row.get("finish_reason") == "submitted"
            runs[task_of(row)] = {"solved": bool(row.get("score") is not None and row["score"] >= 1.0), "submitted": submitted,
                                  "tokens": min(tokens, budget) if submitted else budget, "finish": row.get("finish_reason"),
                                  "seconds": row.get("duration") or 0.0, "ac_rows": row.get("num_ac_rows") or 0}
    return runs


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="NAME:SEED:PATH to a rollouts.jsonl")
    p.add_argument("--compare", action="append", default=[], help="A:B (A minus B)")
    p.add_argument("--budget", type=int, default=50_000)
    p.add_argument("--draws", type=int, default=10_000)
    p.add_argument("--prefixes", default=None, help="index.json: restrict to (and require) these task ids")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    arms: dict[str, dict[tuple[str, int], dict]] = defaultdict(dict)
    order: list[str] = []
    for spec in a.arm:
        name, seed, path = spec.split(":", 2)
        if name not in order:
            order.append(name)
        for task, run in load(path, a.budget).items():
            arms[name][(task, int(seed))] = run
    allowed = None
    if a.prefixes:
        allowed = {entry["task_id"] for entry in json.load(open(a.prefixes))["prefixes"]}
        for name in order:
            arms[name] = {key: run for key, run in arms[name].items() if key[0] in allowed}
    report = {"budget": a.budget, "arms": {}, "comparisons": {}}
    print(f"| arm | seeds | runs | solved % | answered % | mean continuation tokens | errors | finish reasons |")
    print("|---|---|---|---|---|---|---|---|")
    for name in order:
        runs = list(arms[name].values())
        seeds = sorted({seed for _, seed in arms[name]})
        finishes = Counter(run["finish"] for run in runs)
        row = {"seeds": seeds, "runs": len(runs), "solved_pct": 100 * sum(r["solved"] for r in runs) / len(runs),
               "answered_pct": 100 * sum(r["submitted"] for r in runs) / len(runs), "mean_tokens": sum(r["tokens"] for r in runs) / len(runs),
               "errors": finishes.get("error", 0), "finish": dict(finishes),
               "per_seed_solved_pct": {seed: 100 * sum(r["solved"] for (t, s), r in arms[name].items() if s == seed) /
                                       max(1, sum(1 for (t, s) in arms[name] if s == seed)) for seed in seeds}}
        report["arms"][name] = row
        print(f"| {name} | {','.join(map(str, seeds))} | {row['runs']} | {row['solved_pct']:.2f} | {row['answered_pct']:.1f} | {row['mean_tokens']:.0f} | "
              f"{row['errors']} | {json.dumps(row['finish'])} |")
    if a.compare:
        print()
        print("| A − B | pairs | prefixes | solved A % | solved B % | Δ solved (pts) [95 % CI] | A✓B✗ / A✗B✓ | McNemar p | Δ tokens [95 % CI] |")
        print("|---|---|---|---|---|---|---|---|---|")
    rng = random.Random(0)
    for spec in a.compare:
        first, second = spec.split(":")
        keys = sorted(set(arms[first]) & set(arms[second]))
        by_prefix = defaultdict(list)
        for key in keys:
            by_prefix[key[0]].append(key)
        prefixes = sorted(by_prefix)

        def deltas(chosen):
            pairs = [key for prefix in chosen for key in by_prefix[prefix]]
            solved = sum(arms[first][k]["solved"] - arms[second][k]["solved"] for k in pairs) / len(pairs)
            tokens = sum(arms[first][k]["tokens"] - arms[second][k]["tokens"] for k in pairs) / len(pairs)
            return 100 * solved, tokens

        point = deltas(prefixes)
        draws = [deltas([rng.choice(prefixes) for _ in prefixes]) for _ in range(a.draws)]
        ci = lambda values: (sorted(values)[int(0.025 * len(values))], sorted(values)[int(0.975 * len(values)) - 1])
        solved_ci, tokens_ci = ci([d[0] for d in draws]), ci([d[1] for d in draws])
        b = sum(arms[first][k]["solved"] and not arms[second][k]["solved"] for k in keys)
        c = sum(arms[second][k]["solved"] and not arms[first][k]["solved"] for k in keys)
        row = {"pairs": len(keys), "prefixes": len(prefixes), "solved_a_pct": 100 * sum(arms[first][k]["solved"] for k in keys) / len(keys),
               "solved_b_pct": 100 * sum(arms[second][k]["solved"] for k in keys) / len(keys), "delta_solved_pts": point[0],
               "delta_solved_ci": solved_ci, "a_only": b, "b_only": c, "mcnemar_p": mcnemar_exact(b, c), "delta_tokens": point[1],
               "delta_tokens_ci": tokens_ci}
        report["comparisons"][spec] = row
        print(f"| {first} − {second} | {len(keys)} | {len(prefixes)} | {row['solved_a_pct']:.2f} | {row['solved_b_pct']:.2f} | "
              f"{point[0]:+.2f} [{solved_ci[0]:+.2f}, {solved_ci[1]:+.2f}] | {b} / {c} | {row['mcnemar_p']:.3f} | "
              f"{point[1]:+.0f} [{tokens_ci[0]:+.0f}, {tokens_ci[1]:+.0f}] |")
    if a.out:
        with open(a.out, "w") as handle:
            json.dump(report, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
