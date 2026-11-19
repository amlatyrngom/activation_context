"""
dev_hard report (2026-11-19, CPU, this host): score_math.py (the teacher-test scorer and paired statistics, unchanged) on
problems_devhard.jsonl, for every arm whose gen file exists, overall and per band (d8.5; >= 9 = 9, 9.5, 10), against three references:
results (9B results-only), full (9B full runs, math_eval path) and tg9_full (9B full runs, teacher-test path: the pair of tg27_full).
Writes scores/devhard_<band>_vs_<ref>.json and prints one markdown table per band (accuracy, generated tokens, paired diffs + 95 % CI).

    .venv/bin/python -I devhard_report.py ARM=PATH [ARM=PATH ...]
Optional env: DEVHARD_REFS (comma list of reference arms), DEVHARD_PROBLEMS (another problems file, e.g. a sensitivity subset),
DEVHARD_TAG (suffix of the score file names, so a subset never overwrites the standard files).
"""
import os
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
M = ROOT / "IB/TMP/PRE_REBUTTAL/math_recipe"
BANDS = {"all": "", "d8.5": "8.5", "ge9": "9,9.5,10"}


def main() -> int:
    arms = dict(spec.split("=", 1) for spec in sys.argv[1:])
    arms = {k: v for k, v in arms.items() if Path(v).exists()}
    refs = [r for r in os.environ.get("DEVHARD_REFS", "results,full,tg9_full").split(",") if r in arms]
    for band, diff in BANDS.items():
        scored = {}
        for ref in refs:
            out = M / f"scores/devhard{os.environ.get('DEVHARD_TAG', '')}_{band}_vs_{ref}.json"
            cmd = [sys.executable, "-I", str(HERE / "score_math.py"), "--problems", os.environ.get("DEVHARD_PROBLEMS", str(M / "data/problems_devhard.jsonl")), "--reference", ref, "--out", str(out)]
            cmd += [x for name, path in arms.items() for x in ("--arm", f"{name}={path}")]
            if diff:
                cmd += ["--difficulty", diff]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            scored[ref] = json.loads(out.read_text())
        first = scored[refs[0]]
        sr = first["solver_references"]
        print(f"\n### {band}: {first['tasks']} problems (k {first['k_tasks']}); single {100 * sr['single']['all']:.2f}, majority "
              f"{100 * sr['majority']['all']:.2f}, any-of-3 {100 * sr['any']['all']:.2f}")
        print("| arm | rows | acc | per seed | gen median / mean | " + " | ".join(f"− {r} (CI; W/L, p)" for r in refs) + " |")
        print("|---|---|---|---|---|" + "---|" * len(refs))
        for name in arms:
            v = first["arms"][name]
            g = v["generated_tokens"]
            cells = []
            for r in refs:
                x = scored[r]["arms"][name].get(f"vs_{r}")
                cells.append("—" if x is None else f"{100 * x['diff']:+.2f} ({100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}; {x['wins']}/{x['losses']}, p {x['p']:.2f})")
            seeds = "/".join(f"{100 * s:.1f}" for s in v["lenient_per_seed"].values())
            print(f"| {name} | {v['rows']} | {100 * v['lenient']:.2f} | {seeds} | {g['median']:,} / {g['mean']:,.0f} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
