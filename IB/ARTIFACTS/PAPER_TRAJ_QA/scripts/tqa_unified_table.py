"""
Unified DeepMath difficulty table (Oct 18; user: one table, 6-8 in one column, 8.5 and >=9 in their own columns). Two layouts:
  merged:   6-8 = tqa_dmext_solve_v1 tasks < 8.5; 8.5 and >=9 pool both sets (dmext's 28 tasks at >=8.5 join tqa_dmhard's).
  separate: 6-8 from tqa_dmext_solve_v1, 8.5 and >=9 from tqa_dmhard only (dmext's 28 hard tasks left out).
Share = per-task mean over that task's seeds, averaged over tasks (each task weighs the same whatever its seed count; dmext 3 seeds,
dmhard 5). Paired tests are task x seed pooled over seeds, two-sided sign test, as tqa_dm_arms.py (seed i against the reference's
seed i; against majority, each seed against the fixed vote). Scoring as tqa_dm_arms.py.
Majority vote (user, Oct 18: the paper's "Majority Vote"): the answer two runs share; with no such answer, a uniformly random one of
the runs' answers. Its share is the expected score of that draw; its paired tests use one fixed draw (seeded by task id). The strict
vote (no majority = unsolved; the earlier tables) is kept as "majority strict". --json writes the separate layout for the figures.

    PYTHONPATH=../../PAPER_COMPACTION/kit python tqa_unified_table.py --sync-root <SYNC mirror> --out t.txt
"""
import argparse
import json
import random
from pathlib import Path

import score_synth_lenient as lenient
from run_synth import load_solvers, solver_result
from score_synth1call import score, sign_test
from tqa_r2_gate import load

SETS = {"dmext": ("tqa_dmext_solve_v1", "r2dmx", 3), "dmhard": ("tqa_dmhard", "r2dmh", 5)}
ARMS = [("results", "gen", "PAPER_TRAJ_QA/tqa_{p}_res_s{s}/gen_0.jsonl"), ("M1", "ac", "ROLLOUTS/tqa_{p}_ac_m1_s{s}/rollouts.jsonl"),
        ("M1 no rows", "ac", "ROLLOUTS/tqa_{p}_norows_m1_s{s}/rollouts.jsonl")]


def set_marks(root: Path, at3_id: str, tag: str, seeds: int) -> dict:
    sample = json.loads((root / "PAPER_TRAJ_QA" / at3_id / "sample.json").read_text())["sample"]["deepmath"]
    rows = load_solvers(root, at3_id)
    tasks = [t for t in sample["task_ids"] if all((t, s) in rows for s in range(3))]
    out = {}
    for t in tasks:
        task = rows[(t, 0)]["agent_config"]["dataset_task"]
        answers = [solver_result(rows[(t, s)], 40_000) for s in range(3)]
        solved = [score(task, a) >= 1.0 for a in answers]
        majority = lenient.majority_answer("deepmath", answers)
        strict = majority is not None and score(task, majority) >= 1.0
        valid = [ok for answer, ok in zip(answers, solved) if isinstance(answer, str)]
        tied = majority is None and bool(valid)
        out[t] = {"diff": float(t.split("-")[0][1:]), "single": solved, "any@3": [any(solved)], "majority strict": [strict],
                  "majority": [sum(valid) / len(valid) if tied else strict],
                  "majority draw": [random.Random(t).choice(valid) if tied else strict]}
    for name, kind, pattern in ARMS:
        for s in range(seeds):
            got = load(f"{kind}:{root / pattern.format(p=tag, s=s)}")
            for t in tasks:
                row = got.get(t)
                ok = False
                if row is not None and not row["bad"] and not row["length"]:
                    answer, _ = lenient.readings("deepmath", row["text"])["lenient"]
                    ok = score(row["task"], answer) >= 1.0
                out[t].setdefault(name, []).append(ok)
    print(f"{at3_id}: {len(tasks)} tasks", flush=True)
    return out


def paired(sel: list[dict], name: str, ref: str) -> str:
    w = l = 0
    for m in sel:
        refs = m["majority draw"] * len(m[name]) if ref == "majority" else m[ref]
        w += sum(a and not b for a, b in zip(m[name], refs))
        l += sum(b and not a for a, b in zip(m[name], refs))
    return f"+{w}/-{l} (p {sign_test(w, l):.3f})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    root = Path(args.sync_root)
    marks = {key: set_marks(root, *spec) for key, spec in SETS.items()}
    overlap = set(marks["dmext"]) & set(marks["dmhard"])
    ext, hard = marks["dmext"].values(), marks["dmhard"].values()
    easy = [m for m in ext if m["diff"] < 8.5]
    layouts = {
        "merged": [("6-8", easy), ("8.5", [m for m in [*ext, *hard] if 8.5 <= m["diff"] < 9]),
                   (">=9", [m for m in [*ext, *hard] if m["diff"] >= 9]), ("all", [*ext, *hard])],
        "separate": [("6-8", easy), ("8.5", [m for m in hard if m["diff"] < 9]), (">=9", [m for m in hard if m["diff"] >= 9]),
                     ("all", [*easy, *hard])],
    }
    rows = ["single", "majority strict", "majority", "results", "M1 no rows", "M1", "any@3"]
    lines = [f"task overlap between sets: {len(overlap)}"]
    for layout, cols in layouts.items():
        lines += ["", f"== {layout}: % solved (each task = mean over its seeds)",
                  f"{'':<12}" + "".join(f"{c} (n {len(sel)})".rjust(16) for c, sel in cols)]
        for r in rows:
            lines.append(f"{r:<12}" + "".join(f"{100 * sum(sum(m[r]) / len(m[r]) for m in sel) / len(sel):16.1f}" for _, sel in cols))
        lines.append("paired, task x seed pooled:")
        for name, ref in [("M1", "results"), ("M1", "majority"), ("results", "majority"), ("M1", "M1 no rows"), ("M1 no rows", "results")]:
            lines.append(f"  {name} vs {ref}".ljust(26) + "".join(paired(sel, name, ref).rjust(24) for _, sel in cols))
    if args.json:
        cols = layouts["separate"][:3]
        Path(args.json).write_text(json.dumps({
            "layout": "separate: 6-8 tqa_dmext_solve_v1 (3 seeds), 8.5 and >=9 tqa_dmhard (5 seeds); majority = random tie-break",
            "columns": [c for c, _ in cols], "n": [len(sel) for _, sel in cols],
            "solved": {r: [round(100 * sum(sum(m[r]) / len(m[r]) for m in sel) / len(sel), 2) for _, sel in cols] for r in rows},
            "paired": {f"{a} vs {b}": [paired(sel, a, b) for _, sel in cols]
                       for a, b in [("M1", "results"), ("M1", "majority"), ("M1", "M1 no rows"), ("M1 no rows", "results")]}}, indent=1))
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
