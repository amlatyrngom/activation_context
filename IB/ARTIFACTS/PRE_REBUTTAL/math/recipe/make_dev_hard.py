"""
Math recipe dev_hard (2026-11-18): a second development set of DeepMath problems at difficulty >= 8.5, weighted toward >= 9, because
dev (tqa_dev_solve_v1) holds only 16 problems at 8.5 and 4 at 9 while the paper's advantage sits on the hard ones (dmhard: 226 / 74).
No unused hard problem has three 9B solver runs (all 1,063 such problems are in train), so this draws fresh problems from the same
pool and loader as every other set (DeepMathDataset, min difficulty 6, seed 0) and the solver runs are generated afterwards with the
same setup (kit/run_tqa_solve_v2.py, untuned 9B, seeds 0..2, cap 40k).

Excluded, by task id AND by normalised question text (the draw rule of make_tqa_m1_train_v2.py):
  * the math recipe's train (1,500), dev (200) and the test sets dmext (300) + dmhard (300): read from data/splits.json, and their
    question texts from the pool; the teacher test's 200 problems (all in train; asserted separately);
  * every DeepMath sample.json of the trajectory-QA work (--used, repeatable) and tqa_at3_v1 (re-drawn as run_at3.py drew it);
  * every hard DeepMath id found anywhere in IB/TMP (--used-ids: one id per line).
Draw: --count-9 problems at difficulty >= 9 and --count-85 at 8.5, each a seeded shuffle of its band, skipping repeated texts.
Writes PAPER_TRAJ_QA/<id>/sample.json (make_tqa_samples.py's shape; the solver reads it) and refuses to redraw.

    .venv/bin/python -I make_dev_hard.py --splits IB/TMP/PRE_REBUTTAL/math_recipe/data/splits.json --used S1 --used S2 ... --used-ids F --out DIR
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def difficulty(task_id: str) -> float:
    return float(task_id.split("-")[0][1:])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--teacher-prompts", type=Path, required=True, help="the teacher test's prompts.jsonl (its 200 problems: asserted disjoint too)")
    p.add_argument("--used", action="append", default=[], help="a sample.json (repeatable)")
    p.add_argument("--used-ids", action="append", default=[], help="a file of used task ids (repeatable)")
    p.add_argument("--id", default="tqa_math_dev_hard_v1")
    p.add_argument("--count-9", type=int, default=100)
    p.add_argument("--count-85", type=int, default=80)
    p.add_argument("--at3-seed", type=int, default=20261014)
    p.add_argument("--seed", type=int, default=20261118)
    p.add_argument("--out", type=Path, required=True, help="folder for <id>/sample.json")
    a = p.parse_args()
    from activation.dataset.loaders import DeepMathDataset
    deepmath = DeepMathDataset.load(None, 10 ** 6, seed=0, min_difficulty=6.0)
    tasks = deepmath.scorable_tasks
    text = lambda t: norm(tasks[t].task_datum.get("question") or tasks[t].agent_prompt)
    splits = json.loads(a.splits.read_text())
    protected = {"train": set(splits["train"]), "dev": set(splits["dev"]), "dmext": set(splits["test"]["dmext"]), "dmhard": set(splits["test"]["dmhard"])}
    protected["teacher_test"] = {json.loads(line)["task_id"] for line in open(a.teacher_prompts) if line.strip()}
    assert len(protected["teacher_test"]) == 200, len(protected["teacher_test"])
    used = {f"recipe_{k}": v for k, v in protected.items()}
    used["at3"] = set(random.Random(a.at3_seed).sample(sorted(tasks), 100))
    for path in a.used:
        for entry in json.load(open(path))["sample"].values():
            if entry["dataset_id"] == deepmath.dataset_id:
                used[path] = set(entry["task_ids"])
    for path in a.used_ids:
        used[path] = {line.strip() for line in open(path) if line.strip()}
    for name, ids in protected.items():
        missing = ids - set(tasks)
        assert not missing, f"{name}: {len(missing)} ids not in the pool"
    used_ids = set().union(*used.values())
    used_text = {text(t) for t in used_ids if t in tasks}
    pool = {band: [t for t in sorted(tasks) if t not in used_ids and text(t) not in used_text and ok(difficulty(t))]
            for band, ok in (("ge9", lambda d: d >= 9), ("d8.5", lambda d: d == 8.5))}
    print("pool", len(tasks), "used ids", len(used_ids), "used texts", len(used_text), "untouched", {b: len(v) for b, v in pool.items()})
    chosen, seen = [], set()
    for band, count in (("ge9", a.count_9), ("d8.5", a.count_85)):
        taken = 0
        for task_id in random.Random(f"{a.seed}-{band}").sample(pool[band], len(pool[band])):
            if text(task_id) in seen:
                continue
            seen.add(text(task_id)); chosen.append(task_id); taken += 1
            if taken == count:
                break
        assert taken == count, (band, taken, count)
    # the asserts the coordinator asked for: 0 shared ids and 0 shared question texts with train / dev / dmext / dmhard / the teacher test
    overlap = {}
    for name, ids in protected.items():
        shared_ids = set(chosen) & ids
        shared_text = {text(t) for t in chosen} & {text(t) for t in ids}
        overlap[name] = {"task_ids": len(shared_ids), "question_texts": len(shared_text)}
        assert not shared_ids and not shared_text, (name, overlap[name])
    assert len(set(chosen)) == len(chosen) == len(seen)
    sample = {"deepmath": {"dataset_id": deepmath.dataset_id, "pool": len(tasks), "task_ids": chosen, "loader_kwargs": {"min_difficulty": 6},
                           "max_examples": 10 ** 6}}
    sha = hashlib.sha256("\n".join(sorted(chosen)).encode()).hexdigest()
    folder = a.out / a.id
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "sample.json"
    assert not out.exists(), f"{out} exists; a frozen sample is never redrawn"
    doc = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "role": "development (math recipe dev_hard: DeepMath >= 8.5, weighted toward >= 9); never training, never test",
           "sha256": {"deepmath": sha}, "difficulty": dict(sorted(Counter(t.split("-")[0] for t in chosen).items())), "overlap": overlap,
           "excluded": {name: len(ids) for name, ids in used.items()}, "sample": sample}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "sample"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
