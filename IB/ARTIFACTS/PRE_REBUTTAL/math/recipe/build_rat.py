"""
Math recipe, rationalization targets (STaR-style, 2026-11-19, coordinator's step (c)), on CPU (this host).

Problems: training problems at difficulty >= 8.5 whose 8 teacher samples (seeds 0-2 teacher_train + 3-7 teacher_train_s37) hold at most
one correct one (the teacher-test scorer, build_packets.correct). For each, the 9B teacher (synth_gen.py, same engine / sampler / caps)
reads the usual full-runs prompt with one extra block before the final rule, RAT_BLOCK, that states the gold final answer and asks for a
self-contained derivation that never refers to it. The training item keeps the ordinary views (student = the AC view, teacher = the
full-runs prompt WITHOUT the answer); only the target comes from the rationalization prompt.

  prompts  -> DATA/prompts_rat_v1.jsonl (condition "rat", synth_gen.py's prompt format), DATA/tasks_rat_v1.json, rat_select.json
  filter   GEN_DIR -> KEPT_DIR/gen_rat_kept.jsonl + filter.json: a sample is kept when it stopped, reaches the gold answer (lenient
           parse, math-verify) and passes the leak filter (LEAK, case-insensitive, on the whole reply, thinking included); per
           problem the shortest kept sample and every other kept one within 1.3x of it (the stage-2 shortest rule). filter.json counts
           every drop by reason and pattern, and the same patterns' hit rate on the ordinary 8 teacher samples of the same problems
           (the filter's false-positive baseline).

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> .venv/bin/python -I build_rat.py prompts
    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> .venv/bin/python -I build_rat.py filter --gen GEN_DIR --out KEPT_DIR
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_packets as B                                     # noqa: E402  (paths, loaders, the scorer)

GEN_IN = B.M / "gen_in"
MIN_DIFFICULTY, MAX_CORRECT = 8.5, 1
RAT_BLOCK = ("The correct final answer to this problem is known: {answer}\n"
             "Write your reply as a complete, self-contained solution that derives this answer from the problem statement, using whatever "
             "is correct in the agents' runs, as if you had worked it out yourself. Do not mention or refer to the known answer, to being "
             "given it, or to checking against it, anywhere in your reasoning or your reply: every step must follow from the mathematics.")
LEAK = [r"\b(given|known|reference|supplied|stated|target|provided|expected) (final |correct )?answer\b(?!s)",
        r"\b(correct |final )?answer (is|was) (already )?(known|given|provided|stated|supplied)\b", r"\bthe answer is known\b",
        r"\btold (that )?the (final |correct )?answer", r"\b(we are|we're|i am|i'm|i was|we were|i've been|we've been) (told|given) (that )?(the )?(final |correct )?answer",
        r"\b(prompt|instructions?|user|problem statement|task) (says|said|states|stated|tells|told|gives|gave|provides|provided|reveals)[^.\n]{0,40}\banswer (is|=|:|of)",
        r"\banswer (we|i) (need|must|have|should|want) (to )?(reach|get|derive|arrive|show|match|obtain|justify)",
        r"\b(need|must|have|supposed|want) to (reach|derive|arrive at|match|justify) (the|this|that) (known |given |correct )?answer",
        r"\bwork(ing)? backwards? (from|to) (the|this|that) (known |given |final |correct )?(answer|result)", r"\bmatch(es|ing)? the (known|given|reference|stated|provided) ",
        r"\bground[- ]truth\b", r"\bhint[^.\n]{0,30}\banswer", r"\bas if (i|we) had\b", r"\bself-contained\b", r"\bpretend",
        r"\b(without|never|not) (mentioning|referring|referencing)\b", r"\bknown: ", r"\bchecking against (it|the (known|given))"]
LEAK_RE = [re.compile(p, re.IGNORECASE) for p in LEAK]


BINARY = {"yes", "no", "true", "false"}


def is_binary(task: dict) -> bool:
    return task["gold_answer"].strip().lower() in BINARY


def leak_hits(text: str) -> list[str]:
    return [p.pattern for p in LEAK_RE if p.search(text)]


def by_task(folders) -> dict:
    out = {}
    for folder in folders:
        for s in B.load_gen(folder):
            out.setdefault(s["task_id"], []).append(s)
    return out


def select() -> tuple[dict, list]:
    train = B.load_problems("train")
    ordinary = {t: ss for t, ss in by_task([GEN_IN / "teacher_train", GEN_IN / "teacher_train_s37"]).items() if t in train}
    counts = {}
    for t, ss in ordinary.items():
        if float(train[t]["difficulty"]) >= MIN_DIFFICULTY:
            assert len(ss) == 8 and sorted(s["seed"] for s in ss) == list(range(8)), (t, len(ss))
            counts[t] = sum(B.correct(s) for s in ss)
    chosen = sorted(t for t, n in counts.items() if n <= MAX_CORRECT)
    return {"train": train, "ordinary": ordinary, "counts": counts}, chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("prompts", "filter"))
    p.add_argument("--gen", type=Path, help="filter: the synth_gen output folder of the rat prompts")
    p.add_argument("--out", type=Path, help="filter: the kept-sample folder (new)")
    p.add_argument("--within", type=float, default=1.3)
    p.add_argument("--exclude-binary", action="store_true", help="filter: leave Yes/No/True/False golds out of gen_rat_kept.jsonl (counted both ways)")
    a = p.parse_args()
    splits = json.loads((B.DATA / "splits.json").read_text())
    excluded = set(splits["dev"]) | set(splits["test"]["dmext"]) | set(splits["test"]["dmhard"])
    ctx, chosen = select()
    train = ctx["train"]
    assert not set(chosen) & excluded and set(chosen) <= set(splits["train"])
    if a.phase == "prompts":
        from transformers import AutoTokenizer
        from run_synth1call_v2 import FINAL_RULE
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        full = {r["task_id"]: r for r in map(json.loads, open(B.DATA / "prompts_train.jsonl")) if r["condition"] == "full"}
        rows = []
        for t in chosen:
            base = full[t]["prompt"]
            assert base.endswith("\n\n" + FINAL_RULE["deepmath"]), t
            gold = train[t]["dataset_task"]["gold_answer"]
            prompt = base[: -len(FINAL_RULE["deepmath"])] + RAT_BLOCK.format(answer=gold) + "\n\n" + FINAL_RULE["deepmath"]
            text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            rows.append({"id": f"rat/{t}", "condition": "rat", "task_id": t, "k": full[t]["k"], "solver_right": full[t]["solver_right"],
                         "dataset_task": full[t]["dataset_task"], "solver_results": full[t]["solver_results"], "prompt": prompt,
                         "chat_text_sha256": hashlib.sha256(text.encode()).hexdigest(), "prompt_token_ids": tok.encode(text, add_special_tokens=False)})
        with open(B.DATA / "prompts_rat_v1.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (B.DATA / "tasks_rat_v1.json").write_text(json.dumps(chosen))
        lengths = sorted(len(r["prompt_token_ids"]) for r in rows)
        report = {"rule": f"train, difficulty >= {MIN_DIFFICULTY}, <= {MAX_CORRECT} correct of the 8 ordinary teacher samples", "problems": len(chosen),
                  "by_difficulty": dict(Counter(train[t]["difficulty"] for t in chosen)), "by_correct_of_8": dict(Counter(ctx["counts"][t] for t in chosen)),
                  "by_solver_k": dict(Counter(train[t]["k"] for t in chosen)), "rat_block": RAT_BLOCK,
                  "prompt_tokens": {"median": lengths[len(lengths) // 2], "max": lengths[-1]},
                  "prompts_sha256": hashlib.sha256((B.DATA / "prompts_rat_v1.jsonl").read_bytes()).hexdigest()}
        (B.DATA / "rat_select.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return 0
    assert a.gen and a.out and not a.out.exists()
    rat = [s for s in B.load_gen(a.gen) if s["task_id"] in set(chosen)]
    assert all(s["condition"] == "rat" for s in rat)
    drops, pattern_hits, kept_by_task = Counter(), Counter(), {}
    for s in rat:
        if s["finish"] != "stop":
            drops["no_stop"] += 1
            continue
        if not B.correct(s):
            drops["wrong_answer"] += 1
            continue
        hits = leak_hits(s["text"])
        if hits:
            drops["leak"] += 1
            pattern_hits.update(hits)
            continue
        kept_by_task.setdefault(s["task_id"], []).append(s)
    selected = []
    for t, ss in sorted(kept_by_task.items()):
        ss.sort(key=lambda s: (s["generated_tokens"], s["seed"]))
        selected += [s for s in ss if s["generated_tokens"] <= a.within * ss[0]["generated_tokens"]]
    kept = [s for s in selected if not (a.exclude_binary and is_binary(train[s["task_id"]]["dataset_task"]))]

    def split(rows, key=lambda s: s["task_id"]):
        out = {"binary": Counter(), "non_binary": Counter()}
        for s in rows:
            out["binary" if is_binary(train[s["task_id"]]["dataset_task"]) else "non_binary"][key(s)] += 1
        return {k: {"samples": sum(v.values()), "problems": len(v)} for k, v in out.items()}
    by_answer_type = {"problems_selected": {"binary": sum(is_binary(train[t]["dataset_task"]) for t in chosen),
                                            "non_binary": sum(not is_binary(train[t]["dataset_task"]) for t in chosen)},
                      "sampled": split(rat), "passing": split([s for v in kept_by_task.values() for s in v]), "shortest_rule_selected": split(selected),
                      "drops": {}}
    for s in rat:
        reason = "no_stop" if s["finish"] != "stop" else ("wrong_answer" if not B.correct(s) else ("leak" if leak_hits(s["text"]) else None))
        if reason:
            kind = "binary" if is_binary(train[s["task_id"]]["dataset_task"]) else "non_binary"
            by_answer_type["drops"].setdefault(kind, Counter())[reason] += 1
    base_hits, base_n = Counter(), 0                                 # the filter on the ordinary samples of the same problems
    for t in chosen:
        for s in ctx["ordinary"][t]:
            base_n += 1
            base_hits.update(leak_hits(s["text"]))
    a.out.mkdir(parents=True)
    with open(a.out / "gen_rat_kept.jsonl", "w") as f:
        for s in kept:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    report = {"samples": len(rat), "problems_sampled": len({s["task_id"] for s in rat}), "drops": dict(drops), "leak_pattern_hits": dict(pattern_hits.most_common()),
              "passing_samples": sum(len(v) for v in kept_by_task.values()), "problems_with_a_pass": len(kept_by_task), "kept_items": len(kept), "exclude_binary": a.exclude_binary, "by_answer_type": by_answer_type,
              "kept_by_difficulty": dict(Counter(train[s["task_id"]]["difficulty"] for s in kept)),
              "kept_generated_tokens": B.quantiles([s["generated_tokens"] for s in kept]) if kept else None,
              "ordinary_samples_leak_baseline": {"samples": base_n, "flagged": sum(1 for t in chosen for s in ctx["ordinary"][t] if leak_hits(s["text"])),
                                                 "pattern_hits": dict(base_hits.most_common())},
              "leak_patterns": LEAK, "within": a.within}
    (a.out / "filter.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
