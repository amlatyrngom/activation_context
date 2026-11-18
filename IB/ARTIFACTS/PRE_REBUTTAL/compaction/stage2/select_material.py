"""
Compaction stage-2 material (README.md here, user-approved 2026-11-17): per 20k-cut prefix, the shortest correct of 3 continuations
(+ any within 1.3x of it) -> material.jsonl lines in the stage-1 format, so prepare_agent_continuation.py --loss sft builds the packets.

Per prefix (one task per seed group g: prefixes_s2gen_c20_s<g>) the 3 samples are
  orig   the pool run's own recorded continuation after the cut: the stage-1 c20 material line of run key <task>#s<g>#c20000
         (export_pool_material.py; the s2gen prefixes are the same cut of the same runs: checked, see below)
  s21    the untuned full-prefix 9B resumed from the prefix, sampling seed 21: cache s2g_s<g>_r_s21_full
  s22    the same, seed 22: cache s2g_s<g>_r_s22_full
A new sample's line is the original's line (head, prefix, tools, template settings, prefix_tokens, prefix_assistant_token_ids: the
prefix is the same) with the continuation fields replaced by the resume run's (export_continuation_material.material_of: its own
head render check against the run's recorded prompt, one message + sampled tokens per assistant step). Checked per new sample:
  - the run's recorded head and prefix messages equal the original line's (else the sample is left out and counted);
  - its template settings equal the original line's.
Checked per prefix: the s2gen prefix file's head_tokens / prefix_steps equal the original line's (the same cut).

Validity (the stage-1 cut rule): finish_reason not trajectory_cap / error / task_unavailable; total tokens (recorded prompt + every
trajectory step) under 50,000; something to predict. Correct = score >= 1.0.
Length = continuation_written + continuation_context (sampled + tool-result tokens after the cut: the H7 rule's length).
Selection: among the valid correct samples, the shortest, plus every other within --near-factor (1.3) x its length. A prefix with
no valid correct sample contributes nothing. Excluded tasks (the 528 of exclude_h7_dev206.json and the 322 of the three frozen
index files) are refused here and again by the packet build.

Each kept line gets run_key "<task>#s<g>#c20000#<sample>" (one item per line in the packet) and
  stage2 = {sample, length, shortest_length, rank, samples_valid, samples_correct, prefix_id}.

    python select_material.py --groups 0 2 --out material/s2ce_dryrun_s02.jsonl          (dry run on the finished caches)
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage1"))
from export_continuation_material import material_of  # noqa: E402

REPO = HERE.parents[4]
T = REPO / "IB/TMP/PRE_REBUTTAL"
STAGE1_MATERIAL = T / "compaction_stage1/material/s1kl_A_c162024_s012_tv_v1_c20.jsonl"
NS = REPO / "IB/TMP/PAPER_COMPACTION/node_sync"
FROZEN = [NS / "prefixes_dev9_75/index.json", NS / "pc-a_live/pc/prefixes_eval9c_200/index.json", NS / "prefixes_eval_100/index.json"]
DEV206 = HERE.parent / "dev_expand/exclude_h7_dev206.json"
BAD_FINISH = ("trajectory_cap", "error", "task_unavailable")
CONTINUATION_FIELDS = ("config_key", "solved", "score", "finish_reason", "answer", "continuation_messages", "target_token_ids")


def length(line: dict) -> int:
    return line["tokens"]["continuation_written"] + line["tokens"]["continuation_context"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--seeds", type=int, nargs="+", default=[21, 22])
    p.add_argument("--rollouts-root", type=Path, default=T / "compaction_stage2/rollouts")
    p.add_argument("--prefixes-root", type=Path, default=T / "s2gen")
    p.add_argument("--stage1-material", type=Path, default=STAGE1_MATERIAL)
    p.add_argument("--near-factor", type=float, default=1.3)
    p.add_argument("--upper", type=int, default=50_000)
    p.add_argument("--allow-partial", action="store_true", help="dry run: a cache may hold fewer rows than its prefix group")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    excluded = set(json.loads(DEV206.read_text())["task_ids"]) | {e["task_id"] for f in FROZEN for e in json.loads(f.read_text())["prefixes"]}

    prefixes = {}                                             # run key of the original -> s2gen index entry
    for g in a.groups:
        for entry in json.loads((a.prefixes_root / f"prefixes_s2gen_c20_s{g}/index.json").read_text())["prefixes"]:
            prefixes[f"{entry['task_id']}#s{g}#c20000"] = entry | {"group": g}
    originals = {}
    with a.stage1_material.open() as handle:
        for raw in handle:
            key = raw[raw.find('"run_key": "') + 12:].split('"', 1)[0]
            if key in prefixes:
                originals[key] = json.loads(raw)
    missing = sorted(set(prefixes) - set(originals))
    if missing:
        raise SystemExit(f"{len(missing)} prefixes without a stage-1 c20 line, e.g. {missing[:3]}")
    counts, why = Counter(), Counter()
    for key, entry in prefixes.items():
        line = originals[key]
        if entry["task_id"] in excluded:
            raise SystemExit(f"{key}: an excluded task among the prefixes")
        if entry["head_tokens"] != line["tokens"]["head"] or entry["prefix_steps"] != line["teacher"]["prefix_steps"]:
            raise SystemExit(f"{key}: the s2gen prefix is not the stage-1 cut (head {entry['head_tokens']} vs {line['tokens']['head']}, "
                             f"steps {entry['prefix_steps']} vs {line['teacher']['prefix_steps']})")

    samples = {key: [] for key in prefixes}                   # run key -> [(sample name, line, valid, why)]
    for key, line in originals.items():
        total = line["teacher"]["total_tokens"]
        ok = line["finish_reason"] not in BAD_FINISH and total < a.upper
        samples[key].append(("orig", dict(line), ok, "" if ok else f"finish_reason {line['finish_reason']} / total {total}"))
    options = SimpleNamespace(solved_only=False, min_target_tokens=1, reader_tokenizer="Qwen/Qwen3.5-9B", rollouts="")
    caches = {}
    for g in a.groups:
        for seed in a.seeds:
            cache = f"s2g_s{g}_r_s{seed}_full"
            path = a.rollouts_root / cache / "rollouts.jsonl"
            options.rollouts = f"agent_eval_dev75/ROLLOUTS/{cache}/rollouts.jsonl"
            latest = {}
            with path.open() as handle:
                for raw in handle:
                    if raw.strip():
                        row = json.loads(raw)
                        latest[f"{row['agent_config']['dataset_task']['task_id']}#s{g}#c20000"] = row
            expected = sum(1 for e in prefixes.values() if e["group"] == g)
            caches[cache] = {"rows": len(latest), "expected": expected}
            if len(latest) != expected and not a.allow_partial:
                raise SystemExit(f"{cache}: {len(latest)} rows of {expected}")
            for key, row in latest.items():
                if key not in prefixes:
                    raise SystemExit(f"{cache}: {key} is not one of the group's prefixes")
                if row.get("seed") != seed:
                    raise SystemExit(f"{cache}: {key} sampled with seed {row.get('seed')}")
                name = f"s{seed}"
                total = len(row["prompt_token_ids"]) + sum(len(step.get("token_ids") or []) for step in row["trajectory"])
                record, reason = material_of(row, tokenizer, None, None, options)
                original = originals[key]
                if record is None:
                    samples[key].append((name, None, False, reason)); continue
                if record["head_messages"] != original["head_messages"] or record["prefix_messages"] != original["prefix_messages"]:
                    samples[key].append((name, None, False, "head/prefix messages differ from the original's")); continue
                if record["chat_template_kwargs"] != original["chat_template_kwargs"]:
                    samples[key].append((name, None, False, "template settings differ from the original's")); continue
                line = dict(original)
                for field in CONTINUATION_FIELDS:
                    line[field] = record[field]
                line["tokens"] = dict(original["tokens"], **{k: record["tokens"][k] for k in ("continuation_written", "continuation_context", "continuation_turns")})
                line["chat_template_kwargs_recorded"] = record["chat_template_kwargs_recorded"]
                line["source"] = record["source"] | {"kind": "s2g resume (full prefix, untuned 9B)", "recorded_prompt_tokens": len(row["prompt_token_ids"]),
                                                     "total_tokens": total, "original_source": original["source"]}
                ok = row.get("finish_reason") not in BAD_FINISH and total < a.upper
                samples[key].append((name, line, ok, "" if ok else (f"finish_reason {row.get('finish_reason')}" if row.get("finish_reason") in BAD_FINISH
                                                                     else f"total >= {a.upper}" + (" (correct)" if line["solved"] else ""))))

    kept, per_prefix_kept, shortest_sample, medians_target = [], Counter(), Counter(), []
    for key in sorted(samples):
        entries = samples[key]
        for name, line, ok, reason in entries:
            counts[f"samples {name}"] += 1
            if not ok:
                why[f"{name}: {reason}"] += 1
        valid = [(name, line) for name, line, ok, _ in entries if ok]
        correct = sorted([(name, line) for name, line in valid if line["solved"]], key=lambda nl: (length(nl[1]), nl[0]))
        counts[f"prefixes with {len(correct)} correct"] += 1
        if not correct:
            if any(line is not None and line["solved"] for _, line, _, _ in entries):
                counts["prefixes whose every correct sample is invalid (the 50k total cap / ending)"] += 1
            continue
        best = length(correct[0][1])
        chosen = [correct[0]] + [nl for nl in correct[1:] if length(nl[1]) <= a.near_factor * best]
        shortest_sample[correct[0][0]] += 1
        per_prefix_kept[len(chosen)] += 1
        for rank, (name, line) in enumerate(chosen):
            out = dict(line)
            out["run_key"] = f"{key}#{name}"
            original = originals[key]
            out["stage2"] = {"sample": name, "length": length(line), "shortest_length": best, "rank": rank, "prefix_id": key,
                             "original_length": length(original), "original_written": original["tokens"]["continuation_written"],
                             "original_solved": original["solved"],
                             "samples_valid": len(valid), "samples_correct": len(correct), "near_factor": a.near_factor}
            kept.append(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as handle:
        for line in kept:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    targets = [line["tokens"]["continuation_written"] for line in kept]
    summary = {
        "groups": a.groups, "caches": caches, "prefixes": len(prefixes), "prefixes_kept": sum(per_prefix_kept.values()),
        "lines": len(kept), "lines_per_prefix": dict(sorted(per_prefix_kept.items())),
        "lines_by_sample": dict(Counter(line["stage2"]["sample"] for line in kept)), "shortest_by_sample": dict(shortest_sample),
        "share_lines_original": round(sum(line["stage2"]["sample"] == "orig" for line in kept) / max(1, len(kept)), 4),
        "target_tokens": {"total": sum(targets), "median": statistics.median(targets) if targets else None,
                          "mean": round(statistics.mean(targets), 1) if targets else None},
        "length_median": statistics.median([length(line) for line in kept]) if kept else None,
        "shortening": {   # each kept line against the original recorded continuation of its prefix (solved or not)
            "share_length_under_half_original": round(sum(line["stage2"]["length"] < 0.5 * line["stage2"]["original_length"] for line in kept) / max(1, len(kept)), 4),
            "share_written_under_half_original": round(sum(line["tokens"]["continuation_written"] < 0.5 * line["stage2"]["original_written"] for line in kept) / max(1, len(kept)), 4),
            "median_length_over_original": statistics.median([line["stage2"]["length"] / max(1, line["stage2"]["original_length"]) for line in kept]) if kept else None},
        "counts": dict(sorted(counts.items())), "left_out_samples": dict(sorted(why.items())),
        "near_factor": a.near_factor, "upper": a.upper, "stage1_material": str(a.stage1_material),
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
