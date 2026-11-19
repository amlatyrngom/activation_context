"""
Math synthesis clean recipe (2026-11-17): trainer packets and the dev panel, on CPU (this host), from build_data.py's records and
synth_gen.py's teacher syntheses. Every item goes through the trainer's own example builder (run_channel_mixture.checked_example:
72,000-token limit on both reader views, 65,536 encoder positions per part); items past a limit are dropped and counted.

The AC part of one solver run (every phase): ac_part([{"role": "user", "content": <the run rendered as text>}], AC, r) -- the same
text the teacher reads (render_run, 40k rule), in the same one-user-message form as the shared FineWeb warm-up's passages
(generate_reconstruction_samples). rows = clamp(ceil(encoder tokens x r), 2, 4,096): r is the fraction of the run's own length,
as the paper's M1 (ratio 0.1).

  lcrec   long-context reconstruction (compaction/stage1/LCREC.md, applied to the math runs): one item per training run (1,500
          problems x 3 seeds), r drawn per run from {1/10, 1/5} (seeded), reader prompt = the warm-up's reconstruction prompt
          (system RECONSTRUCTION_SYSTEM_PROMPT, user "Passage:\\n" + part + "\\n\\n" + RECONSTRUCTION_INSTRUCTION), target = the run
          text (SFT). Held panel: 50 DEV runs (seeded) at 1/10 and 1/5 (dev runs never train; nothing of the training material is
          held out).
  stage1  KL distillation, one packet per ratio r: one item per (training problem, teacher sample) whose sample stopped
          (finish "stop"). student = [user: task text, instruction, per agent "Agent i\\nIts run:\\n" + PART + "\\n\\nFinal answer:
          <result>", final rule] + the sample; teacher = [user: the full-runs prompt] + the sample (the exact prompt the sample was
          drawn from: asserted token-equal); targets = the sample's own tokens (+ <|im_end|> when vLLM did not return it); loss kl.
          The student's text equals the teacher's with each run replaced by its part (asserted).
  stage2  plain cross-entropy (SFT), one packet per r: per training problem, of its k teacher samples that stopped and are correct
          (lenient parse, math-verify: the teacher-test scorer), the shortest and every other within 1.3x its generated tokens;
          the same student view.
  panel   the dev proxy panel: per dev problem, the teacher's seed-0 dev synthesis (synth_gen.py on dev, stopped), teacher-forced:
          mdev/r10 and mdev/r5 (AC views), mdev/none (the results-only prompt: the paper's text arm). Files panel_r10.pkl,
          panel_r5.pkl, panel_none.pkl, panel.json.

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo> CONTINUATION_BENCH_DIR=<repo>/IB/TMP/PAPER_CORPUS_QA/kit/bench \\
      .venv/bin/python -I build_packets.py lcrec  --out PACKETS/m_lcrec_v1
      .venv/bin/python -I build_packets.py stage1 --r 0.1 --gen GEN_DIR --out PACKETS/m_s1_r10_v1
      .venv/bin/python -I build_packets.py stage2 --r 0.1 --gen GEN_DIR --out PACKETS/m_s2_r10_v1
      .venv/bin/python -I build_packets.py panel  --gen DEV_GEN_DIR --out PANELS/m_dev_panel_v1
"""
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import unicodedata
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
os.environ.setdefault("CONTINUATION_BENCH_DIR", str(ROOT / "IB/TMP/PAPER_CORPUS_QA/kit/bench"))
for p in (ROOT, Path(os.environ["CONTINUATION_BENCH_DIR"]), ROOT / "IB/TMP/PAPER_COMPACTION/kit", ROOT / "IB/TMP/PAPER_TRAJ_QA/kit",
          ROOT / "IB/ARTIFACTS/PRE_REBUTTAL/math/teacher_test"):
    sys.path.insert(0, str(p))

M = ROOT / "IB/TMP/PRE_REBUTTAL/math_recipe"
DATA = M / "data"
INIT = ROOT / "IB/TMP/PRE_REBUTTAL/compaction_stage1/init/R_res_rq9fr_final"     # s3 .../sec35/runs/R_res_rq9fr/sync/ckpt/final (the shared warm-up)
READER_LIMIT, ENCODER_LIMIT = 72_000, 65_536
CHUNK = 4
SEED = 20261117


def load_problems(split: str) -> dict:
    return {r["task_id"]: r for r in map(json.loads, open(DATA / f"problems_{split}.jsonl"))}


def load_gen(folder: Path, arm: str | None = None) -> list[dict]:
    rows = {}
    for path in sorted(Path(folder).glob("gen_*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if arm is None or r["arm"] == arm:
                rows[r["id"]] = r
    return list(rows.values())


def setup():
    """The pair on CPU (tokenizers, row limits), as prepare_agent_continuation.py builds it."""
    import run_qa
    from activation.ac_model import ActivationContextStudyGenerator
    h = run_qa.harness(str(INIT / "ac"), str(INIT / "reader"))
    study = ActivationContextStudyGenerator(h, run_qa.AC)
    return h, study


def run_part(text: str, ac_name: str, r: float) -> dict:
    from activation.common.ac_parts import ac_part
    return ac_part([{"role": "user", "content": text}], ac_name, r)


def synth_views(problem: dict, ac_name: str, r: float) -> tuple[str, list]:
    """(the full-runs prompt text, the student's content list with each run replaced by its part)."""
    from build_prompts import INSTRUCTION_FULL
    from run_synth1call_v2 import FINAL_RULE
    content, text = [], problem["task_body"] + "\n\n" + INSTRUCTION_FULL + "\n\n"
    for position, (run, result) in enumerate(zip(problem["runs"], problem["solver_results"]), start=1):
        content += [{"type": "text", "text": text + f"Agent {position}\nIts run:\n"}, run_part(run, ac_name, r)]
        text = f"\n\nFinal answer: {result}\n\n"
    content.append({"type": "text", "text": text + FINAL_RULE["deepmath"]})
    flat = "".join(piece["text"] if piece.get("type") == "text" else piece["messages"][0]["content"] for piece in content)
    return flat, content


def target_ids(tokenizer, sample: dict) -> list[int]:
    """The sample's own tokens; a stopped reply ends with <|im_end|> (appended when vLLM left it out)."""
    ids = list(sample["token_ids"])
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if sample["finish"] == "stop" and (not ids or ids[-1] != end):
        if sample.get("stop_reason") not in (None, end):
            raise ValueError(f"{sample['id']}: stopped on {sample.get('stop_reason')!r}, not <|im_end|>")
        ids.append(end)
    return ids


def synth_item(problem: dict, sample: dict, r: float, item_id: str, group: str, ac_name: str, tokenizer, student_text: str | None = None):
    """One synthesis item: student = the AC view (or a plain-text prompt: `student_text`), teacher = the full-runs prompt."""
    from activation.ac_model import ActivationContextTrainingItem
    full, content = synth_views(problem, ac_name, r)
    output = {"role": "assistant", "content": sample["text"]}
    teacher = [{"role": "user", "content": full}]
    student = [{"role": "user", "content": student_text if student_text is not None else content}]
    return ActivationContextTrainingItem(
        item_id=item_id, kind="synthesis", in_context_messages=teacher + [deepcopy(output)], ac_messages=student + [deepcopy(output)],
        in_context_start=1, ac_start=1, assistant_token_ids=[target_ids(tokenizer, sample)], tools=None, weight=1.0,
        dataset_id="deepmath_synthesis", doc_ids=[problem["task_id"]],
        info={"family": "math_synthesis", "source": "math_synthesis", "panel_group": group, "chat_template_kwargs": {"enable_thinking": True},
              "task_id": problem["task_id"], "split": problem["split"], "k": problem["k"], "difficulty": problem["difficulty"], "sample_id": sample["id"],
              "sample_seed": sample["seed"], "generated_tokens": sample["generated_tokens"], "r": r, "ratio": r if student_text is None else None,
              "full_prompt_sha256": hashlib.sha256(full.encode()).hexdigest()})


def recon_item(problem: dict, seed: int, r: float, item_id: str, group: str, ac_name: str, tokenizer):
    """The warm-up's reconstruction item over one run (compaction/stage1/build_lcrec.py's layout, the run text as the passage)."""
    from activation.ac_model import ActivationContextTrainingItem
    from activation.ac_model.ac_model_study import RECONSTRUCTION_INSTRUCTION, RECONSTRUCTION_SYSTEM_PROMPT
    from activation.ac_model.ac_model_utils import assistant_target_ids
    text = problem["runs"][seed]
    for bad in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"):
        if bad in text:
            raise ValueError(f"{item_id}: run text holds {bad}")
    output = {"role": "assistant", "content": text}
    ids = assistant_target_ids(tokenizer, output)
    if unicodedata.normalize("NFC", tokenizer.decode(ids)) != unicodedata.normalize("NFC", text + "<|im_end|>"):
        raise ValueError(f"{item_id}: the target tokens do not decode to the run text + <|im_end|>")
    system = [{"role": "system", "content": RECONSTRUCTION_SYSTEM_PROMPT}]
    content = [{"type": "text", "text": "Passage:\n"}, run_part(text, ac_name, r), {"type": "text", "text": f"\n\n{RECONSTRUCTION_INSTRUCTION}"}]
    return ActivationContextTrainingItem(
        item_id=item_id, kind="reconstruct",
        in_context_messages=system + [{"role": "user", "content": f"Passage:\n{text}\n\n{RECONSTRUCTION_INSTRUCTION}"}, output],
        ac_messages=system + [{"role": "user", "content": content}, deepcopy(output)], in_context_start=2, ac_start=2,
        assistant_token_ids=[ids], tools=None, dataset_id="math_long_reconstruction", doc_ids=[problem["task_id"]],
        info={"family": "math_long_reconstruction", "source": "math_long_reconstruction", "panel_group": group, "variant": "reconstruct",
              "task_id": problem["task_id"], "split": problem["split"], "seed": seed, "r": r, "ratio": r})


def build_all(h, study, items: list, workers: int) -> tuple[list, list]:
    """Every item through the trainer's builder and limits (fork workers); (built items, failures). Row counts are checked."""
    import run_channel_mixture as mix
    from run_channel_mixture import ForkPool
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    mix.READER_LIMIT = READER_LIMIT
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(loss_kind="sft", learning_rate_target_lora=0.0, examples_per_update=8,
                                       num_epochs=1, completion_samples=0, seed=mix.SEED, schedule_total_updates=max(1, len(items) // 8)))
    ac = study.ac_model
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mix.BUILD_CONTEXT = {"trainer": trainer, "study": study, "scan": True, "output": items}
    built, failures = {}, []
    try:
        with ForkPool(workers) as pool:
            for results in pool.map(mix.example_chunk, [range(i, min(i + CHUNK, len(items))) for i in range(0, len(items), CHUNK)]):
                for value, failure in results:
                    if failure:
                        failures.append(failure)
                        continue
                    item, example = value[:2]
                    rows = [end - start for start, end in example.spans]
                    want = [ac.part_view_rows(req.messages, req.compression_ratio, req.tools) for req in example.part_requests]
                    if rows != want or any(req.compression_ratio != item.info["r"] for req in example.part_requests):
                        raise ValueError(f"{item.item_id}: rows {rows} vs {want}")
                    item.info.update(rows=rows, side_tokens=[ac.side_token_count(req.messages, [], req.tools) for req in example.part_requests],
                                     clamped=sum(n >= ac.config.max_view_rows for n in rows))
                    built[item.item_id] = item
    finally:
        mix.BUILD_CONTEXT = None
    return [built[i.item_id] for i in items if i.item_id in built], failures


def check_teacher_prompt(h, study, item, prompt_ids: list[int]) -> None:
    """The teacher view's prompt tokens equal the ids the sample was generated from."""
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(loss_kind="kl", learning_rate_target_lora=0.0, max_example_tokens=10 ** 7))
    example = trainer.build_example(study.ac_model, item)
    n = len(prompt_ids)
    if example.teacher_ids[:n] != prompt_ids or example.teacher_ids[n:] != item.assistant_token_ids[0]:
        raise ValueError(f"{item.item_id}: the teacher view is not the generation prompt + the sample")


def quantiles(values: list) -> dict:
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values), "p90": values[int(0.9 * (len(values) - 1))], "max": values[-1],
            "mean": round(statistics.fmean(values), 1)} if values else {"n": 0}


def dump(path: Path, obj) -> str:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct(sample: dict) -> bool:
    """The teacher test's scorer (score_teacher.py): lenient parse, a length stop is unsolved, math-verify against gold + aliases."""
    import score_synth_lenient as lenient
    from score_synth1call import score
    if sample["finish"] != "stop":
        return False
    return score(sample["dataset_task"], lenient.readings("deepmath", sample["text"])["lenient"][0]) >= 1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=("lcrec", "stage1", "stage2", "panel"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--r", type=float, default=None, help="stage1/stage2: the packet's one ratio (0.1 or 0.2)")
    p.add_argument("--gen", type=Path, help="synth_gen.py output folder (stage1/stage2: train; panel: dev)")
    p.add_argument("--gen-extra", type=Path, default=None,
                   help="stage2 hard8 (2026-11-18): a second synth_gen folder with more teacher samples (seeds 3-7) of the training problems at "
                        ">= --extra-min-difficulty; they join the selection of those problems only (the others keep their 3 samples)")
    p.add_argument("--extra-min-difficulty", type=float, default=8.5)
    p.add_argument("--random-min-difficulty", type=float, default=None,
                   help="stage2 variant (2026-11-19): on problems at >= this difficulty keep ONE correct sample drawn uniformly at random "
                        "(seeded by task id) instead of the shortest + within --within; the other problems keep the shortest rule")
    p.add_argument("--arm", default=None, help="the synth_gen arm to read (default: every arm in --gen)")
    p.add_argument("--within", type=float, default=1.3, help="stage2: keep correct samples up to this multiple of the shortest")
    p.add_argument("--held", type=int, default=50, help="lcrec: dev runs in the reconstruction panel")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="first N problems only (a smoke packet)")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(f"{a.out} exists: use a new name")
    splits = json.loads((DATA / "splits.json").read_text())
    excluded = set(splits["dev"]) | set(splits["test"]["dmext"]) | set(splits["test"]["dmhard"])
    train_ids = set(splits["train"])
    h, study = setup()
    tok, ac_name = study.tokenizer, study.ac_name
    items, held, report = [], [], {"phase": a.phase, "seed": SEED, "init_pair": str(INIT), "data_sha256": {n: (DATA / f"{n}.sha256").read_text().split()[0]
                                                                                                           for n in ("problems_train.jsonl", "problems_dev.jsonl", "splits.json")}}
    if a.phase == "lcrec":
        train, dev = load_problems("train"), load_problems("dev")
        tasks = sorted(train)[:a.limit] if a.limit else sorted(train)
        left_out = []                                             # runs whose text holds a chat-control string (cannot be a plain-text target)
        for task_id in tasks:
            for seed in (0, 1, 2):
                r = (0.1, 0.2)[random.Random(f"{SEED}:lcrec:{task_id}:{seed}").randrange(2)]
                try:
                    items.append(recon_item(train[task_id], seed, r, f"mlcrec:{task_id}#s{seed}", "mlcrec/train", ac_name, tok))
                except ValueError as error:
                    left_out.append(str(error)[:200])
        report["left_out_runs"] = left_out
        clean = [(t, s) for t in dev for s in (0, 1, 2) if len(dev[t]["runs"][s]) < 100_000           # the panel must build whole: runs under 100k characters
                 and not any(b in dev[t]["runs"][s] for b in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"))]
        keys = random.Random(f"{SEED}:lcrec:held").sample(sorted(clean), a.held)
        for task_id, seed in sorted(keys):
            for label, r in (("r10", 0.1), ("r5", 0.2)):
                held.append(recon_item(dev[task_id], seed, r, f"panel:mlcrec/{label}:{task_id}#s{seed}", f"mlcrec/{label}", ac_name, tok))
    elif a.phase in ("stage1", "stage2"):
        assert a.r in (0.1, 0.2) and a.gen, "--r 0.1|0.2 and --gen"
        train = load_problems("train")
        samples = [s for s in load_gen(a.gen, a.arm) if s["task_id"] in train]
        assert all(s["seed"] in (0, 1, 2) for s in samples), "--gen holds the three base samples (seeds 0-2)"
        if a.gen_extra:
            assert a.phase == "stage2", "--gen-extra is a stage-2 option"
            extra = [s for s in load_gen(a.gen_extra) if s["task_id"] in train]
            assert all(s["seed"] >= 3 for s in extra), "--gen-extra holds seeds >= 3 only"
            assert all(float(train[s["task_id"]]["difficulty"]) >= a.extra_min_difficulty for s in extra), "an extra sample below the difficulty floor"
            hard = {t for t in train if float(train[t]["difficulty"]) >= a.extra_min_difficulty}
            per = Counter(s["task_id"] for s in extra)
            report["extra"] = {"folder": str(a.gen_extra), "samples": len(extra), "problems": len(per), "hard_problems": len(hard),
                               "hard_problems_without_extra": len(hard - set(per)), "samples_per_problem": dict(Counter(per.values())),
                               "finish": dict(Counter(s["finish"] for s in extra))}
            samples += extra
        prompts = {r["task_id"]: r["prompt_token_ids"] for r in map(json.loads, open(DATA / "prompts_train.jsonl")) if r["condition"] == "full"}
        by_task: dict = {}
        for s in samples:
            by_task.setdefault(s["task_id"], []).append(s)
        tasks = sorted(by_task)[:a.limit] if a.limit else sorted(by_task)
        report["samples"] = {"read": len(samples), "problems": len(by_task), "finish": dict(Counter(s["finish"] for s in samples))}
        chosen = []
        if a.phase == "stage1":
            for task_id in tasks:
                chosen += [s for s in sorted(by_task[task_id], key=lambda s: s["seed"]) if s["finish"] == "stop"]
        else:
            kept_by_k, n_correct = Counter(), 0
            for task_id in tasks:
                good = sorted((s for s in by_task[task_id] if correct(s)), key=lambda s: (s["generated_tokens"], s["seed"]))
                n_correct += len(good)
                if good and a.random_min_difficulty is not None and float(train[task_id]["difficulty"]) >= a.random_min_difficulty:
                    keep = [random.Random(f"{SEED}:random_correct:{task_id}").choice(good)]
                    chosen += keep
                    kept_by_k["random1"] += 1
                elif good:
                    keep = [s for s in good if s["generated_tokens"] <= a.within * good[0]["generated_tokens"]]
                    chosen += keep
                    kept_by_k[str(len(keep))] += 1
            if a.gen_extra:                                       # problems with >= 1 correct sample: out of the 3 base samples vs out of all, by band
                bands = {}
                for task_id in tasks:
                    band = "d8.5" if float(train[task_id]["difficulty"]) == 8.5 else ("ge9" if float(train[task_id]["difficulty"]) >= 9 else "lt8.5")
                    ok3 = any(correct(s) for s in by_task[task_id] if s["seed"] <= 2)
                    okall = any(correct(s) for s in by_task[task_id])
                    b = bands.setdefault(band, {"problems": 0, "correct_of_3": 0, "correct_of_all": 0, "samples": 0})
                    b["problems"] += 1; b["correct_of_3"] += ok3; b["correct_of_all"] += okall; b["samples"] += len(by_task[task_id])
                report["bands"] = bands
            report["selection_random_min_difficulty"] = a.random_min_difficulty
            report["selection"] = {"rule": f"per problem: the shortest correct sample (lenient parse, math-verify) and every other correct one within {a.within}x its generated tokens"
                                           + (f"; at difficulty >= {a.random_min_difficulty}: one correct sample drawn at random (seeded)" if a.random_min_difficulty is not None else ""),
                                   "correct_samples": n_correct, "problems_with_a_correct_sample": sum(kept_by_k.values()), "problems_by_kept_count": dict(kept_by_k)}
        label = "r10" if a.r == 0.1 else "r5"
        for s in chosen:
            items.append(synth_item(train[s["task_id"]], s, a.r, f"m{a.phase[-1]}:{s['task_id']}#s{s['seed']}", f"m{a.phase}/{label}", ac_name, tok))
        for item in items[:24]:                                   # the teacher view = the generation prompt + the sample (spot check of 24)
            flat, _ = synth_views(train[item.info["task_id"]], ac_name, a.r)
            check_teacher_prompt(h, study, item, prompts[item.info["task_id"]])
        for item in items:
            flat, content = synth_views(train[item.info["task_id"]], ac_name, a.r)
            if item.in_context_messages[0]["content"] != flat:
                raise ValueError(f"{item.item_id}: the student's text with runs restored is not the teacher's prompt")
    else:
        assert a.gen, "--gen (the dev teacher syntheses)"
        dev = load_problems("dev")
        results_prompt = {r["task_id"]: r["prompt"] for r in map(json.loads, open(DATA / "prompts_dev.jsonl")) if r["condition"] == "results"}
        samples = {s["task_id"]: s for s in load_gen(a.gen, a.arm) if s["seed"] == 0 and s["task_id"] in dev}
        report["samples"] = {"seed0": len(samples), "finish": dict(Counter(s["finish"] for s in samples.values()))}
        for task_id in sorted(samples):
            s = samples[task_id]
            if s["finish"] != "stop":
                continue
            for label, r in (("r10", 0.1), ("r5", 0.2)):
                items.append(synth_item(dev[task_id], s, r, f"mdev/{label}:{task_id}", f"mdev/{label}", ac_name, tok))
            items.append(synth_item(dev[task_id], s, 0.1, f"mdev/none:{task_id}", "mdev/none", ac_name, tok, student_text=results_prompt[task_id]))
    tasks_out = {i.info["task_id"] for i in items + held}
    if a.phase == "panel":
        assert not tasks_out & (train_ids | set(splits["test"]["dmext"]) | set(splits["test"]["dmhard"])), "a non-dev problem in the panel"
    else:
        assert not {i.info["task_id"] for i in items} & excluded, "a dev/test problem in the training items"
        assert {i.info["task_id"] for i in items} <= train_ids
        assert not {i.info["task_id"] for i in held} & train_ids, "a training problem in the held panel"
    built, failures = build_all(h, study, items + held, a.workers)
    held_ids = {i.item_id for i in held}
    held_built = [i for i in built if i.item_id in held_ids]
    if len(held_built) != len(held):
        raise ValueError(f"held items failed: {[f for f in failures if f['id'].startswith('panel:')][:3]}")
    built = [i for i in built if i.item_id not in held_ids]
    if a.phase == "panel":                                        # a dev problem enters the panel only with all three views
        complete = {t for t, n in Counter(i.info["task_id"] for i in built).items() if n == 3}
        report["panel_dropped_problems"] = sorted({i.info["task_id"] for i in built} - complete | {f["id"].split(":", 1)[1] for f in failures})
        built = [i for i in built if i.info["task_id"] in complete]
    a.out.mkdir(parents=True)
    report.update({"items_requested": len(items), "items": len(built), "dropped": dict(Counter(f["error"][:90] for f in failures)), "failures": failures[:50],
                   "problems": len({i.info["task_id"] for i in built}), "by_group": dict(Counter(i.info["panel_group"] for i in built)),
                   "target_tokens": quantiles([i.info["target_tokens"] for i in built]), "reader_tokens": quantiles([i.info["reader_tokens"] for i in built]),
                   "rows_per_item": quantiles([sum(i.info["rows"]) for i in built if i.info.get("rows")]),
                   "clamped_parts": sum(i.info.get("clamped", 0) for i in built), "r_counts": dict(Counter(str(i.info["r"]) for i in built))})
    if a.phase == "panel":
        for label in ("r10", "r5", "none"):
            report[f"panel_{label}_sha256"] = dump(a.out / f"panel_{label}.pkl", {"items": [i for i in built if i.info["panel_group"] == f"mdev/{label}"]})
        (a.out / "panel.json").write_text(json.dumps(report, indent=1))
    else:
        updates = len(built) // 8
        report.update({"updates_per_pass": updates, "leftover_items": len(built) - 8 * updates,
                       "recipe": {"loss": "sft" if a.phase in ("lcrec", "stage2") else "kl", "encoder_lr": 5e-5, "reader_lr": 0, "warmup": 50,
                                  "schedule": "cosine", "batch": 8, "passes": 1}})
        report["items_sha256"] = dump(a.out / "items.pkl", {"train": built, "held": []})
        if held_built:
            report["held_panel_sha256"] = dump(a.out / "held_panel.pkl", {"items": held_built})
            report["held_items"] = len(held_built)
        (a.out / "prepared.json").write_text(json.dumps({"training_examples": len(built), "held_examples": len(held_built), "items_sha256": report["items_sha256"],
                                                         "builder": "math/recipe/build_packets.py " + a.phase}, indent=1))
        (a.out / "build.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
