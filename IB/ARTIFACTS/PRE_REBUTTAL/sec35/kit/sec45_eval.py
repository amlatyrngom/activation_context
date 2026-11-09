"""
Section 4.5 accuracy of one checkpoint: the 200 HotpotQA dev questions (build_hotpot_tasks.py dev200: top-20 BM25 with both golds
injected, ONE question-first encoder call at 1/10), one GREEDY sample each, scored with rq_common.score on the reply's last
"Final answer:" line (acc = exact match or the normalized gold inside the normalized prediction; also EM, token F1, answered).
Same batched HF decode as rq_eval.py (ActivationContextTrainer._sample_completions_batched), with completion_sampling None (greedy).

  python sec45_eval.py --mode MODE --data HOTPOT_DIR --pair AC_DIR READER_DIR --out OUT.jsonl [--label T_kv_u500] [--limit N]
  python sec45_eval.py --data HOTPOT_DIR --score OUT.jsonl            (score only)
OUT.jsonl gets one row per question (appended per batch; a rerun skips the ids already there); OUT.summary.json the scores.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sec45_common as S  # noqa: E402


def summarize(rows, by_id):
    import rq_common as C
    scores = [C.score(r["pred"], by_id[r["id"]]) for r in rows]
    n = len(scores)
    out = {k: round(100 * sum(s[k] for s in scores) / max(1, n), 2) for k in ("acc", "em", "f1", "answered")}
    return out | {"n": n, "length_stops": sum(r.get("length_stop", False) for r in rows),
                  "new_tokens_mean": round(sum(r.get("new_tokens", 0) for r in rows) / max(1, n), 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=sorted(S.MODES)); p.add_argument("--data", type=Path, required=True); p.add_argument("--split", default="dev200")
    p.add_argument("--pair", nargs=2, metavar=("AC_DIR", "READER_DIR")); p.add_argument("--out", type=Path); p.add_argument("--score", type=Path)
    p.add_argument("--label", default=""); p.add_argument("--limit", type=int); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-new", type=int, default=128, help="new-token cap (a trained reply is ~10 tokens; a cut reply scores wrong)")
    p.add_argument("--k", help="retrieval rows: passages read (SEC45_K; one value for an eval)")
    a = p.parse_args()
    if a.k:
        import os
        os.environ["SEC45_K"] = a.k
    started = time.monotonic()                     # the whole process after argument parsing (the planner scales load + decode from it)
    S.setup_environment(a.mode or "in")
    import rq_common as C
    rows = S.load_tasks(a.data, a.split)
    by_id = {r["id"]: r for r in rows}
    if a.score:
        print(json.dumps(summarize([json.loads(line) for line in open(a.score)], by_id)))
        return
    if not (a.mode and a.pair and a.out):
        raise SystemExit("--mode, --pair and --out are needed to decode")
    rows = rows[:a.limit] if a.limit else rows
    done = {json.loads(line)["id"] for line in open(a.out)} if a.out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    import torch
    S.require_cuda(torch)
    from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
    from activation.autotuners import configure_fla_runtime
    from run_qa import AC, harness
    S.check_pair_mode(a.pair[0], a.pair[1], a.mode)
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    corpus = C.load_corpus(a.data)
    items = S.task_items(study, corpus, todo, AC) if todo else []
    print(f"sec45_eval: {len(items)} questions to decode ({len(done)} already in {a.out}), greedy, batch {a.batch}, max_new {a.max_new}", flush=True)
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(loss_kind="sft", distributed=False, completion_sampling=None,
                                                                          completion_max_new_tokens=a.max_new, max_example_tokens=72000))
    tokenizer = ac.target.tokenizer
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def on_batch(indexes, samples):
        with open(a.out, "a") as f:
            for index, sample in zip(indexes, samples):
                text = sample["student"]
                new_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                pred = C.final_answer(text, need_think=False)
                f.write(json.dumps({"id": items[index].item_id, "pred": pred, "new_tokens": new_tokens, "length_stop": new_tokens >= a.max_new - 1,
                                    "text": text}) + "\n")
        print(f"sec45_eval: batch done ({time.monotonic() - started:.0f}s)", flush=True)

    stats = {}
    if items:
        progress = ActivationContextTrainingProgress(1, 1, 1, 0, 1, 0, 0., 0.)
        with configure_fla_runtime(trainer._autotuning_variables(ac)["fla_config"]):
            ac.prepare()
            h.module_manager.ensure_lora(ac.config.target_model_lora_name)
            for parameter in h.module_manager.lora_parameters(ac.config.target_model_lora_name):   # as rq_eval.py: no fp32 GEMMs in the decode
                parameter.data = parameter.data.to(torch.bfloat16 if torch.cuda.is_available() else parameter.dtype)
            decode_started = time.monotonic()
            trainer._sample_completions_batched(ac, items, progress, batch_size=a.batch, on_batch=on_batch, stats=stats)
            stats["decode_seconds"] = round(time.monotonic() - decode_started, 1)
    result = summarize([json.loads(line) for line in open(a.out)], by_id) | {"label": a.label, "pair": a.pair, "mode": a.mode, "decode": stats,
                                                                               "seconds": round(time.monotonic() - started)}
    summary_path = a.out.with_suffix(".summary.json")                 # atomic: the job's eval watcher deletes the pair once this exists
    summary_path.with_suffix(".tmp").write_text(json.dumps(result, indent=1)); summary_path.with_suffix(".tmp").replace(summary_path)
    print(f"sec45_eval: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
