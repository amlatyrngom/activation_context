"""
Proxy evaluator of compaction stage 1 (one GPU): how much of the full prefix's effect on the next steps a pair's AC rows
carry, on the held-out dev-75 prefixes and their ORIGINAL continuations (build_dev_panel.py), teacher-forced, no sampling.

Per checkpoint (a frozen-reader pair, e.g. R_res_rq9fr or a stage-1 checkpoint), per panel group, token-weighted over every
assistant target token of the 75 continuations:
    kl_token           mean per-token KL(teacher || student), the trainer's top-k KL (k = 32, the rest of the mass as one bucket)
    nll_token          the student's NLL of the original continuation's tokens
    teacher_nll_token  the teacher's NLL of the same tokens = NLL_full: the untuned 9B reading head + the full prefix
                       (the teacher is the pair's reader with its adapter; for a frozen-reader pair that adapter is all zeros)
Groups: dev75/r5, dev75/r10 (AC rows at 1/5, 1/10 of the prefix) and, with --references, dev75/none (the untuned 9B with the
head and the `none` arm's note: NLL_none). gap_closed[g] = (NLL_none - NLL_g) / (NLL_none - NLL_full).

The none group and the teacher do not depend on the encoder, so for a frozen reader they are computed once:
  --teacher-targets FILE   loaded when it exists (its reader-adapter hash must match this pair's), else computed and written
  --references-from JSON   an earlier output of this script holding dev75/none (instead of --references)
Deterministic kernels by default (needs CUBLAS_WORKSPACE_CONFIG=:4096:8 at process start); --no-deterministic turns them off.
The environment is the sec35 jobs' (sec45_common.setup_environment('res') with SEC45_ENCODER, as jobs/rq_eval_job.sh).

Older deep pairs (2026-11-16, the paper's compaction model H8a: 2B -> 9B, deltas at every full-attention layer, TRAINED reader adapter):
  --deltas-off             score the rows alone, the pair's per-layer deltas withheld (the paper served H8a this way: vLLM rows only)
  With a trained reader adapter the teacher is that adapted reader, so kl_token and teacher_nll_token are relative to it, not to the
  untuned 9B; nll_token (same target tokens) stays comparable. With --references-from, reference_nll_full / gap_closed_vs_reference
  use the reference file's NLL_full (the untuned 9B with the full prefix) instead of this pair's teacher.

    python cmp_proxy_eval.py --pair PAIR/ac PAIR/reader --panel PANEL --out OUT.json --teacher-targets T.pt [--references | --references-from REF.json]
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

KIT = os.environ.get("SEC45_KIT")                         # the bundle's kit/ (sec45_common); on PYTHONPATH in the jobs
if KIT:
    sys.path.insert(0, KIT)
import sec45_common as S  # noqa: E402


def adapter_hash(reader: str) -> str:
    h = hashlib.sha256()
    with open(Path(reader) / "adapter_model.safetensors", "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", nargs=2, required=True, metavar=("AC", "READER"))
    p.add_argument("--panel", type=Path, required=True, help="build_dev_panel.py's folder")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--teacher-targets", type=Path, required=True)
    p.add_argument("--references", action="store_true", help="also score dev75/none")
    p.add_argument("--references-from", type=Path, help="an earlier output holding dev75/none")
    p.add_argument("--limit", type=int, default=0, help="first N prefixes only (a quick check; 0 = all 75)")
    p.add_argument("--label", default="")
    p.add_argument("--deltas-off", action="store_true", help="a deep pair's rows only, its per-layer deltas withheld")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    started = time.monotonic()
    S.setup_environment("res")
    import torch
    S.require_cuda(torch)
    from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
    from run_qa import AC, harness
    S.check_pair_mode(a.pair[0], a.pair[1], "res")
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    if not ac.is_deep:
        raise SystemExit(f"{a.pair[0]} is not a deep (residual) model")
    with open(a.panel / "panel_ac.pkl", "rb") as f:
        items = list(pickle.load(f)["items"])
    if a.references:
        with open(a.panel / "panel_none.pkl", "rb") as f:
            items += list(pickle.load(f)["items"])
    if a.limit:
        keep = sorted({i.info["task_id"] for i in items})[:a.limit]
        items = [i for i in items if i.info["task_id"] in keep]
    groups = {i.item_id: i.info["panel_group"] for i in items}
    config = ActivationContextTrainingConfig(loss_kind="kl", learning_rate_target_lora=0.0, teacher_cache_top_k=32, max_example_tokens=72000,
                                             micro_batch_examples=1, completion_samples=0, deterministic=a.deterministic, distributed=False)
    trainer = ActivationContextTrainer(h, config)
    if a.deltas_off:                                       # the trainer's own deltas_off path (rows kept, delta payload withheld)
        import functools
        trainer._evaluate = functools.partial(trainer._evaluate, deltas_off=True)
    reader_sha = adapter_hash(a.pair[1])
    meta_file = a.teacher_targets.with_suffix(".json")
    teacher = "current"
    if a.teacher_targets.exists():
        meta = json.loads(meta_file.read_text())
        if meta["reader_adapter_sha256"] != reader_sha:
            raise SystemExit(f"{a.teacher_targets} was made with another reader adapter ({meta['reader_adapter_sha256'][:12]} vs {reader_sha[:12]})")
        trainer.load_teacher_targets(a.teacher_targets)
        teacher = "stored"
        keys = {trainer.build_example(ac, i).teacher_cache_key for i in items}
        if not keys <= set(trainer.teacher_cache):
            teacher = "current"                            # a few missing (e.g. a first run with --limit): stored ones are kept, the rest computed
    torch.cuda.reset_peak_memory_stats()
    phase = time.monotonic()
    summary = trainer.eval(AC, items, teacher=teacher, reporting_name="dev75")
    eval_seconds = time.monotonic() - phase
    if teacher == "current":
        trainer.save_teacher_targets(a.teacher_targets)
        meta_file.write_text(json.dumps({"reader_adapter_sha256": reader_sha, "pair": a.pair, "entries": len(trainer.teacher_cache), "top_k": 32}))
    metrics = S.group_metrics(summary["per_item"], groups)
    reference = json.loads(a.references_from.read_text()) if a.references_from else {}
    if a.references_from and "dev75/none" not in metrics:
        metrics["dev75/none"] = reference["groups"]["dev75/none"]
    full = [g["teacher_nll_token"] for name, g in metrics.items() if name != "dev75/none"]
    nll_full = full[0] if full else None
    gap = {}
    if "dev75/none" in metrics and nll_full is not None:
        none = metrics["dev75/none"]["nll_token"]
        gap = {name: (none - g["nll_token"]) / (none - nll_full) for name, g in metrics.items() if name != "dev75/none"}
    ref_full = reference.get("nll_full")
    gap_ref = {}
    if "dev75/none" in metrics and ref_full is not None:
        none = metrics["dev75/none"]["nll_token"]
        gap_ref = {name: (none - g["nll_token"]) / (none - ref_full) for name, g in metrics.items() if name != "dev75/none"}
    out = {"label": a.label, "pair": a.pair, "reader_adapter_sha256": reader_sha, "panel": str(a.panel), "items": len(items),
           "deltas_off": a.deltas_off, "reference_nll_full": ref_full, "gap_closed_vs_reference": gap_ref,
           "prefixes": len({i.info["task_id"] for i in items}), "groups": metrics, "nll_full": nll_full,
           "nll_none": metrics.get("dev75/none", {}).get("nll_token"), "gap_closed": gap, "teacher": teacher, "deterministic": a.deterministic,
           "item_mean_kl": summary.get("kl"), "eval_seconds": round(eval_seconds, 1), "wall_seconds": round(time.monotonic() - started, 1),
           "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "device": torch.cuda.get_device_name(0)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    with open(a.out.with_suffix(".items.jsonl"), "w") as f:
        for row in summary["per_item"]:
            f.write(json.dumps({"item_id": row["item_id"], "group": groups[row["item_id"]], "tokens": len(row["target_logp"]),
                                "kl_mean": row["kl"], "nll": row["gold_nll"], "teacher_nll": row["teacher_gold_nll"],
                                "student_tokens": row["student_tokens"], "teacher_tokens": row["teacher_tokens"]}) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "groups"} | {"groups": metrics}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
