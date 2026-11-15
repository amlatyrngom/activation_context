"""
The held-out proxy panel of compaction stage 1: dev-75 (`prefixes_dev9_75`) with each run's ORIGINAL continuation (the
untuned 9B's own steps after the cut, from `pc_pool9_eval_v3`), as trainer items. CPU only (CUDA_VISIBLE_DEVICES= ).

Per dev prefix, three student views of the same continuation, all scored against the same teacher (the frozen reader
reading head + full prefix, i.e. the untuned 9B: the R_res_rq9fr reader adapter is all zeros):

    dev75/r5     head + [user: AC part of head + prefix at 1/5, COMPACTION_INSTRUCTIONS] + continuation   (4,000 rows)
    dev75/r10    the same at 1/10                                                                         (2,000 rows)
    dev75/none   head + [user: the `none` arm's note that nothing was kept] + continuation               (no rows)

The AC views are built by prepare_agent_continuation.build_item exactly as the training items (sizes against the nominal
20,000-token prefix, as the benchmark's ac_4000 / ac_2000 arms). The `none` view is the eval-200 `none` arm's prompt
(kit/resume_program.py: head + NOTHING_KEPT_NOTE). Targets: every assistant turn of the continuation (the recorded tokens).

Input: the eval-side material from export_pool_material.py (same cut rule as the dev-75 index: window 20,000, run under
50,000, not capped, unsolved runs kept); each dev entry is matched by task id and must have the index's `prefix_steps`.
Output folder: panel_ac.pkl {"items": r5 + r10 items}, panel_none.pkl {"items": none items}, panel.json (counts, hashes).

    CONTINUATION_BENCH_DIR=... CONTINUATION_RUN_QA_DIR=<bundle>/bench CUDA_VISIBLE_DEVICES= python build_dev_panel.py \
        --material eval_v3_c20.jsonl --index prefixes_dev9_75/index.json --initial-ac PAIR/ac --initial-reader PAIR/reader --out PANEL
"""
import argparse
import hashlib
import json
import pickle
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import prepare_agent_continuation as P

# kit/resume_program.py NOTHING_KEPT_NOTE (the eval `none` arm), copied verbatim.
NOTHING_KEPT_NOTE = ("The conversation so far was compacted and no summary of it was kept. Continue the task; "
                     "what was done before is not available.")
SIZES = (("r5", 1 / 5), ("r10", 1 / 10))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True, help="prefixes_dev9_75/index.json")
    p.add_argument("--initial-ac", required=True)
    p.add_argument("--initial-reader", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--nominal-prefix-tokens", type=int, default=20_000)
    a = p.parse_args()
    if (a.out / "panel.json").exists():
        raise FileExistsError(f"{a.out}/panel.json exists: use a new folder")
    index = json.loads(a.index.read_text())["prefixes"]
    material = {}
    for line in a.material.open():
        if line.strip():
            record = json.loads(line)
            material[record["task_id"]] = record
    missing = [entry["task_id"] for entry in index if entry["task_id"] not in material]
    moved = [entry["task_id"] for entry in index if entry["task_id"] in material and material[entry["task_id"]]["teacher"]["prefix_steps"] != entry["prefix_steps"]]
    if missing or moved:
        raise ValueError(f"dev entries without material: {missing[:5]} ({len(missing)}); cut elsewhere than the index: {moved[:5]} ({len(moved)})")

    h = P.harness(a.initial_ac, a.initial_reader)
    study = P.ActivationContextStudyGenerator(h, P.AC)
    ac = study.ac_model
    trainer = P.ActivationContextTrainer(h, P.ActivationContextTrainingConfig(loss_kind="kl", learning_rate_target_lora=0.0, max_example_tokens=P.READER_TOKENS))
    ac_items, none_items = [], []
    for entry in index:
        record = material[entry["task_id"]]
        view = P.flattened_view(record)
        ids, _ = P.tokenize_with_parts(ac.side.tokenizer, view, [], ac.pad_id, tools=record["tools"] or None)
        for label, r in SIZES:
            item = P.build_item(record, view, len(ids), r, f"dev75/{label}:{record['task_id']}", f"dev75/{label}", ac, study.ac_name, a.nominal_prefix_tokens)
            item.info.update(panel=True, dev_index=str(a.index), solved_original=record["solved"])
            ac_items.append(item)
        last = ac_items[-1]
        head = deepcopy(record["head_messages"]) + [{"role": "user", "content": NOTHING_KEPT_NOTE}]
        continuation = last.ac_messages[last.ac_start:]
        none_items.append(replace(last, item_id=f"dev75/none:{record['task_id']}", ac_messages=head + deepcopy(continuation), ac_start=len(head),
                                  info={**last.info, "panel_group": "dev75/none", "r": 0.0, "ratio": None, "rows_wanted": 0, "rows": 0}))
    for item in ac_items + none_items:                # the trainer's own builder: limits, row counts, targets
        example = trainer.build_example(ac, item)
        if max(len(example.student_ids), len(example.teacher_ids)) > P.READER_TOKENS:
            raise ValueError(f"{item.item_id}: over {P.READER_TOKENS} reader tokens")
        rows = sum(end - start for start, end in example.spans)
        if rows != item.info["rows"]:
            raise ValueError(f"{item.item_id}: {rows} rows, the item says {item.info['rows']}")
        item.info.update(target_tokens=len(example.target_ids), reader_tokens=len(example.student_ids), teacher_tokens=len(example.teacher_ids))
    a.out.mkdir(parents=True, exist_ok=True)
    for name, items in (("panel_ac.pkl", ac_items), ("panel_none.pkl", none_items)):
        with open(a.out / name, "wb") as f:
            pickle.dump({"items": items}, f)
    sha = {name: hashlib.sha256((a.out / name).read_bytes()).hexdigest() for name in ("panel_ac.pkl", "panel_none.pkl")}
    summary = {"index": str(a.index), "material": str(a.material), "prefixes": len(index), "initial_pair": [a.initial_ac, a.initial_reader],
               "groups": {g: sum(i.info["panel_group"] == g for i in ac_items + none_items) for g in ("dev75/r5", "dev75/r10", "dev75/none")},
               "rows": {g: sorted({i.info["rows"] for i in ac_items if i.info["panel_group"] == g}) for g in ("dev75/r5", "dev75/r10")},
               "target_tokens": sum(i.info["target_tokens"] for i in none_items), "teacher_tokens_mean": round(sum(i.info["teacher_tokens"] for i in none_items) / len(none_items)),
               "solved_original": sum(i.info["solved_original"] for i in none_items), "sha256": sha, "none_note": NOTHING_KEPT_NOTE}
    (a.out / "panel.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
