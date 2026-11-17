"""
Teacher-forced garble probe (2026-11-17), one GPU, read-only: does a pair's reader close its thinking with "[/" instead of
"</think>"? Hypothesis under test: the encoder's view escapes past reasoning delimiters to "[Reasoning]" / "[/Reasoning]"
(ac_model_utils._training_template_message, reasoning_as_text), and the long-context reconstruction stage (build_lcrec.py, target =
that rendering) taught the pair to make the reader reproduce "[/Reasoning]".

On the stage-1 proxy panel (build_dev_panel.py: the 75 dev-75 prefixes with their ORIGINAL continuations, dev75/r5 = 4,000 rows,
dev75/r10 = 2,000 rows), scored exactly as cmp_proxy_eval.py scores it (the trainer's own eval forward, here with loss_kind sft so
no stored teacher is needed), at every target position whose gold token is "</think>" (the reader closing its thinking) or
"<tool_call>" (the first token of a call), it records the reader's log-prob of "</think>", "[/", "[" and "<tool_call>" and whether
the gold token is top-1. --references adds the pair-independent views of a frozen reader: dev75/none (panel_none.pkl, no rows)
and full (the teacher view: head + the whole prefix, no rows).

Run inside the stage-1 code bundle's environment (as cmp_ckpt_proxy_job.sh: SEC45_KIT, PYTHONPATH bundle:kit:rq_kit:bench, ...):
    python probe_think.py --pair AC READER --label NAME --panel PANEL --out OUT.json [--references] [--limit N]
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

KIT = os.environ.get("SEC45_KIT")
if KIT:
    sys.path.insert(0, KIT)
import sec45_common as S  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, required=True); p.add_argument("--label", required=True)
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--references", action="store_true"); p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    started = time.monotonic()
    S.setup_environment("res")
    import torch
    import torch.nn.functional as F
    S.require_cuda(torch)
    from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.agent_training.agent_training_utils import head_logits
    from run_qa import AC, harness
    S.check_pair_mode(a.pair[0], a.pair[1], "res")
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    ac = ActivationContextStudyGenerator(h, AC).ac_model
    assert ac.is_deep, "not a deep (residual) pair"
    with open(a.panel / "panel_ac.pkl", "rb") as f:
        items = list(pickle.load(f)["items"])
    if a.references:
        with open(a.panel / "panel_none.pkl", "rb") as f:
            items += list(pickle.load(f)["items"])
    if a.limit:
        keep = sorted({i.info["task_id"] for i in items})[:a.limit]
        items = [i for i in items if i.info["task_id"] in keep]
    groups = {i.item_id: i.info["panel_group"] for i in items}
    tok = ac.target.tokenizer
    ids = {"</think>": tok.convert_tokens_to_ids("</think>"), "<tool_call>": tok.convert_tokens_to_ids("<tool_call>")}
    for text in ("[/", "["):
        encoded = tok.encode(text, add_special_tokens=False)
        assert len(encoded) == 1, f"{text!r} is {encoded}"
        ids[text] = encoded[0]
    print(f"probe_think: token ids {ids}", flush=True)
    watched = {ids["</think>"]: "think", ids["<tool_call>"]: "call"}
    config = ActivationContextTrainingConfig(loss_kind="sft", learning_rate_target_lora=0.0, max_example_tokens=72000, micro_batch_examples=1,
                                             completion_samples=0, deterministic=False, distributed=False)
    trainer = ActivationContextTrainer(h, config)
    records = []
    current = {"item": None}

    def capture(states, head, targets, group, item_id):
        picks = [i for i, t in enumerate(targets.tolist()) if t in watched]
        if not picks:
            return
        with torch.no_grad():
            index = torch.tensor(picks, device=states.device)
            logp = F.log_softmax(head_logits(states.index_select(0, index), head, config.fp32_head_matmul).float(), dim=-1)
            top = logp.argmax(dim=-1).tolist()
            chosen = {name: logp[:, token].tolist() for name, token in ids.items()}
        gold = targets[index].tolist()
        for row, (position, token) in enumerate(zip(picks, gold)):
            records.append({"item": item_id, "group": group, "kind": watched[token], "pos": position,
                            **{name: round(values[row], 4) for name, values in chosen.items()}, "gold_top1": top[row] == token,
                            "top1": tok.convert_ids_to_tokens(top[row])})

    original_sft = trainer._chunked_sft

    def chunked_sft(states, head, targets, weights=None):
        item = current["item"]
        capture(states, head, targets, groups[item], item)
        return original_sft(states, head, targets, weights)

    original_loss = trainer._example_loss

    def example_loss(ac_model, example, *args, **kwargs):
        current["item"] = example.item.item_id
        return original_loss(ac_model, example, *args, **kwargs)

    trainer._chunked_sft, trainer._example_loss = chunked_sft, example_loss
    summary = trainer.eval(AC, items, teacher="current", reporting_name="probe", release=False)
    eval_seconds = time.monotonic() - started
    if a.references:                                       # the full-context teacher view (the frozen reader, no rows), once per prefix
        target = ac.target
        base = target.model
        embedding, head = base.get_input_embeddings(), base.get_output_embeddings()
        lora = ac.config.target_model_lora_name
        base.eval()
        seen = set()
        import dataclasses
        teacher_view = ActivationContextTrainer(h, dataclasses.replace(config, loss_kind="kl", teacher_cache_top_k=32))   # builds the teacher view too (ids only)
        with torch.no_grad():
            for item in items:
                if groups[item.item_id] != "dev75/r5" or item.info["task_id"] in seen:
                    continue
                seen.add(item.info["task_id"])
                example = teacher_view.build_example(ac, item)
                assert example.teacher_ids and len(example.teacher_positions) == len(example.target_ids), "no teacher view"
                device = embedding.weight.device
                embeds = embedding(torch.tensor(example.teacher_ids, device=device, dtype=torch.long))
                hidden = target.decoder_forward(embeds[None], None, lora_name=lora)[0]
                capture(hidden[example.teacher_positions], head, torch.tensor(example.target_ids, device=device), "full", f"full:{item.info['task_id']}")
    table = defaultdict(lambda: defaultdict(list))
    for r in records:
        table[r["group"]][r["kind"]].append(r)
    out = {"label": a.label, "pair": a.pair, "token_ids": ids, "items": len(items), "eval_gold_nll_token": summary.get("gold_nll_token"),
           "groups": {}, "eval_seconds": round(eval_seconds, 1), "wall_seconds": round(time.monotonic() - started, 1)}
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows) if rows else None
    share = lambda rows, test: 100 * sum(1 for r in rows if test(r)) / len(rows) if rows else None
    for group, kinds in sorted(table.items()):
        out["groups"][group] = {}
        for kind, rows in sorted(kinds.items()):
            other = "</think>" if kind == "think" else "<tool_call>"
            out["groups"][group][kind] = {
                "positions": len(rows), "mean_lp_think": mean(rows, "</think>"), "mean_lp_bracket_slash": mean(rows, "[/"),
                "mean_lp_bracket": mean(rows, "["), "mean_lp_tool_call": mean(rows, "<tool_call>"),
                "pct_bracket_slash_over_gold": share(rows, lambda r: r["[/"] > r[other]), "pct_gold_top1": share(rows, lambda r: r["gold_top1"]),
                "pct_top1_bracket_slash": share(rows, lambda r: r["top1"] == "[/"),
                "top1_counts": dict(sorted(((t, sum(1 for r in rows if r["top1"] == t)) for t in {r["top1"] for r in rows}), key=lambda x: -x[1])[:6])}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    with open(a.out.with_suffix(".positions.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
