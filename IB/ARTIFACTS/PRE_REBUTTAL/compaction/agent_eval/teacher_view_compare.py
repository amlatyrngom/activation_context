"""
Old vs corrected KL-teacher view, scored by the frozen reader (2026-11-17, compaction/TEACHER_VIEW_FIX.md), one GPU, read-only.

For every item of the given packets (one per prefix / run key: the teacher view does not depend on the item's size), the teacher
history is built twice with the trainer's own builder (ac_model_utils.prepare_training_history, as build_example calls it):
    old   earlier assistant turns rendered by the chat template with the reasoning escaped to "[Reasoning]" / "[/Reasoning]"
    new   earlier assistant turns as their recorded tokens (info["in_context_history_ids"]; the agent's runtime prompt)
and both are read by the untuned Qwen3.5-9B (the frozen-reader pair's zero adapter), teacher-forced on the same targets. Per
target position: the exact KL(old || new) over the full vocabulary, top-1 agreement, top-32 overlap (the trainer stores the top 32),
the gold token's log-prob (NLL_full = its token-weighted mean); at gold "</think>" positions the log-probs of "</think>" and "[/".
Per item: mean KL, and whether its stored top-32 targets would change materially (reported at several thresholds).

    python teacher_view_compare.py --pair AC READER --packet NAME=ITEMS.pkl [--packet ...] --out OUT.json [--limit N]
Run in the stage-1 bundle environment with the repository's activation package first on PYTHONPATH (it has history_ids).
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

KIT = os.environ.get("SEC45_KIT")
if KIT:
    sys.path.insert(0, KIT)
import sec45_common as S  # noqa: E402

TOP = 32


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, required=True); p.add_argument("--packet", action="append", required=True, help="NAME=items pickle")
    p.add_argument("--out", type=Path, required=True); p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    started = time.monotonic()
    S.setup_environment("res")
    import torch
    import torch.nn.functional as F
    S.require_cuda(torch)
    from activation.ac_model import ActivationContextStudyGenerator
    from activation.ac_model.ac_model_utils import prepare_training_history
    from activation.agent_training.agent_training_utils import head_logits
    import inspect
    assert "history_ids" in inspect.signature(prepare_training_history).parameters, "this activation package has no history_ids"
    from run_qa import AC, harness
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    ac = ActivationContextStudyGenerator(h, AC).ac_model
    target = ac.target
    h.module_manager.ensure_lora(ac.config.target_model_lora_name)
    from activation.harness.hf_utils import TARGET_DEVICE
    target.model_to_device(TARGET_DEVICE)
    base = target.model
    base.eval()
    embedding, head = base.get_input_embeddings(), base.get_output_embeddings()
    device = embedding.weight.device
    lora = ac.config.target_model_lora_name
    tok = target.tokenizer
    think, slash = tok.convert_tokens_to_ids("</think>"), tok.encode("[/", add_special_tokens=False)[0]
    report = {"pair": a.pair, "packets": {}}

    def forward(ids, positions):
        with torch.no_grad():
            hidden = target.decoder_forward(embedding(torch.tensor(ids, device=device, dtype=torch.long))[None], None, lora_name=lora)[0]
            return hidden[positions]

    for spec in a.packet:
        name, path = spec.split("=", 1)
        with open(path, "rb") as f:
            data = pickle.load(f)
        items = data["train"] if "train" in data else data["items"]
        chosen, seen = [], set()
        for item in items:
            key = item.info.get("run_key") or item.info.get("task_id")
            if key in seen or item.in_context_messages is None:
                continue
            seen.add(key)
            chosen.append(item)
        if a.limit:
            chosen = chosen[:a.limit]
        per_item = []
        for item in chosen:
            history = item.info.get("in_context_history_ids")
            assert history is not None, f"{item.item_id}: built without the recorded teacher view"
            kwargs = dict(tools=item.tools, chat_template_kwargs=item.info.get("chat_template_kwargs"))
            new_ids, _, new_pos = prepare_training_history(tok, item.in_context_messages, item.in_context_start, item.assistant_token_ids, [], history_ids=history, **kwargs)
            old_ids, _, old_pos = prepare_training_history(tok, item.in_context_messages, item.in_context_start, item.assistant_token_ids, [], **kwargs)
            gold = [t for turn in item.assistant_token_ids for t in turn]
            first_turn = len(item.assistant_token_ids[0])
            new_states, old_states = forward(new_ids, new_pos), forward(old_ids, old_pos)
            gold_t = torch.tensor(gold, device=device)
            kl, top1, overlap, lp_old, lp_new, rows = [], [], [], [], [], []
            with torch.no_grad():
                for start in range(0, len(gold), 512):
                    lo = F.log_softmax(head_logits(old_states[start:start + 512], head).float(), dim=-1)
                    ln = F.log_softmax(head_logits(new_states[start:start + 512], head).float(), dim=-1)
                    g = gold_t[start:start + 512]
                    kl.append((lo.exp() * (lo - ln)).sum(-1))
                    top1.append(lo.argmax(-1) == ln.argmax(-1))
                    to, tn = lo.topk(TOP, dim=-1).indices, ln.topk(TOP, dim=-1).indices
                    overlap.append((to[:, :, None] == tn[:, None, :]).any(-1).float().mean(-1))
                    lp_old.append(lo.gather(-1, g[:, None])[:, 0]); lp_new.append(ln.gather(-1, g[:, None])[:, 0])
                    for i in (g == think).nonzero()[:, 0].tolist():
                        rows.append({"pos": start + i, "first_turn": start + i < first_turn,
                                     "old": [round(float(lo[i, think]), 4), round(float(lo[i, slash]), 4)],
                                     "new": [round(float(ln[i, think]), 4), round(float(ln[i, slash]), 4)]})
            kl, top1, overlap, lp_old, lp_new = (torch.cat(x).float().cpu() for x in (kl, top1, overlap, lp_old, lp_new))
            per_item.append({"item": item.item_id, "key": item.info.get("run_key") or item.info.get("task_id"), "targets": len(gold),
                             "teacher_tokens_old": len(old_ids), "teacher_tokens_new": len(new_ids),
                             "kl_mean": float(kl.mean()), "kl_first_turn": float(kl[:first_turn].mean()), "kl_max": float(kl.max()),
                             "top1_agree": float(top1.float().mean()), "top32_overlap": float(overlap.mean()),
                             "nll_old_sum": float(-lp_old.sum()), "nll_new_sum": float(-lp_new.sum()), "think": rows})
            print(f"teacher_view_compare: {name} {per_item[-1]['key']} kl {per_item[-1]['kl_mean']:.4f} first {per_item[-1]['kl_first_turn']:.4f} "
                  f"top1 {per_item[-1]['top1_agree']:.3f} nll {per_item[-1]['nll_old_sum'] / len(gold):.4f} -> {per_item[-1]['nll_new_sum'] / len(gold):.4f}", flush=True)
        tokens = sum(r["targets"] for r in per_item)
        thinks = [t for r in per_item for t in r["think"]]

        def think_stats(rows, view):
            if not rows:
                return None
            return {"positions": len(rows), "mean_lp_think": sum(t[view][0] for t in rows) / len(rows), "mean_lp_slash": sum(t[view][1] for t in rows) / len(rows),
                    "pct_slash_over_think": 100 * sum(t[view][1] > t[view][0] for t in rows) / len(rows)}

        kls = sorted(r["kl_mean"] for r in per_item)
        quant = lambda q: kls[min(len(kls) - 1, int(q * len(kls)))]
        report["packets"][name] = {
            "path": path, "items": len(per_item), "target_tokens": tokens,
            "nll_full_old": sum(r["nll_old_sum"] for r in per_item) / tokens, "nll_full_new": sum(r["nll_new_sum"] for r in per_item) / tokens,
            "kl_token_mean": sum(r["kl_mean"] * r["targets"] for r in per_item) / tokens,
            "kl_item_quantiles": {q: quant(q) for q in (0.1, 0.5, 0.9, 1.0)},
            "items_kl_over": {th: sum(r["kl_mean"] > th for r in per_item) for th in (0.005, 0.01, 0.02, 0.05)},
            "items_first_turn_kl_over": {th: sum(r["kl_first_turn"] > th for r in per_item) for th in (0.01, 0.05, 0.1, 0.5)},
            "top1_agree_token_mean": sum(r["top1_agree"] * r["targets"] for r in per_item) / tokens,
            "top32_overlap_token_mean": sum(r["top32_overlap"] * r["targets"] for r in per_item) / tokens,
            "think_all": {"old": think_stats(thinks, "old"), "new": think_stats(thinks, "new")},
            "think_first_turn": {"old": think_stats([t for t in thinks if t["first_turn"]], "old"), "new": think_stats([t for t in thinks if t["first_turn"]], "new")},
            "per_item": [{k: v for k, v in r.items() if k != "think"} for r in per_item]}
        print(json.dumps({k: v for k, v in report["packets"][name].items() if k != "per_item"}, indent=1), flush=True)
    report["wall_seconds"] = round(time.monotonic() - started, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1))
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
