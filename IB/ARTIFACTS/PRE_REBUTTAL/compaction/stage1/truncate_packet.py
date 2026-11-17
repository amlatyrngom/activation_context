"""
Phase-A option of the compaction stage-1 retrain (2026-11-17): a copy of a stage-1 packet whose items keep only the first T
continuation target tokens. CPU only, run on this host after prepare_agent_continuation.py.

What changes per item (and nothing else): the continuation is cut right after its T-th target token. Whole assistant turns are
kept while their tokens fit; the turn that crosses T keeps its first tokens up to T; every message after it (tool results, later
turns) is dropped from both views (teacher in_context_messages and student ac_messages, which share the continuation). The
teacher view's prefix (info["in_context_history_ids"]), the student's AC part, r and the rows are untouched, so the KL is the
same loss on a shorter span: the first T tokens the run wrote after the cut. Items with at most T target tokens are unchanged.
info gains continuation_token_limit = T and target_tokens_full; target_tokens / assistant_turns are the kept ones
(reader_tokens stays the untruncated teacher length).

    python truncate_packet.py --packet PACKETS/<name> --tokens 4096 --out PACKETS/<name>_t4k

Outputs: OUT/items.pkl, OUT/prepared.json (items sha256 / bytes updated, the rest copied), OUT/recipe.json (copied, plus
"continuation_token_limit"), OUT/truncate.json (counts). prepared_examples.pkl is not copied (sec45_train reads items.pkl only).
"""
import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path


def truncate(item, limit: int):
    """The item with its continuation cut after `limit` target tokens (the same object when it already fits)."""
    total = sum(len(ids) for ids in item.assistant_token_ids)
    if total <= limit:
        return item, total, total
    cont = item.in_context_messages[item.in_context_start:]
    student_cont = item.ac_messages[item.ac_start:]
    if [m.get("role") for m in cont] != [m.get("role") for m in student_cont]:
        raise ValueError(f"{item.item_id}: teacher and student continuations differ")
    kept_ids, left, turn = [], limit, 0
    for index, message in enumerate(cont):
        if message.get("role") != "assistant":
            continue
        ids = item.assistant_token_ids[turn]
        turn += 1
        kept_ids.append(list(ids[:left]))
        left -= len(kept_ids[-1])
        if left == 0:
            last = index
            break
    out = copy.copy(item)
    out.in_context_messages = item.in_context_messages[:item.in_context_start + last + 1]
    out.ac_messages = item.ac_messages[:item.ac_start + last + 1]
    out.assistant_token_ids = kept_ids
    out.info = dict(item.info, continuation_token_limit=limit, target_tokens_full=total, target_tokens=limit, assistant_turns=len(kept_ids))
    assert sum(len(ids) for ids in kept_ids) == limit and out.in_context_messages[-1].get("role") == "assistant"
    return out, total, limit


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packet", type=Path, required=True); p.add_argument("--tokens", type=int, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (packet names are never reused)")
    with open(a.packet / "items.pkl", "rb") as f:
        data = pickle.load(f)
    if data.get("held"):
        raise SystemExit("the packet has held items (stage-1 packets have none): not handled")
    items, full, kept, cut = [], 0, 0, 0
    for item in data["train"]:
        new, total, k = truncate(item, a.tokens)
        items.append(new); full += total; kept += k; cut += new is not item
    a.out.mkdir(parents=True)
    blob = pickle.dumps({"train": items, "held": data.get("held", [])}, protocol=pickle.HIGHEST_PROTOCOL)
    (a.out / "items.pkl").write_bytes(blob)
    prepared = json.loads((a.packet / "prepared.json").read_text())
    prepared.update(items_sha256=hashlib.sha256(blob).hexdigest(), items_bytes=len(blob), prepared_examples_sha256=None,
                    truncated_from=str(a.packet), continuation_token_limit=a.tokens)
    (a.out / "prepared.json").write_text(json.dumps(prepared, indent=2))
    recipe = json.loads((a.packet / "recipe.json").read_text())
    recipe["continuation_token_limit"] = a.tokens
    (a.out / "recipe.json").write_text(json.dumps(recipe, indent=2))
    summary = {"source": str(a.packet), "source_items_sha256": hashlib.sha256((a.packet / "items.pkl").read_bytes()).hexdigest(),
               "continuation_token_limit": a.tokens, "items": len(items), "items_cut": cut, "target_tokens_full": full, "target_tokens_kept": kept,
               "kept_fraction": round(kept / max(1, full), 4), "items_sha256": prepared["items_sha256"]}
    (a.out / "truncate.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
