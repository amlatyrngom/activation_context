"""
Equivalence numbers from hf_reference.py (REF_DIR) and vllm_run.py (OUT_DIR/{stock,res}_equiv.pt):
  res  vs HF res   rows + deltas: vLLM residual model vs the HF hooks
  rows vs HF rows  rows only: upstream vLLM vs HF (the engine gap the deltas path inherits)
  zero vs rows     zero-delta pack on the residual model vs upstream rows only (expected bit for bit)
  delta effect     HF res vs HF rows (how far the deltas move the reader; the tolerance has to be well below it)
Log-prob differences are over vLLM's top-20 tokens at the first generated position (HF value from its full distribution).
  python compare.py REF_DIR OUT_DIR  ->  OUT_DIR/compare.json and a table on stdout
"""
import json
import sys
from pathlib import Path

import torch


def logprob_diffs(vllm_lp: dict, hf_lp: torch.Tensor):
    return [abs(value - float(hf_lp[token])) for token, value in vllm_lp.items()]


def first_divergence(a: list, b: list):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summary(diffs_per_item, matches, divergences, top1):
    flat = [d for diffs in diffs_per_item for d in diffs]
    return {"logprob_max_abs_diff": round(max(flat), 5), "logprob_mean_abs_diff": round(sum(flat) / len(flat), 5),
            "per_item_max": [round(max(d), 4) for d in diffs_per_item], "top1_agree": f"{sum(top1)}/{len(top1)}",
            "greedy_exact_match": f"{sum(matches)}/{len(matches)}", "first_divergence": divergences}


def main():
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = json.loads((ref / "manifest.json").read_text())
    items = [torch.load(ref / "items" / f"{entry['index']:02d}.pt", weights_only=False) for entry in manifest["items"]]
    stock, res = torch.load(out / "stock_equiv.pt"), torch.load(out / "res_equiv.pt")
    result = {"items": [{"id": e["id"], "k": e["k"], "prompt_tokens": e["prompt_tokens"], "rows": e["rows"]} for e in manifest["items"]],
              "reader_lora_B_max_abs": manifest.get("reader_lora_B_max_abs")}

    eos = set(manifest["eos_ids"])

    def strip(tokens):                                       # the end token is compared by its absence: either side may keep it
        tokens = list(tokens)
        while tokens and tokens[-1] in eos:
            tokens.pop()
        return tokens

    def versus(vllm_runs, hf_mode):
        diffs, matches, divergences, top1 = [], [], [], []
        for run, item in zip(vllm_runs, items):
            hf = item[hf_mode]
            diffs.append(logprob_diffs(run["logprobs"], hf["logprobs"]))
            matches.append(strip(run["tokens"]) == strip(hf["tokens"]))
            divergences.append(first_divergence(strip(run["tokens"]), strip(hf["tokens"])))
            top1.append(max(run["logprobs"], key=run["logprobs"].get) == int(hf["logprobs"].argmax()))
        return summary(diffs, matches, divergences, top1)

    result["res_vs_hf_res"] = versus(res["res"], "res")
    result["rows_vs_hf_rows"] = versus(stock["rows"], "rows")
    result["res_vs_hf_rows (wrong pairing, for scale)"] = versus(res["res"], "rows")
    def same(runs_a, runs_b):                                 # two vLLM runs of the same prompts: bit-for-bit or how far apart
        exact = [x["tokens"] == y["tokens"] and x["logprobs"] == y["logprobs"] for x, y in zip(runs_a, runs_b)]
        diffs = [max(abs(x["logprobs"][t] - y["logprobs"][t]) for t in x["logprobs"] if t in y["logprobs"]) for x, y in zip(runs_a, runs_b)]
        return {"bit_exact_items": f"{sum(exact)}/{len(exact)}", "max_abs_logprob_diff": round(max(diffs), 5),
                "greedy_exact_match": f"{sum(strip(x['tokens']) == strip(y['tokens']) for x, y in zip(runs_a, runs_b))}/{len(exact)}"}

    result["zero_vs_stock_rows"] = same(res["zero"], stock["rows"])
    if (out / "stock_eager_equiv.pt").exists() and (out / "res_eager_equiv.pt").exists():    # eager engines: no compiled-graph differences
        stock_eager, res_eager = torch.load(out / "stock_eager_equiv.pt"), torch.load(out / "res_eager_equiv.pt")
        result["eager: zero_vs_stock_rows"] = same(res_eager["zero"], stock_eager["rows"])
        result["eager: res_vs_hf_res"] = versus(res_eager["res"], "res")
        result["eager: rows_vs_hf_rows"] = versus(stock_eager["rows"], "rows")
        result["stock compiled vs stock eager (same model, compile only)"] = same(stock["rows"], stock_eager["rows"])
        result["res compiled vs res eager (same model, compile only)"] = same(res["res"], res_eager["res"])
    effect = []
    for item in items:
        hf_res, hf_rows = item["res"]["logprobs"], item["rows"]["logprobs"]
        top = torch.topk(hf_res, 20).indices
        effect.append(float((hf_res[top] - hf_rows[top]).abs().max()))
    result["delta_effect_hf"] = {"logprob_max_abs_shift_per_item": [round(x, 3) for x in effect],
                                 "greedy_same_with_and_without_deltas": f"{sum(i['res']['tokens'] == i['rows']['tokens'] for i in items)}/{len(items)}"}
    (out / "compare.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
