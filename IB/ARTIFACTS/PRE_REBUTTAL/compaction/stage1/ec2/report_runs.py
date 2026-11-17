"""
Retrain report (2026-11-17), this host: the dev-75 tables of stage-1 runs from their S3 folders, against the corrected reference
(S3 compaction/stage1/proxy_ref_tv/proxy_ref_tv.json: NLL_full = the untuned 9B with the full prefix in the recorded-token view,
NLL_none = no context). gap closed = (NLL_none - NLL) / (NLL_none - NLL_full), per size (r10 = 2,000 rows, r5 = 4,000 rows).
  panel     sync/<RUN>.held.jsonl: the in-run 150-item dev-75 panel (every report interval and the end; smoke runs: 2 items only)
  proxy     proxy/final.json: cmp_proxy_eval.py on ckpt/final (the same panel, deterministic kernels)
  probe     probe/probe_final.json (+ .positions.jsonl): probe_think.py at gold "</think>" / "<tool_call>" positions, and by the
            position of the "</think>" in its continuation (1st, 2nd-5th, 6th-20th, 21st+)

    python report_runs.py RUN [RUN ...]
"""
import collections
import json
import subprocess
import sys

S3 = "s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1"


def s3_text(key):
    r = subprocess.run(["aws", "s3", "cp", f"{S3}/{key}", "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main():
    ref = json.loads(s3_text("proxy_ref_tv/proxy_ref_tv.json"))
    full, none = ref["nll_full"], ref["nll_none"]
    gap = lambda nll: (none - nll) / (none - full)
    print(f"reference: NLL_full {full:.6f}, NLL_none {none:.6f} (proxy_ref_tv)\n")
    for run in sys.argv[1:]:
        print(f"## {run}\n\n| update | items | KL r10 | KL r5 | NLL r10 | NLL r5 | gap closed r10 | gap closed r5 |\n|---|---|---|---|---|---|---|---|")
        for line in (s3_text(f"runs/{run}/sync/{run}.held.jsonl") or "").splitlines():
            d = json.loads(line); g = d["groups"]
            r10, r5 = g.get("dev75/r10"), g.get("dev75/r5")
            if r10 and r5:
                print(f"| {d['update']} | {r10['items']} | {r10['kl_token']:.4f} | {r5['kl_token']:.4f} | {r10['nll_token']:.4f} | {r5['nll_token']:.4f} | {gap(r10['nll_token']):.3f} | {gap(r5['nll_token']):.3f} |")
        proxy = s3_text(f"runs/{run}/proxy/final.json")
        if proxy:
            d = json.loads(proxy); g = d["groups"]
            print(f"| final proxy | {g['dev75/r10'].get('items', '-')} | {g['dev75/r10']['kl_token']:.4f} | {g['dev75/r5']['kl_token']:.4f} | {g['dev75/r10']['nll_token']:.4f} | {g['dev75/r5']['nll_token']:.4f} "
                  f"| {gap(g['dev75/r10']['nll_token']):.3f} | {gap(g['dev75/r5']['nll_token']):.3f} |")
        stats = s3_text(f"runs/{run}/sync/{run}.stats.json")
        if stats:
            t = json.loads(stats).get("timing", {})
            print(f"\ntiming: {json.dumps({k: t.get(k) for k in ('updates', 'effective_seconds_per_update', 'seconds_per_update_median', 'peak_gb', 'wall_seconds')})}")
        probe = s3_text(f"runs/{run}/probe/probe_final.json")
        if probe:
            d = json.loads(probe)
            print("\nprobe (gold </think> / <tool_call> positions):\n\n| view | think pos | mean lp </think> | mean lp [/ | % [/ > </think> | % [/ top-1 | call pos | mean lp <tool_call> | % [/ > <tool_call> |\n|---|---|---|---|---|---|---|---|---|---|")
            for group, kinds in d["groups"].items():
                t, c = kinds.get("think", {}), kinds.get("call", {})
                f = lambda v: "-" if v is None else f"{v:.3f}" if abs(v) < 1 else f"{v:.2f}"
                print(f"| {group} | {t.get('positions')} | {f(t.get('mean_lp_think'))} | {f(t.get('mean_lp_bracket_slash'))} | {f(t.get('pct_bracket_slash_over_gold'))} "
                      f"| {f(t.get('pct_top1_bracket_slash'))} | {c.get('positions')} | {f(c.get('mean_lp_tool_call'))} | {f(c.get('pct_bracket_slash_over_gold'))} |")
            rows = [json.loads(l) for l in (s3_text(f"runs/{run}/probe/probe_final.positions.jsonl") or "").splitlines()]
            if rows:
                order = collections.defaultdict(list)
                for r in rows:
                    if r["kind"] == "think":
                        order[(r["group"], r["item"])].append(r)
                bins = collections.defaultdict(list)
                for (group, _), rs in order.items():
                    for k, r in enumerate(rs):                     # capture order = order in the continuation
                        bins[(group, "1st" if k == 0 else "2nd-5th" if k < 5 else "6th-20th" if k < 20 else "21st+")].append(r)
                print("\nby position of the </think> in the continuation (mean lp [/, % [/ > </think>, n):\n\n| view | 1st | 2nd-5th | 6th-20th | 21st+ |\n|---|---|---|---|---|")
                for group in sorted({g for g, _ in bins}):
                    cells = []
                    for b in ("1st", "2nd-5th", "6th-20th", "21st+"):
                        rs = bins.get((group, b), [])
                        cells.append(f"{sum(r['[/'] for r in rs) / len(rs):.2f}, {100 * sum(r['[/'] > r['</think>'] for r in rs) / len(rs):.1f} %, {len(rs)}" if rs else "-")
                    print(f"| {group} | " + " | ".join(cells) + " |")
        print()


if __name__ == "__main__":
    main()
