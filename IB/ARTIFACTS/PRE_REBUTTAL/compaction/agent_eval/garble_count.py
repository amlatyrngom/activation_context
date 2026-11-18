"""Tool-call format failures per cache (AGENT_EVAL_DEV75.md §5 definition): assistant turns without a parsed call, of which the
"\n[…]" garble ("[/shell]"-style line, no <tool_call>), runs with >= 1 garble, first continuation turn garbled.
    python3 -I garble_count.py CACHE ...     (reads S3 agent_eval_dev75/ROLLOUTS/<cache>/rollouts.jsonl)"""
import json, re, subprocess, sys
S3 = "s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/agent_eval_dev75/ROLLOUTS"
G = re.compile(r"(^|\n)\[/?[A-Za-z_]+\]")
print("| cache | runs | assistant turns | no parsed call | garble | runs with garble (solved) | first turn garbled |\n|---|---|---|---|---|---|---|")
for c in sys.argv[1:]:
    out = subprocess.run(["aws", "s3", "cp", f"{S3}/{c}/rollouts.jsonl", "-"], capture_output=True).stdout
    runs = turns = nocall = garb = rg = rgs = first = 0
    for line in out.splitlines():
        try: r = json.loads(line)
        except Exception: break
        runs += 1; hit = False; seen_first = False
        for s in r.get("trajectory", []):
            if s.get("role") != "assistant": continue
            turns += 1
            g = False
            if not s.get("tool_calls"):
                nocall += 1
                if G.search(s.get("content") or ""): garb += 1; g = True; hit = True
            if not seen_first:
                seen_first = True; first += g
        rg += hit; rgs += hit and bool(r.get("score") and r["score"] >= 1.0)
    print(f"| {c} | {runs} | {turns} | {nocall} | {garb} | {rg} ({rgs}) | {first} |")
