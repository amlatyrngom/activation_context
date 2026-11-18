"""Paper-style difficulty breakdown of dev agent evals (paper §3.3, fig_compaction: P50 / P75 = the hardest 50 % / 25 % of
tasks, difficulty = continuation tokens the full-context reference needs; here the mean over the given full seeds, budget rule).
    python3 -I by_difficulty.py DIR FULLTAG ARM=TAG[,TAG...] ...   (DIR holds <cache>.jsonl; TAG like cdv1base_s{seed}_full)
Prints per arm: solved % and mean tokens (budget rule) over all / P50 / P75 tasks, on the seeds the arm has."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from report import load
B = 50_000
d = Path(sys.argv[1]); full_tags = sys.argv[2].split(",")
def runs(tag):
    p = d / f"{tag}.jsonl"
    return load(str(p), B) if p.exists() and p.stat().st_size else {}
full = [runs(t) for t in full_tags]
tasks = sorted(set.intersection(*[set(r) for r in full]))
diff = {t: sum(r[t]["tokens"] for r in full) / len(full) for t in tasks}
order = sorted(tasks, key=lambda t: diff[t])
sets = {"all": set(tasks), "P50": set(order[len(order) // 2:]), "P75": set(order[3 * len(order) // 4:])}
print(f"tasks {len(tasks)}  P50 {len(sets['P50'])}  P75 {len(sets['P75'])}  (difficulty from {','.join(full_tags)})")
print("| arm | seeds | " + " | ".join(f"{k} solved % | {k} tokens" for k in sets) + " |")
for spec in sys.argv[3:]:
    name, tags = spec.split("=")
    rs = [r for r in (runs(t) for t in tags.split(",")) if len(r) >= len(tasks)]
    cells = []
    for k, s in sets.items():
        rows = [r[t] for r in rs for t in s if t in r]
        cells += [f"{100 * sum(x['solved'] for x in rows) / len(rows):.1f}", f"{sum(x['tokens'] for x in rows) / len(rows):,.0f}"] if rows else ["–", "–"]
    print(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
