"""
One compact progress line per stage-1 run, for phone reading (2026-11-17), from the run's S3 folder:
  RUN: u<current>/<total>, <s>/upd, ETA <UTC>, last panel u<N>: KL r10/r5, gap closed r10/r5 (corrected reference proxy_ref_tv)
current update and s/update from logs/train_*.log (mean of the last 40 step lines, which exclude the panels; the ETA adds the panels'
measured share: wall seconds / updates between the latest two consecutive panels of one attempt, incl. checkpoint saves).

    python status_lines.py RUN [RUN ...]
"""
import datetime
import json
import re
import subprocess
import sys

S3 = "s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1"


def s3_text(key):
    r = subprocess.run(["aws", "s3", "cp", f"{S3}/{key}", "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def line(run):
    ref = json.loads(s3_text("proxy_ref_tv/proxy_ref_tv.json"))
    full, none = ref["nll_full"], ref["nll_none"]
    gap = lambda nll: (none - nll) / (none - full)
    logs = subprocess.run(["aws", "s3", "ls", f"{S3}/runs/{run}/logs/"], capture_output=True, text=True).stdout
    names = [l.split()[-1] for l in logs.splitlines() if re.search(r"train_.*\.log$", l)]
    steps = []
    for name in names:
        steps += [(int(m.group(1)), int(m.group(2)), float(m.group(3))) for m in re.finditer(r"step (\d+)/(\d+) global \d+: \S+ loss [\d.]+, ([\d.]+)s", s3_text(f"runs/{run}/logs/{name}"))]
    panels = [json.loads(l) for l in s3_text(f"runs/{run}/sync/{run}.held.jsonl").splitlines() if l.strip()]
    out = run + ":"
    if steps:
        steps.sort()
        cur, total = steps[-1][0], steps[-1][1]
        recent = [s for _, _, s in steps[-40:]]
        per = sum(recent) / len(recent)
        share = 1.0
        for a, b in reversed(list(zip(panels, panels[1:]))):      # the latest two consecutive panels of one attempt (wall_seconds restarts on a resume)
            if b["wall_seconds"] > a["wall_seconds"] and b["update"] > a["update"]:
                share = max(1.0, (b["wall_seconds"] - a["wall_seconds"]) / (b["update"] - a["update"]) / per); break
        eta = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=(total - cur) * per * share)
        out += f" u{cur}/{total}, {per:.0f} s/upd ({per * share:.0f} with panels), ETA {eta:%m-%d %H:%M}Z"
    else:
        out += " no step logged yet"
    if panels:
        g = panels[-1]["groups"]; r10, r5 = g["dev75/r10"], g["dev75/r5"]
        out += f"; panel u{panels[-1]['update']}: KL {r10['kl_token']:.4f}/{r5['kl_token']:.4f}, gap {gap(r10['nll_token']):.3f}/{gap(r5['nll_token']):.3f}"
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(line(run))
