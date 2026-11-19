"""
Math recipe summary tables (2026-11-19, CPU, existing score files only): prints the dev / dev_hard markdown tables of RECIPE_SUMMARY.md
from score_math.py outputs, so every number in the summary is read from a file.

  dev       scores/summary_dev_{std,lmx}_vs_{results,ac_r10_s1,ac_r5_s1}.json
  dev_hard  scores/devhard[_lmx]_{all,d8.5,ge9}_vs_{results,ac_r10_s1,ac_r5_s1}.json  (devhard_report_v6 runs)
std = all problems; lmx = sensitivity, label-mismatch problems excluded (scores/labelmismatch.json).

    .venv/bin/python -I recipe_summary_tables.py
"""
import json
from pathlib import Path

S = Path(__file__).resolve().parents[5] / "IB/TMP/PRE_REBUTTAL/math_recipe/scores"
ARMS = [("results", "9B results-only", None), ("full", "9B full runs", None), ("ac_r10_s1", "AC r10 stage 1", None),
        ("ac_r10_s2rec", "AC r10 stage 2 (recipe)", "ac_r10_s1"), ("ac_r5_s1", "AC r5 stage 1", None), ("ac_r5_s2rec", "AC r5 stage 2 (recipe)", "ac_r5_s1")]
DEVHARD_NAME = {"ac_r10_s2rec": "ac_r10_s2hr", "ac_r5_s2rec": "ac_r5_s2hr"}


def diff(x) -> str:
    return f"{100 * x['diff']:+.2f} ({100 * x['ci95'][0]:+.2f}, {100 * x['ci95'][1]:+.2f}; p {x['p']:.2f})"


def table(files: dict, rename: dict, title: str) -> None:
    base = files["results"]
    sr = base["solver_references"]
    print(f"\n**{title}** — {base['tasks']} problems × 3 seeds; single solver {100 * sr['single']['all']:.2f}, majority of 3 "
          f"{100 * sr['majority']['all']:.2f}, any of 3 {100 * sr['any']['all']:.2f}\n")
    print("| arm | accuracy % | gen tokens median | − results-only (95 % CI; p) | stage 2 − own stage 1 (95 % CI; p) |")
    print("|---|---|---|---|---|")
    for key, label, s1 in ARMS:
        name = rename.get(key, key)
        v = base["arms"][name]
        vs_res = files["results"]["arms"][name].get("vs_results")
        vs_s1 = files[s1]["arms"][name].get(f"vs_{s1}") if s1 else None
        print(f"| {label} | {100 * v['lenient']:.2f} | {v['generated_tokens']['median']:,} | {diff(vs_res) if vs_res else '—'} | {diff(vs_s1) if vs_s1 else ''} |")


def main() -> int:
    for tag, label in (("std", "all problems"), ("lmx", "sensitivity: label-mismatch problems excluded")):
        files = {ref: json.loads((S / f"summary_dev_{tag}_vs_{ref}.json").read_text()) for ref in ("results", "ac_r10_s1", "ac_r5_s1")}
        table(files, {}, f"dev, {label}")
    for tag, label in (("", "all problems"), ("_lmx", "sensitivity: label-mismatch problems excluded")):
        for band, band_label in (("all", "all bands"), ("d8.5", "d8.5"), ("ge9", "≥ 9")):
            files = {ref: json.loads((S / f"devhard{tag}_{band}_vs_{ref}.json").read_text()) for ref in ("results", "ac_r10_s1", "ac_r5_s1")}
            table(files, DEVHARD_NAME, f"dev_hard {band_label}, {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
