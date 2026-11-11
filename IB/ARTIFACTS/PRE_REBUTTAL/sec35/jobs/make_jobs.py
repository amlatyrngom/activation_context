"""
Writes the twelve section 4.5 job scripts from job_template.sh (one self-contained bash file per run):
  P0_{in,res,kv}.sh   throughput probe on an H200 (5 short trainings on the longest items + a decode probe) -> probe_summary_<mode>.json
  R_{in,res,kv}.sh    plain FineWeb-Edu reconstruction from a fresh pair (no-context penalty 1.0, as S/T)
  S_{in,res,kv}.sh    HotpotQA task training from a fresh pair
  T_{in,res,kv}.sh    the same task training from R_<mode>'s final pair (same steps, data, rates, order as S_<mode>)
Slurm limits: P0 3 h, R/S/T 12 h (v4: resumable; ORCD passes --time=6:00:00 and the run continues in the next job).

Values are keyed by the GPU name the P0 probes ran on (nvidia-smi name); each R/S/T script refuses any other GPU.
  python make_jobs.py                                   placeholders only (R/S/T refuse to start)
  python make_jobs.py --probes runs/P0_*/probe_summary_*.json
        merges the three probe summaries into jobs/values_<gpu>.json (never replaced) and writes jobs/generated_<gpu>/
  python make_jobs.py --values VALUES.json --gpu "NVIDIA H200"
        VALUES.json = {"NVIDIA H200": {"in": {"R": {"steps", "seconds", "ckpt_every", "micro_batch", "pair_gb"?}, "S_T": {...}},
                                       "res": {...}, "kv": {...}}}   (S_T steps: a multiple of 4)
Existing folders and files are never replaced: --out names a new folder.
v6 (Oct 27): --encoder Qwen/Qwen3-0.6B writes runs with a smaller encoder (SEC45_ENCODER; modes in/res only, --modes in): the
template's @ENCODER@ (empty = the encoder is a copy of the reader, as before). Needs bundle v6 (sec45_common.py accepts SEC45_ENCODER).
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S3 = "s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35"
WHAT = {"P0": "P0 throughput probe", "R": "plain passage reconstruction on FineWeb-Edu, fresh pair",
        "S": "HotpotQA task training from a fresh pair (scratch)", "T": "HotpotQA task training from the final R pair"}
NAMES = {"in": "Input Embeddings", "res": "Hidden Residuals, every 4th layer", "kv": "All-Layer KVs"}
MODES = ("in", "res", "kv")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def from_probes(paths: list[Path]) -> tuple[str, dict]:
    values, gpus = {}, set()
    for path in paths:
        summary = json.loads(path.read_text())
        if summary.get("problems"):
            raise SystemExit(f"{path}: problems {summary['problems']} (fix or write the values by hand)")
        if set(summary["proposal"]) != {"R", "S_T"}:
            raise SystemExit(f"{path}: the proposal lacks R or S_T")
        gpus.add(summary["gpu"])
        values[summary["mode"]] = summary["proposal"]
    if len(gpus) != 1 or set(values) != set(MODES):
        raise SystemExit(f"need one probe summary per mode {MODES} from ONE GPU type; got modes {sorted(values)} on {sorted(map(str, gpus))}")
    gpu = gpus.pop()
    return gpu, {gpu: values}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--values", type=Path); p.add_argument("--gpu"); p.add_argument("--probes", type=Path, nargs="+")
    p.add_argument("--out", type=Path); p.add_argument("--bundle", default="sec35_code_v3.tar.gz")
    p.add_argument("--values-out", type=Path, help="with --probes: where the merged values go (default jobs/values_<gpu>.json)")
    p.add_argument("--tag", default="", help="run-name suffix (v4: '_v4' -> R_in_v4, S_in_v4, T_in_v4 starting from R_in_v4); keeps new runs apart from old folders")
    p.add_argument("--reader-lr", default="2e-5", help="reader LoRA learning rate; 0 = frozen reader (exactly 0, never a tiny rate)")
    p.add_argument("--encoder-lr", default="2e-5", help="encoder (AC modules + encoder LoRA) learning rate")
    p.add_argument("--penalty", default="1.0", help="no-context penalty weight (only while the reader trains)")
    p.add_argument("--seed", default="0", help="data order seed")
    p.add_argument("--reader", default="Qwen/Qwen3-4B-Instruct-2507", help="reader Hub id (retrieval: Qwen/Qwen3.5-9B)")
    p.add_argument("--res-layers", default="", help="res mode reader layers (SEC45_RES_LAYERS; retrieval 9B: all)")
    p.add_argument("--k", default="", help="retrieval rows: passages per item, one k or a mix (SEC45_K, e.g. 20,100,200)")
    p.add_argument("--eval-k", default="", help="retrieval rows: k of the accuracy evals during training")
    p.add_argument("--ratio", default="", help="compression ratio (SEC45_RATIO); empty = 0.1")
    p.add_argument("--linear-skip", action="store_true", help="fresh smaller-encoder pair: least-squares row head (SEC45_LINEAR_SKIP=1)")
    p.add_argument("--task-data", default="hotpot_tasks_v2", help="task data folder under W/data")
    p.add_argument("--p0-gpu", default="H200", help="P0 probe: a text the GPU name must contain")
    p.add_argument("--eval-split", default="dev1000", help="accuracy evals during training: dev1000 (choices) or test1000 (reported)")
    p.add_argument("--steps", type=int, help="override S/T steps (items = steps x 8; the train pool holds 90,447 = 11,305 updates)")
    p.add_argument("--ckpt-every", type=int, help="override S/T ckpt_every (multiple of 8; also the eval points)")
    p.add_argument("--runs", default="P0,R,S,T", help="which run kinds to write (comma list)")
    p.add_argument("--also-gpu", action="append", default=[], help="another exact nvidia-smi name the R/S/T scripts accept (only when the "
                   "P0 timings transfer to it, e.g. the same H200 model reported under another name on the other cluster)")
    p.add_argument("--encoder", default="", help="v6: a smaller encoder Hub id (e.g. Qwen/Qwen3-0.6B); empty = a copy of the reader")
    p.add_argument("--modes", default=",".join(MODES), help="which write modes to write (comma list)")
    a = p.parse_args()
    modes_wanted = a.modes.split(",")
    if set(modes_wanted) - set(MODES):
        raise SystemExit(f"--modes {a.modes}: one of {MODES}")
    if a.encoder and "kv" in modes_wanted:
        raise SystemExit("--encoder: modes in/res and runs R,S,T only (All-Layer KVs need the reader's own base)")
    if a.encoder and a.bundle in ("sec45_code_v3.tar.gz", "sec45_code_v4.tar.gz", "sec45_code_v5.tar.gz"):
        raise SystemExit(f"--encoder needs bundle v6 or later (got {a.bundle})")
    if a.probes:
        gpu, values = from_probes(a.probes)
        written = a.values_out or HERE / f"values_{slug(gpu)}.json"
        if written.exists():
            if json.loads(written.read_text()) != values:
                raise SystemExit(f"{written} exists with other values (never replaced): pass it with --values, or move it")
        else:
            written.write_text(json.dumps(values, indent=1))
            print(f"wrote {written}")
    elif a.values:
        values = json.loads(a.values.read_text())
        gpu = a.gpu or (next(iter(values)) if len(values) == 1 else None)
        if gpu not in values:
            raise SystemExit(f"--gpu must name one of {sorted(values)}")
    else:
        gpu, values = None, {}
    out = a.out or HERE / ("generated" if gpu is None else f"generated_{slug(gpu)}")
    if out.exists():
        raise SystemExit(f"{out} exists (never replaced): name another --out")
    modes = values.get(gpu, {})
    template = (HERE / "job_template.sh").read_text()
    s3x = (HERE.parent / "kit/s3x.py").read_text()
    out.mkdir(parents=True)
    for mode in [m for m in MODES if m in modes_wanted]:
        for run in [r for r in ("P0", "R", "S", "T") if r in a.runs.split(",")]:
            entry = dict(modes.get(mode, {}).get("R" if run == "R" else "S_T", {}))
            if run in ("S", "T"):
                entry |= ({"steps": a.steps} if a.steps else {}) | ({"ckpt_every": a.ckpt_every} if a.ckpt_every else {})
            if run == "P0":
                fill = {"STEPS": "23", "SECONDS": "1500", "CKPT_EVERY": "0", "MICRO": "1", "KIND": "probe", "HOURS": "3", "INIT": "fresh",
                        "GPU": f"contains:{a.p0_gpu}", "PAIR_GB": "3", "OPT_GB": "0"}
            else:
                key = "R" if run == "R" else "ST"
                fill = {"STEPS": str(entry.get("steps", f"__P0_{key}_STEPS__")), "SECONDS": str(entry.get("seconds", f"__P0_{key}_SECONDS__")),
                        "CKPT_EVERY": str(entry.get("ckpt_every", f"__P0_{key}_CKPT_EVERY__")), "MICRO": str(entry.get("micro_batch", f"__P0_{key}_MICRO__")),
                        "KIND": "recon" if run == "R" else "task", "HOURS": "96", "INIT": f"R_{mode}{a.tag}" if run == "T" else "fresh",
                        "GPU": "exact:" + "|".join([gpu, *a.also_gpu]) if gpu else "__P0_GPU__", "PAIR_GB": str(entry.get("pair_gb") or 3),
                        "OPT_GB": str(entry.get("opt_gb", f"__P0_{key}_OPT_GB__"))}
                if "ckpt_every" in entry and entry["ckpt_every"] % 8:
                    raise SystemExit(f"{mode} {run} ckpt_every {entry['ckpt_every']} is not a multiple of 8 (penalty-reference blocks of 64 examples = 8 updates must align with a resume point)")
                if "steps" in entry and entry["steps"] % entry.get("ckpt_every", 1):
                    raise SystemExit(f"{mode} {run} steps {entry['steps']} is not a multiple of ckpt_every {entry.get('ckpt_every')} (evals / slots on exact updates)")
            if float(a.reader_lr) != 0 and float(a.reader_lr) < 1e-7:
                raise SystemExit(f"--reader-lr {a.reader_lr}: a tiny rate is not a frozen reader; pass 0")
            fill |= {"RUN": f"{run}_{mode}{a.tag}", "MODE": mode, "READER_LR": a.reader_lr, "WHAT": f"{WHAT[run]}; write mode {NAMES[mode]}" + (f"; encoder {a.encoder}" if a.encoder else ""), "S3": S3, "BUNDLE": a.bundle, "ENC_LR": a.encoder_lr, "PENALTY": a.penalty, "SEED": a.seed, "EVAL_SPLIT": a.eval_split, "READER": a.reader, "RES_LAYERS": a.res_layers,
                     "K": a.k, "EVAL_K": a.eval_k, "LINEAR_SKIP": "1" if a.linear_skip else "", "RATIO": a.ratio, "TASK_DATA": a.task_data,
                     "ENCODER": a.encoder}
            text = template
            for key, value in fill.items():
                text = text.replace(f"@{key}@", value)
            text = text.replace("@S3X@", s3x.rstrip("\n"))
            left = re.findall(r"@[A-Z0-9_]+@", text)
            assert not left, f"unfilled template fields {left}"
            path = out / f"{run}_{mode}{a.tag}.sh"
            path.write_text(text)
            path.chmod(0o755)
    print(f"wrote {len(list(out.glob('*.sh')))} job scripts to {out}" + (f" for GPU '{gpu}'" if gpu else " (R/S/T with __P0_*__ placeholders)"))


if __name__ == "__main__":
    main()
