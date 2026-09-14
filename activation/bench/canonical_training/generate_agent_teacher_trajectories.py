"""
Canonical entry point that generates and stores the teacher trajectories: the study's campaign over the benchmarks,
cached under ROLLOUTS/<caching_id> in the synced area (IB/TMP/SYNC locally, ~/activation_artifacts/SYNC on nodes) and,
later, in S3 (`activation.common.s3_client`, bucket set there; not used yet). A run first probes on the defaults
(`--probe-tasks-per-gpu` x GPUs), tunes the rollout configuration under `--autotune-id`, then continues; a run under an
existing autotune id applies its saved configuration from the start. `--max-wall-hours` stops starting new tasks past
the deadline and clips the running ones, so a 12 h run ends at 12 h with everything finished so far in the cache.

    uv run python -m activation.bench.canonical_training.generate_agent_teacher_trajectories --total 10000 --caching-id teacher_v1 \\
        --report-folder AGENT_AC_PRETRAINING/teacher_v1 --autotune-id campaign_27b_medium --max-wall-hours 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from activation.agent_training.agent_training_teacher_study import (
    BENCHMARKS, DEFAULT_DATASET_WEIGHTS, DEFAULT_TEACHER_WEIGHTS, ORACLE_MODEL_ID, REASONING_EFFORTS, AgentTeacherKind, AgentTrainingStudy, make_harness,
)
from activation.agent.rollout_caching import CACHE_FILENAME, RedoPolicy
from activation.common.data_syncing import resolve_path
from activation.dataset import DatasetTaskKind

PROBE_TASKS_PER_GPU = 200
FULL_RUN_COUNT = 10_000


def gpu_count() -> int:
    try:
        import torch
        return max(1, torch.cuda.device_count())
    except Exception:
        return 1


def check_draw(caching_id: str, draw: dict, allow_change: bool = False) -> Path:
    """
    The draw (seed, total, weights, reasoning, model) recorded next to the cache the first time a caching id is used; a
    later run under the same id must reproduce it, since one random generator assigns tasks and teacher kinds and any
    change turns every cache key into a miss. Limits (turns, duration, concurrency) are free to change.
    """
    path = resolve_path(f"ROLLOUTS/{caching_id}", create=True) / "draw.json"
    if not path.exists():
        path.write_text(json.dumps(draw, indent=1, default=str))
        return path
    recorded = json.loads(path.read_text())
    current = json.loads(json.dumps(draw, default=str))
    changed = {key for key in set(recorded) | set(current) if recorded.get(key) != current.get(key)}
    if changed and not allow_change:
        raise SystemExit(f"caching id {caching_id!r} was drawn with {json.dumps({k: recorded.get(k) for k in sorted(changed)}, default=str)}; "
                         f"this run gives {json.dumps({k: current.get(k) for k in sorted(changed)}, default=str)}. "
                         "Use a new --caching-id, the recorded values, or --allow-draw-change.")
    if changed:
        print(f"draw changed for caching id {caching_id!r} ({sorted(changed)}); cached rows keyed under the old draw are misses.", flush=True)
    return path


def cached_draw_head(cache_path: Path) -> list[dict]:
    """
    The draw entries of an existing cache written before draw records existed (rows in file order, one per key: the
    last row wins as in the cache), so the recorded draw starts with every task already rolled out under this id.
    """
    if not cache_path.exists():
        return []
    entries: dict[tuple[str, str, str], dict] = {}
    with open(cache_path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            task, meta = (row.get("agent_config") or {}).get("dataset_task") or {}, (row.get("agent_config") or {}).get("metadata") or {}
            if not task or "template" not in meta:
                continue
            entry = {"benchmark": meta["benchmark"], "dataset_id": task["dataset_id"], "task_id": task["task_id"],
                     "teacher_kind": meta["teacher_kind"], "template": meta["template"]}
            entries[(entry["dataset_id"], entry["task_id"], entry["template"])] = entry
    print(f"draw head: {len(entries)} cached rollouts under {cache_path} come first in the recorded draw.", flush=True)
    return list(entries.values())


def parse_weights(items: list[str] | None, default: dict) -> dict:
    """'name=weight' pairs; missing names keep 0 (only the given ones run) unless nothing is given (the defaults)."""
    if not items:
        return dict(default)
    out = {}
    for item in items:
        name, _, value = item.partition("=")
        out[name.strip()] = float(value) if value else 1.0
    return out


def parse_teacher_weights(items: list[str] | None) -> dict:
    """'task_kind:teacher_kind=weight' (task_kind 'all' applies to every kind); defaults otherwise."""
    if not items:
        return dict(DEFAULT_TEACHER_WEIGHTS)
    out = dict(DEFAULT_TEACHER_WEIGHTS)
    for item in items:
        key, _, value = item.partition("=")
        kind_name, _, teacher_name = key.partition(":")
        kinds = list(DatasetTaskKind) if kind_name.strip() == "all" else [DatasetTaskKind(kind_name.strip())]
        for kind in kinds:
            out[(kind, AgentTeacherKind(teacher_name.strip()))] = float(value) if value else 1.0
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--total", type=int, default=FULL_RUN_COUNT, help="tasks to roll out (each benchmark is drawn by weight, no repeats while it has unused tasks)")
    parser.add_argument("--per-benchmark", type=int, default=None, help="tasks loaded per benchmark; default: enough for --total at the weights, with a margin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--caching-id", default="teacher_v1")
    parser.add_argument("--report-folder", default="AGENT_AC_PRETRAINING/teacher_v1")
    parser.add_argument("--dataset-weights", nargs="*", default=None, help=f"name=weight; default {DEFAULT_DATASET_WEIGHTS}")
    parser.add_argument("--teacher-weights", nargs="*", default=None, help="task_kind:teacher_kind=weight (task_kind 'all'); default base 0.2, sequential 0.4, parallel 0.4")
    parser.add_argument("--model-id", default=ORACLE_MODEL_ID)
    parser.add_argument("--reasoning", choices=(*REASONING_EFFORTS, "off"), default="medium")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--max-duration", type=float, default=1800.0)
    parser.add_argument("--agents-per-gpu", type=int, default=None, help="pool bound per GPU (default 40); with a saved --autotune-id it overrides the saved bound")
    parser.add_argument("--concurrent", type=int, default=None, help="explicit total pool bound; default agents-per-gpu x GPUs")
    parser.add_argument("--autotune-id", default=None)
    parser.add_argument("--probe-tasks-per-gpu", type=int, default=PROBE_TASKS_PER_GPU)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--no-scoring", action="store_true")
    parser.add_argument("--redo-finish-reasons", nargs="*", default=None,
                        help="roll out again the cached rows with these finish reasons (e.g. max_duration max_turns); the new rows supersede them")
    parser.add_argument("--redo-teacher-kinds", nargs="*", default=None, help="restrict --redo-finish-reasons to these teacher kinds")
    parser.add_argument("--redo-only", action="store_true", help="run nothing but the rows to redo (tasks without a cached row are skipped)")
    parser.add_argument("--allow-draw-change", action="store_true",
                        help="proceed although seed, total, weights, reasoning or model differ from the draw recorded with this caching id")
    args = parser.parse_args(argv)

    dataset_weights = parse_weights(args.dataset_weights, DEFAULT_DATASET_WEIGHTS)
    scale = sum(dataset_weights.values()) or 1.0
    per_benchmark = args.per_benchmark or {name: int(args.total * w / scale * 1.2) + 5 for name, w in dataset_weights.items()}
    teacher_weights = parse_teacher_weights(args.teacher_weights)
    redo = None
    if args.redo_finish_reasons or args.redo_only:
        if not args.redo_finish_reasons:
            raise SystemExit("--redo-only needs --redo-finish-reasons")
        redo = RedoPolicy(finish_reasons=frozenset(args.redo_finish_reasons), teacher_kinds=frozenset(args.redo_teacher_kinds or ()), only=args.redo_only)
    check_draw(args.caching_id, {"seed": args.seed, "total": args.total, "dataset_weights": dataset_weights,
                                 "teacher_weights": {f"{kind}:{teacher}": weight for (kind, teacher), weight in sorted(teacher_weights.items(), key=str)},
                                 "reasoning": args.reasoning, "model_id": args.model_id}, allow_change=args.allow_draw_change)
    harness = make_harness(args.model_id, agent_max_concurrent=args.concurrent, agents_per_gpu=args.agents_per_gpu or 40)
    study = AgentTrainingStudy(harness, dataset_weights, teacher_weights, reasoning=None if args.reasoning == "off" else args.reasoning,
                               max_turns=args.max_turns, max_duration=args.max_duration)
    started = time.time()
    try:
        study.setup_reusable_artifacts(per_benchmark, seed=args.seed)
        record = resolve_path(f"ROLLOUTS/{args.caching_id}", create=True) / "draw_tasks.jsonl"
        head = [] if record.exists() else cached_draw_head(resolve_path(f"ROLLOUTS/{args.caching_id}", create=False) / CACHE_FILENAME)
        configs = study.generate_rollout_campaign_tasks(args.total, seed=args.seed, record=record, head=head)
        folder = resolve_path(args.report_folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "campaign.json").write_text(json.dumps({
            "args": vars(args), "gpu_count": gpu_count(), "tasks": len(configs),
            "by_benchmark": {name: sum(c.metadata["benchmark"] == name for c in configs) for name in BENCHMARKS},
            "by_teacher_kind": {str(k): sum(c.metadata["teacher_kind"] == str(k) for c in configs) for k in AgentTeacherKind},
        }, indent=1, default=str))
        results = study.execute_rollout_campaign_tasks(
            configs, seed=args.seed, caching_id=args.caching_id, report_folder=args.report_folder, title=f"Teacher campaign {args.caching_id}",
            autotune_id=args.autotune_id, probe_tasks_per_gpu=args.probe_tasks_per_gpu,
            max_wall_seconds=None if args.max_wall_hours is None else args.max_wall_hours * 3600.0, perform_scoring=not args.no_scoring,
            redo=redo, agents_per_gpu=args.agents_per_gpu,
        )
        tuning = harness.rollout_manager.tuning_report
        (folder / "campaign.json").write_text(json.dumps({
            **json.loads((folder / "campaign.json").read_text()), "finished": len(results), "wall_seconds": round(time.time() - started, 1), "tuning": tuning,
        }, indent=1, default=str))
    finally:
        from activation.harness import FREE_DEVICE
        for loaded in harness.loaded_models.values():
            loaded.engine_to_device(FREE_DEVICE)
    print(f"{len(results)} rollouts in {time.time() - started:.0f}s; cache ROLLOUTS/{args.caching_id}; report {folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
