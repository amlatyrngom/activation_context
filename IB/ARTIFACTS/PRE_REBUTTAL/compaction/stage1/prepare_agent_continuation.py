"""
Stage 2 of the continuation training data: material.jsonl (export_continuation_material.py) -> trainer packet(s) that
`run_channel_mixture.py train --output PACKET` loads unchanged. CPU only (CUDA_VISIBLE_DEVICES= ).

What one training example teaches. The encoder compresses a task's prefix into AC rows; the reader sees only the task head
and those rows and must write what the untuned reader wrote with the whole prefix in view:

    student view   head + [user: <AC part: head + prefix, tool definitions> + "\\n\\n" + COMPACTION_INSTRUCTIONS] + continuation
    teacher view   head + prefix + continuation                          (the raw text; read by the KL / match losses only)
    targets        the reader's recorded sampled tokens, one list per continuation assistant turn

The student view up to the first target is the prompt `Agent.compact("")` builds in a live run (activation/agent/agent.py):
system message(s) + first user message(s) + one user message [part, text], the part being
`activation_part("compaction", flattened view, tool definitions)` resolved with a compression target. The flattened view is
head + prefix with content lists joined to text (`strip_ac_parts`) and a head user message past AC_FLAT_HEAD_CHARS cut to
its head and tail (`head_message`), as `Agent.flattened_view` does. Tool results and nudges between the targets are
context, never targets. Messages after the last continuation assistant turn are dropped (nothing is predicted after them).

Sizes are fractions r of the PREFIX (ac_sizes.py): rows_wanted = round(prefix_tokens x r), and the part's compression
target is the one that gives exactly that many rows for this part's encoder-side length. Per item `info` records r,
rows_wanted, rows (after the encoder's own limits), side_tokens, prefix_tokens and `ratio` (= the part's compression
target, the meaning `ratio` has in the other packets of the bench).
  training   r drawn per item in [--r-range] from a generator seeded by (--seed, task id, item number). Default
             log-uniform: the evaluated sizes 1/20, 1/10, 1/5 are equally spaced in log, so each octave gets the same
             number of examples (uniform would put two thirds of them between 1/10 and 1/5). With --ratios-per-prefix K the
             range is cut into K equal slices and item k draws inside slice k, so the K items of a prefix differ in size.
  held       the runs of --held-repos repositories (seeded sample; no held repository trains), each present three times
             at r = 1/20, 1/10, 1/5 in panel groups continuation/r20, continuation/r10, continuation/r5.

Limits (the trainer's): reader history 72,000 tokens (both views), encoder input 65,536 positions, and the encoder's
max_view_rows (4,096). rows_wanted past max_view_rows is clamped by the encoder (a prefix longer than 20,480 tokens at
r = 1/5): the item is kept, marked `clamped`, and counted. An item past the token limits is dropped and counted with the
reason; a held item past them stops the build.

Recipes: --loss sft | kl | sft_match (several allowed). One loss writes the packet into --output; several write
--output/<loss>/, the same items in each (hard links), one recipe per folder.
  sft         loss_kind "sft": cross-entropy on the recorded tokens
  kl          loss_kind "kl": top-k KL to the starting reader reading the teacher view
  sft_match   sft + full_context_match_weight (--match-weight) with full_context_match_kinds ["compaction"]
Updates: 8 examples per update. A pass has ceil(training items / 8) updates; --epochs passes (default 1), or --updates U
to take the fewest whole passes that reach U. Schedule fields keep the trainer's defaults (cosine to 10 %, 50 warm-up
updates) unless set with --train KEY=JSON.

    python prepare_agent_continuation.py --material material.jsonl --output PACKETS/cont_v1 --loss sft kl \
        --initial-ac PAIR/ac --initial-reader PAIR/reader --held-repos 8 --ratios-per-prefix 4 --updates 1000

Clean compaction recipe, stage 1 (2026-11-15):
  --r-values 0.2 0.1   each training item draws its size from these values (seeded by --seed, run key, item number) instead of
                       the --r-range draw: one model for the two reported sizes.
  --held-repos 0       no training-side held panel (the held-out check is dev-75, compaction/stage1/build_dev_panel.py).
  run keys             a material line's `run_key` (export_pool_material.py: task, pool seed, cut) names the run: one item set per
                       run key, so later seeds of the same tasks and extra cuts of the same runs add items. Lines without one
                       (older material) are keyed by task id as before.
  CONTINUATION_BENCH_DIR  set it to IB/TMP/PAPER_CORPUS_QA/kit/bench (the sec35 bundle's bench): its run_qa.harness builds
                       the pair's own models and write mode from ac_modules.pt (2B encoder -> 9B reader, Hidden Residuals).

Which runs may train: --sides (default: training). Two exclusion levels:
  --evaluation-index  (repository level, the rule until Oct 14) a run from any repository holding an evaluation prefix is refused;
  --exclude-tasks     (task level, the user's rule from Oct 14 ≈16:30Z: same-repository training is allowed, since the paper
                      pitches domain specialization) only runs of the exact evaluation/development tasks are refused.
With --exclude-tasks, pass --sides training evaluation so evaluation-side tasks outside the frozen sets may train.
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from ac_sizes import compression_target_for, rows_wanted_for

# ======================================================================================================================
# EVERYTHING THIS SCRIPT TAKES FROM THE RESEARCH TREE (bench scripts + the `activation` package) IS IMPORTED HERE.
# Training runs from another copy of the tree: check each name below against that copy before building a packet there.
# Line numbers are those of /workspace at the time of writing (Oct 13 2026). B = IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench.
#
#   B/run_channel_mixture.py (imported as `mix`)
#     SEED :22                     the training config's seed, as in the other packets
#     panel_group(item) :31        the training group name written into the recipe (falls through to info["family"])
#     digest :78, write :79, file_sha :681, dump :684      the packet's hashes and files, written the trainer's way
#     reader_tokens(builder, study, item, example) :221    the longer of the two reader views (the 72,000 limit)
#     BUILD_CONTEXT :304, ForkPool :307, example_chunk :429 -> checked_example :416
#                                  builds each item's prepared example (limits 72,000 / 65,536 are written in checked_example),
#                                  sets info["target_tokens"], info["reader_tokens"] and example.execution_cost
#     StageClock :445, nested_parts :762
#     train :856                   the packet reader this script writes for. It reads: prepared.json {recipe_sha256, items_sha256,
#                                  prepared_examples_sha256, validated_capacity}; recipe.json {training_config, initial_pair
#                                  {ac_checkpoint, reader_checkpoint}, held {panel, panel_sha256, held_sha256}, epochs, phase,
#                                  actual_train, unique_sources, target_tokens}; prepared_examples.pkl {train, held, examples}
#                                  with example.item being the very item object; task_panel/panel.json {sha256, held_sha256,
#                                  items, groups}, panel.pkl {items, examples}, every panel item with info["panel_group"] (:877-:916).
#                                  For kl / sft_match its PreparedTrainer (:918-:931) adds the teacher view to the prepared
#                                  examples at load time: an older copy without that block cannot train those two from this packet.
#   B/run_qa.py
#     harness(ac_path, reader_path) :26     the model pair on CPU (tokenizers, row limits; min_view_rows=2); AC :7 its AC model name
#   B/run_trajectory_tasks.py
#     compact_example :257 (through example_chunk)
#   activation/ac_model/ac_model_training.py
#     ActivationContextTrainingItem :58     fields item_id, kind, in_context_messages, ac_messages, in_context_start, ac_start,
#                                           assistant_token_ids, tools, weight, dataset_id, doc_ids, info
#     ActivationContextTrainingConfig :78   fields loss_kind, learning_rate_ac, learning_rate_target_lora, examples_per_update,
#                                           num_epochs, completion_samples, teacher_cache_block_examples, logits_chunk_tokens,
#                                           gradient_checkpointing_min_tokens, seed, deterministic, schedule_total_updates,
#                                           checkpoint_every_updates, reporting_interval, shuffle_examples; for sft_match also
#                                           full_context_match_weight :115 and full_context_match_kinds :120
#     ActivationContextTrainer :436         build_example :458 (item.info["chat_template_kwargs"] is read there);
#                                           _Example :418 fields item, student_ids, target_ids, spans, part_requests
#   activation/ac_model/ac_model_study.py
#     ActivationContextStudyGenerator :150  .ac_model, .ac_name, .tokenizer
#   activation/ac_model/ac_model.py
#     ActivationContextModel                .config.min_view_rows :122 / .max_view_rows :123, .side.tokenizer, .pad_id :302,
#                                           num_view_rows :306 and part_view_rows :310 (rows = clamp(ceil(side tokens x target)))
#   activation/ac_model/ac_model_utils.py
#     tokenize_with_parts :77               the part's encoder-side tokens, exactly as part_view_rows counts them
#     EncodeRequest :732                    .compression_ratio of example.part_requests[0]
#   activation/common/ac_parts.py
#     AC_FLAT_MAX_TOKENS :18, AC_FLAT_HEAD_CHARS :19, COMPACTION_INSTRUCTIONS :44, activation_part :69,
#     resolve_activation_part :87 (reads config.ac_model_name and config.ac_compaction_ratio), strip_ac_parts :146
#   activation/agent/agent.py
#     head_message :65                      the head-and-tail cut of a long head user message
#   Mirrored, not imported: Agent.compact :344-369 and Agent.flattened_view :185-230 (see `flattened_view` and `build_item` below).
BENCH = Path(os.environ.get("CONTINUATION_BENCH_DIR") or Path(__file__).resolve().parents[3] / "ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench")
sys.path.insert(0, str(BENCH))
import run_channel_mixture as mix                                                                         # noqa: E402
from run_channel_mixture import ForkPool, StageClock, digest, dump, file_sha, nested_parts, write          # noqa: E402
from run_qa import AC, harness                                                                             # noqa: E402
from activation.ac_model import (ActivationContextStudyGenerator, ActivationContextTrainer,              # noqa: E402
                                 ActivationContextTrainingConfig, ActivationContextTrainingItem)
from activation.ac_model.ac_model_utils import tokenize_with_parts                                         # noqa: E402
from activation.agent.agent import head_message                                                            # noqa: E402
from activation.common.ac_parts import (AC_FLAT_HEAD_CHARS, AC_FLAT_MAX_TOKENS, COMPACTION_INSTRUCTIONS,  # noqa: E402
                                        activation_part, resolve_activation_part, strip_ac_parts)
# ======================================================================================================================

FAMILY = "agent_continuation"
HELD_SIZES = (("r20", 1 / 20), ("r10", 1 / 10), ("r5", 1 / 5))        # panel group continuation/<label> at this fraction of the prefix
READER_TOKENS, ENCODER_TOKENS = 72_000, 65_536                         # the limits written in mix.checked_example
EXAMPLES_PER_UPDATE = 8                                                # the trainer's fixed global batch (train() resumes by it)
LOSSES = ("sft", "kl", "sft_match")
CHUNK = 4                                                              # items per worker task (each holds a 20,000-token prefix)


def flattened_view(record: dict) -> list[dict]:
    """What Agent.flattened_view returns at the cut: head + prefix as flat text messages, a long head user message cut to head and tail."""
    head = [head_message(message, [], AC_FLAT_HEAD_CHARS) if message.get("role") == "user" else message
            for message in strip_ac_parts(deepcopy(record["head_messages"]))]
    return head + strip_ac_parts(deepcopy(record["prefix_messages"]))


def draw_r(seed: int, task_id: str, number: int, count: int, low: float, high: float, how: str) -> float:
    """Item `number` of `count` for one prefix: a seeded draw inside its slice of [low, high] (log-uniform or uniform)."""
    position = (number + random.Random(f"{seed}:r:{task_id}:{number}").random()) / count
    return low * (high / low) ** position if how == "log_uniform" else low + (high - low) * position


ESCAPED_TEACHER_PREFIX = False      # --escaped-teacher-prefix: the packets before 2026-11-17 (teacher view rendered with escaped earlier reasoning)


def build_item(record: dict, view: list[dict], side_tokens: int, r: float, item_id: str, group: str, ac, ac_name: str, nominal_prefix_tokens: int | None = None) -> ActivationContextTrainingItem:
    """
    One (prefix, size) item: the student view Agent.compact("") would build, the raw teacher view, the recorded targets.
    The teacher view's prefix turns are their recorded tokens (info["in_context_history_ids"] = the material's
    prefix_assistant_token_ids): the teacher reads the prompt the agent read, with real "</think>" delimiters, the prompt of the
    agent eval's `full` arm. Before 2026-11-17 (ESCAPED_TEACHER_PREFIX) the trainer rendered them by the chat template with the
    reasoning escaped to "[Reasoning]" / "[/Reasoning]" text, and the teacher copied "[/" at the first continuation turn
    (compaction/TEACHER_VIEW_FIX.md).
    """
    rows_wanted = rows_wanted_for(nominal_prefix_tokens or record["prefix_tokens"], r)   # the benchmark states sizes against the nominal 20,000-token prefix (1,000 / 2,000 / 4,000 rows)
    target = compression_target_for(rows_wanted, side_tokens)
    rows = ac.num_view_rows(side_tokens, target)
    if rows != min(ac.config.max_view_rows, max(ac.config.min_view_rows, rows_wanted)):
        raise ValueError(f"{item_id}: the encoder gives {rows} rows for {rows_wanted} wanted")
    tools = deepcopy(record["tools"]) or None
    part = resolve_activation_part(activation_part("compaction", deepcopy(view), deepcopy(tools)),
                                   SimpleNamespace(ac_model_name=ac_name, ac_compaction_ratio=target))
    messages = record["continuation_messages"]
    last = max(index for index, message in enumerate(messages) if message.get("role") == "assistant")
    continuation = messages[:last + 1]                                   # nothing is predicted after the last assistant turn
    head = deepcopy(record["head_messages"])
    student = head + [{"role": "user", "content": [part, {"type": "text", "text": "\n\n" + COMPACTION_INSTRUCTIONS}]}]
    teacher = deepcopy(record["head_messages"]) + deepcopy(record["prefix_messages"])
    return ActivationContextTrainingItem(
        item_id=item_id, kind="compaction",
        in_context_messages=teacher + deepcopy(continuation), ac_messages=student + deepcopy(continuation),
        in_context_start=len(teacher), ac_start=len(student),
        assistant_token_ids=[list(ids) for ids in record["target_token_ids"]],
        tools=tools, weight=1.0, dataset_id=record.get("benchmark") or FAMILY, doc_ids=[record["task_id"]],
        info={"family": FAMILY, "source": FAMILY, "panel_group": group, "chat_template_kwargs": dict(record["chat_template_kwargs"]),
              "task_id": record["task_id"], "run_key": record.get("run_key") or record["task_id"], "seed": record.get("seed"), "repo": record["repo"], "language": record.get("language"), "benchmark": record.get("benchmark"),
              "side": record.get("side"), "solved": record.get("solved"), "finish_reason": record.get("finish_reason"),
              "r": r, "ratio": target, "rows_wanted": rows_wanted, "rows": rows, "clamped": rows_wanted > ac.config.max_view_rows,
              "side_tokens": side_tokens, "prefix_tokens": record["prefix_tokens"], "head_tokens": record["tokens"]["head"], "assistant_turns": len(record["target_token_ids"]),
              **teacher_history(record)})


def teacher_history(record: dict) -> dict:
    """info["in_context_history_ids"]: the recorded tokens of the prefix assistant turns (none with ESCAPED_TEACHER_PREFIX)."""
    if ESCAPED_TEACHER_PREFIX:
        return {}
    ids = record.get("prefix_assistant_token_ids")
    if ids is None:
        raise ValueError(f"{record.get('run_key') or record['task_id']}: material without prefix_assistant_token_ids (export it again with "
                         "export_pool_material.py from 2026-11-17, or build with --escaped-teacher-prefix for the old teacher view)")
    if len(ids) != sum(message.get("role") == "assistant" for message in record["prefix_messages"]):
        raise ValueError(f"{record.get('run_key') or record['task_id']}: prefix_assistant_token_ids do not match the prefix's assistant turns")
    return {"in_context_history_ids": [list(turn) for turn in ids]}


def failure_reason(error: str) -> str:
    """A build failure's text as one of the limits (or the text itself, shortened)."""
    if "72000" in error or "max_example_tokens" in error:
        return f"reader history over {READER_TOKENS:,} tokens"
    if "AC side forward" in error:
        return f"encoder input over {ENCODER_TOKENS:,} positions"
    return error[:160]


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--material", type=Path, nargs="+", required=True, help="material.jsonl file(s) of stage 1")
    p.add_argument("--output", type=Path, required=True, help="the packet folder (one --loss) or the folder holding one packet per loss")
    p.add_argument("--loss", nargs="+", choices=LOSSES, default=["sft"])
    p.add_argument("--initial-ac", required=True, help="AC checkpoint the run starts from, as the training machine will find it")
    p.add_argument("--initial-reader", required=True, help="reader adapter checkpoint the run starts from, as the training machine will find it")
    p.add_argument("--fresh-harness", action="store_true", help="the starting checkpoints are not on THIS machine: prepare with fresh modules "
                   "(only the tokenizers and the row limits are used here, and they do not depend on the checkpoint)")
    p.add_argument("--lr-ac", type=float, default=2e-5)
    p.add_argument("--lr-reader", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=None, help="passes over the training items (default 1)")
    p.add_argument("--updates", type=int, default=None, help="instead of --epochs: the fewest whole passes that give at least this many updates")
    p.add_argument("--match-weight", type=float, default=1.0, help="sft_match: full_context_match_weight")
    p.add_argument("--train", action="append", default=[], metavar="KEY=JSON", help="set another training-config field in every recipe (lr_schedule=\"constant\")")
    p.add_argument("--ratios-per-prefix", type=int, default=1, help="training items per prefix, each at its own size")
    p.add_argument("--r-range", type=float, nargs=2, default=(1 / 20, 1 / 5), metavar=("LOW", "HIGH"), help="training sizes, as fractions of the prefix")
    p.add_argument("--r-draw", choices=("log_uniform", "uniform"), default="log_uniform")
    p.add_argument("--r-values", type=float, nargs="+", default=None, help="training sizes drawn from these values instead of --r-range (e.g. 0.2 0.1)")
    p.add_argument("--nominal-prefix-tokens", type=int, default=20_000, help="sizes are fractions of this length, as in the benchmark (rows = 20,000 x r); 0: of each prefix's own length")
    p.add_argument("--held-repos", type=int, required=True, help="repositories whose runs form the held panel (never trained)")
    p.add_argument("--sides", nargs="+", default=["training"], help="split sides whose runs may be used (other runs are left out and counted)")
    p.add_argument("--evaluation-index", type=Path, default=None, help="index.json of the frozen evaluation prefixes: runs from its repositories are left out")
    p.add_argument("--exclude-tasks", type=Path, nargs="+", default=[], help="index.json file(s) of evaluation/development prefixes: runs of those exact tasks are left out (task-level; other tasks of the same repositories may train)")
    p.add_argument("--seed", type=int, default=20261013, help="held repositories and size draws")
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--reporting-interval", type=float, default=.125, help="panel spacing, a fraction of each pass")
    p.add_argument("--workers", type=int, default=0, help="fork workers (0: every CPU of the allocation; 1: one process)")
    p.add_argument("--escaped-teacher-prefix", action="store_true", help="the teacher view of the packets before 2026-11-17 (earlier reasoning "
                   "rendered as escaped [Reasoning] text); default: the recorded prefix tokens (needs material with prefix_assistant_token_ids)")
    a = p.parse_args()
    global ESCAPED_TEACHER_PREFIX
    ESCAPED_TEACHER_PREFIX = a.escaped_teacher_prefix
    if len(set(a.loss)) != len(a.loss):
        p.error("--loss names a recipe twice")
    if a.epochs is not None and a.updates is not None:
        p.error("give --epochs or --updates, not both")
    if not 0 < a.r_range[0] <= a.r_range[1] or a.ratios_per_prefix < 1 or a.held_repos < 0 or any(not 0 < r <= 1 for r in a.r_values or []):
        p.error("--r-range needs 0 < LOW <= HIGH; --r-values in (0, 1]; --ratios-per-prefix at least 1; --held-repos at least 0")
    overrides = {}
    for entry in a.train:
        key, _, value = entry.partition("=")
        overrides[key] = json.loads(value)
    folders = {loss: a.output if len(a.loss) == 1 else a.output / loss for loss in a.loss}
    for folder in folders.values():
        if (folder / "prepared.json").exists() or (folder / "task_panel" / "panel.json").exists():
            raise FileExistsError(f"{folder}: packet or panel already committed; use a new output")
    workers = a.workers or len(os.sched_getaffinity(0))
    if workers > 1:
        import torch
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized():
            raise ValueError("Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ---- the runs: training-side only, none from an evaluation repository, one per task
    excluded = Counter()
    evaluation_repos = set()
    if a.evaluation_index:
        evaluation_repos = {entry["repo"] for entry in json.loads(a.evaluation_index.read_text())["prefixes"]}
    excluded_tasks = {entry["task_id"] for path in a.exclude_tasks for entry in json.loads(path.read_text())["prefixes"]}
    records, sources = {}, []
    for path in a.material:
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                count += 1
                if record.get("side") not in a.sides:
                    excluded[f"side {record.get('side')}"] += 1
                elif record["repo"] in evaluation_repos:
                    excluded["repository holds an evaluation prefix"] += 1
                elif record["task_id"] in excluded_tasks:
                    excluded["task is an evaluation or development prefix"] += 1
                elif not record.get("repo"):
                    excluded["no repository recorded (the held split is by repository)"] += 1
                elif (record.get("run_key") or record["task_id"]) in records:
                    excluded["run key already taken from an earlier line"] += 1
                else:
                    records[record.get("run_key") or record["task_id"]] = record
        sources.append({"path": str(path.resolve()), "sha256": file_sha(path), "runs": count})
    if not records:
        raise ValueError(f"no usable run in the material: {dict(excluded)}")
    repos = sorted({record["repo"] for record in records.values()})
    if a.held_repos >= len(repos):
        raise ValueError(f"--held-repos {a.held_repos} leaves no training repository ({len(repos)} in the material)")
    held_repos = sorted(random.Random(f"{a.seed}:held_repos").sample(repos, a.held_repos))
    held_runs = [record for record in records.values() if record["repo"] in held_repos]
    train_runs = [record for record in records.values() if record["repo"] not in held_repos]
    print(json.dumps({"runs": len(records), "repositories": len(repos), "excluded": dict(excluded), "held_repositories": held_repos,
                      "held_runs": len(held_runs), "training_runs": len(train_runs)}), flush=True)

    # ---- the model pair on CPU: tokenizers and row limits only
    h = harness() if a.fresh_harness else harness(a.initial_ac, a.initial_reader)
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    if workers > 1:
        torch.set_num_threads(1)

    # ---- items
    low, high = a.r_range
    items, panel_count = [], 0
    for record in train_runs + held_runs:
        view = flattened_view(record)
        ids, _ = tokenize_with_parts(ac.side.tokenizer, view, [], ac.pad_id, tools=record["tools"] or None)   # as part_view_rows counts
        side_tokens = len(ids)
        if side_tokens > AC_FLAT_MAX_TOKENS:                             # a live flattened view would drop its oldest messages: not this view
            excluded["prefix view over the flattened-view budget"] += 1
            continue
        key = record.get("run_key") or record["task_id"]
        if record["repo"] in held_repos:
            sizes = [(f"panel:continuation/{label}:{key}", f"continuation/{label}", r) for label, r in HELD_SIZES]
            panel_count += len(sizes)
        elif a.r_values:
            sizes = [(f"{FAMILY}:{key}:{number}", "continuation/train",
                      a.r_values[random.Random(f"{a.seed}:r:{key}:{number}").randrange(len(a.r_values))]) for number in range(a.ratios_per_prefix)]
        else:
            sizes = [(f"{FAMILY}:{key}:{number}", "continuation/train",
                      draw_r(a.seed, key, number, a.ratios_per_prefix, low, high, a.r_draw)) for number in range(a.ratios_per_prefix)]
        for item_id, group, r in sizes:
            item = build_item(record, view, side_tokens, r, item_id, group, ac, study.ac_name, a.nominal_prefix_tokens or None)
            if group != "continuation/train":
                item.info.update(panel=True, reconstruction_overlap=False)
            items.append(item)

    # ---- prepared examples: the trainer's own builder and limits (mix.checked_example), failures recorded
    steps_guess = math.ceil(max(1, len(items) - panel_count) / EXAMPLES_PER_UPDATE)

    def config_of(loss: str, steps: int, epochs: int) -> ActivationContextTrainingConfig:
        kwargs = dict(loss_kind="kl" if loss == "kl" else "sft", learning_rate_ac=a.lr_ac, learning_rate_target_lora=a.lr_reader,
                      examples_per_update=EXAMPLES_PER_UPDATE, num_epochs=epochs, completion_samples=0, teacher_cache_block_examples=32,
                      logits_chunk_tokens=64, gradient_checkpointing_min_tokens=0, seed=mix.SEED, deterministic=False,
                      schedule_total_updates=steps * epochs, checkpoint_every_updates=a.checkpoint_every,
                      reporting_interval=a.reporting_interval, shuffle_examples=True)
        if loss == "sft_match":
            kwargs.update(full_context_match_weight=a.match_weight, full_context_match_kinds=("compaction",))
        kwargs.update(overrides)
        return ActivationContextTrainingConfig(**kwargs)

    # Examples are built the SFT way whatever the recipe: the student view does not depend on the loss, and the trainer adds
    # the teacher view itself when a recipe needs it (PreparedTrainer in run_channel_mixture.train).
    trainer = ActivationContextTrainer(h, config_of("sft", steps_guess, 1))
    clock = StageClock("examples", len(items))
    built, failures, done = [], [], 0
    mix.BUILD_CONTEXT = {"trainer": trainer, "study": study, "scan": True, "output": items}
    with ForkPool(workers) as pool:
        for results in pool.map(mix.example_chunk, [range(i, min(i + CHUNK, len(items))) for i in range(0, len(items), CHUNK)]):
            for value, failure in results:
                if failure:
                    failures.append(failure)
                else:
                    built.append(value[:2])
            done += len(results)
            clock(done)
    mix.BUILD_CONTEXT = None
    for item, example in built:
        (request,) = example.part_requests
        (start, end), = example.spans
        if request.compression_ratio != item.info["ratio"] or end - start != item.info["rows"]:
            raise ValueError(f"{item.item_id}: the prepared example has {end - start} rows at target {request.compression_ratio}, "
                             f"the item says {item.info['rows']} at {item.info['ratio']}")
        if math.ceil(item.info["side_tokens"] * item.info["ratio"]) != item.info["rows_wanted"]:
            raise ValueError(f"{item.item_id}: ceil(side tokens x target) is not the wanted row count")
        if nested_parts(item):
            raise ValueError(f"{item.item_id}: an AC part nests another part (depth-1 rule)")
    train = [(item, example) for item, example in built if not item.info.get("panel")]
    panel = [(item, example) for item, example in built if item.info.get("panel")]
    panel_failures = [failure for failure in failures if failure["id"].startswith("panel:")]
    if panel_failures or len(panel) != panel_count:
        raise ValueError(f"held panel items failed to build: {panel_failures[:3]}")
    if not train:
        raise ValueError(f"no training item was built: {failures[:3]}")
    train_items, panel_items = [item for item, _ in train], [item for item, _ in panel]
    if {item.info["repo"] for item in train_items} & {item.info["repo"] for item in panel_items}:
        raise ValueError("a held repository trains")
    if len({item.item_id for item in train_items + panel_items}) != len(train_items) + len(panel_items):
        raise ValueError("duplicate item ids")
    dropped = Counter(failure_reason(failure["error"]) for failure in failures)

    steps = math.ceil(len(train_items) / EXAMPLES_PER_UPDATE)
    epochs = max(1, math.ceil(a.updates / steps)) if a.updates is not None else (a.epochs or 1)
    group = mix.panel_group(train_items[0])
    target_tokens = sum(item.info["target_tokens"] for item in train_items)
    panel_groups = Counter(item.info["panel_group"] for item in panel_items)
    pair = {"ac_checkpoint": str(a.initial_ac), "reader_checkpoint": str(a.initial_reader)}

    def clamp_counts(chosen: list) -> dict:
        return {"items": len(chosen), "clamped_to_max_view_rows": sum(item.info["clamped"] for item in chosen)}

    report = {
        "material": sources, "runs": len(records), "excluded": dict(excluded), "training_runs": len({item.info["run_key"] for item in train_items}),
        "training_tasks": len({item.info["task_id"] for item in train_items}), "training_seeds": dict(Counter(str(item.info["seed"]) for item in train_items)),
        "training_r_counts": {str(r): n for r, n in sorted(Counter(round(item.info["r"], 4) for item in train_items).items())},
        "held_runs": len({item.info["task_id"] for item in panel_items}), "held_repositories": held_repos,
        "training_repositories": len({item.info["repo"] for item in train_items}),
        "training": clamp_counts(train_items), "held": {name: clamp_counts([item for item in panel_items if item.info["panel_group"] == name]) for name in sorted(panel_groups)},
        "dropped": dict(dropped), "max_view_rows": ac.config.max_view_rows, "limits": {"reader_tokens": READER_TOKENS, "encoder_tokens": ENCODER_TOKENS},
        "r_range": [low, high], "r_draw": a.r_draw, "r_values": a.r_values, "ratios_per_prefix": a.ratios_per_prefix,
        "training_r": sorted(round(item.info["r"], 4) for item in train_items),
        "updates_per_pass": steps, "epochs": epochs, "updates": steps * epochs, "updates_requested": a.updates,
        "training_target_tokens": target_tokens, "mean_reader_tokens": round(sum(item.info["reader_tokens"] for item in train_items) / len(train_items)),
        "mean_rows": round(sum(item.info["rows"] for item in train_items) / len(train_items)), "packets": {loss: str(folder.resolve()) for loss, folder in folders.items()},
    }

    # ---- the packet(s): one set of files, one recipe per loss
    first = None
    for loss, out in folders.items():
        panel_dir = out / "task_panel"
        panel_dir.mkdir(parents=True, exist_ok=True)
        if first is None:
            write(panel_dir / "held.json", {"rag_qa": {"questions": []}, "trajectory_qa": {"ids": []},      # the two keys mix.load_held reads
                                            FAMILY: {"rule": f"every run of {a.held_repos} repositories drawn with seed {a.seed}; no held repository trains",
                                                     "repos": held_repos, "task_ids": sorted({item.info["task_id"] for item in panel_items})}})
            dump(panel_dir / "panel.pkl", {"items": panel_items, "examples": [example for _, example in panel]})
            write(panel_dir / "panel.json", {
                "sha256": file_sha(panel_dir / "panel.pkl"), "held_sha256": file_sha(panel_dir / "held.json"), "items": len(panel_items),
                "groups": dict(sorted(panel_groups.items())), "target_tokens": sum(item.info["target_tokens"] for item in panel_items),
                "reader_tokens": sum(item.info["reader_tokens"] for item in panel_items), "initial_pair": pair, "seed": a.seed,
                "sizes": {f"continuation/{label}": r for label, r in HELD_SIZES},
                "rule": "the continuation runs of the held repositories, each at three fixed sizes (fractions of the prefix)"})
            bundle = {"train": train_items, "held": []}
            dump(out / "items.pkl", bundle)
            dump(out / "prepared_examples.pkl", {**bundle, "examples": [example for _, example in train]})
            first = out
        else:
            for name in ("task_panel/held.json", "task_panel/panel.pkl", "task_panel/panel.json", "items.pkl", "prepared_examples.pkl"):
                link_or_copy(first / name, out / name)
        config = config_of(loss, steps, epochs)
        arm = {"task": "agent continuation after a compacted prefix", "loss": loss, "lr_ac": config.learning_rate_ac, "lr_reader": config.learning_rate_target_lora,
               "r_range": [low, high], "r_draw": a.r_draw, "r_values": a.r_values, "ratios_per_prefix": a.ratios_per_prefix, "held_sizes": [r for _, r in HELD_SIZES],
               "teacher_view": "escaped" if a.escaped_teacher_prefix else "recorded",
               **({"match_weight": a.match_weight} if loss == "sft_match" else {}), **({"train_overrides": overrides} if overrides else {})}
        recipe = {"seed": a.seed, "epochs": epochs, "phase": "Agent continuation after a compacted prefix (%s): %s" % (loss, json.dumps(arm, sort_keys=True)),
                  "arm": arm, "counts": {"wanted": {group: len(train_runs) * a.ratios_per_prefix}}, "actual_train": {group: len(train_items)},
                  "unique_sources": {group: len(train_items)}, "target_tokens": {group: target_tokens}, "excluded": dict(excluded), "failures": failures,
                  "loss_shares": "natural: every example weighs 1", "global_batch": EXAMPLES_PER_UPDATE, "material": sources, "sides": a.sides,
                  "held": {"rule": "task_panel/held.json", "panel": str(panel_dir.resolve()), "panel_sha256": file_sha(panel_dir / "panel.pkl"),
                           "held_sha256": file_sha(panel_dir / "held.json")},
                  "headline_note": "the headline held loss averages the three sizes: read the groups",
                  "sizes": {key: report[key] for key in ("training", "held", "dropped", "max_view_rows")},
                  "initial_pair": pair, "training_config": asdict(config)}
        write(out / "recipe.json", recipe)
        write(out / "prepared.json", {
            "recipe_sha256": digest(recipe), "items_sha256": file_sha(out / "items.pkl"), "items_bytes": (out / "items.pkl").stat().st_size,
            "prepared_examples_sha256": file_sha(out / "prepared_examples.pkl"),
            "validated_capacity": {"reader_tokens": READER_TOKENS, "encoder_tokens": ENCODER_TOKENS},
            "training_examples": len(train_items), "held_examples": 0, "one_pass": epochs == 1, "preparation": {"workers": workers, "stage_seconds": {"examples": clock.seconds()}}})
        write(out / "continuation_build.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "training_r"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
