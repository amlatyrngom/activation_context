# Mixed run plan — agent-facing source

Status: **decisions integrated (review Sept 27 14:55Z); M2 code + M4 probes in progress; launch ON HOLD (D9)**. Human projection:
`MIXED_RUN_PLAN.tressoir.md` (same folder). This file is the verbose source of truth; keep both in agreement.

## 1. Goal and accepted shape

One mixed training pass that teaches the reader to answer from activation-context (AC) inputs (RAG QA,
trajectory QA, compaction summaries) while keeping ordinary agent behavior (text-only agent SFT), starting from the
finished 75M reconstruction pair. Reconstruction-only training made RAG/trajectory QA worse on the task panel
(e.g. rag_qa/contextual 1.099 → 1.317 nats/token, trajectory_qa/multi 2.795 → 3.452); this run is the fix.

Accepted in chat (September 27):

| Item | Accepted value |
|---|---|
| Multi-input counting | Option (a): each source row used once; multi-input groups = 10% of a family's examples (2–3 inputs each, shortest intact sources first) |
| RAG QA | whole bank, 70/20/10 contextual / uncontextual / multi of examples |
| Trajectory QA | whole bank, same 70/20/10 |
| Compaction | whole bank (≈65.9k histories), 10% multi |
| Regular agent SFT | every exported segment (≈23k), text only, exactly what the agent saw, never an AC part |
| Evidence QA | 10k plain teacher trajectories |
| Reconstruction | dropped (superseded by the uncontextual Q/A views) |
| Recursive (parent-context) Q/A view | dropped (depth-1 rule) |
| Held-out percentages of the source banks | non-canonical; ignored completely |
| Panel | fundamentals only: flat single RAG QA and trajectory QA, more items per group than 32 |
| Wall-clock | up to ~72h acceptable with regular panel updates |

## 2. Data inventory (verified)

| Family | Bank (manifest) | Location | Notes |
|---|---|---|---|
| rag_qa | 100,000 rows | CSAIL `$R/data/rag_qa`, local `IB/TMP/SYNC/REUSABLE_DATA/rag_qa` | `reserved_questions.json`: 1,900 questions reserved for existing evaluations (manifest: "respect reserved questions") |
| trajectory_qa | 112,007 rows | `$R/data/trajectory_qa` | question + exact source history per row |
| compaction | 65,924 rows | `$R/data/compaction` | 27B oracle summaries; official compaction target (canon) |
| agent_teacher (evidence QA) | 66,146 rows | `$R/data/agent_teacher` | plain single-agent runs; builder raises on AC parts/compactions/subagents |
| agent_regular, old campaign | 12,103 segments | `$R/data/agent_teacher_regular/campaign_12h_20260911` | 7,446 subagent segments; 1,156 segments carry the old compaction wording; 0 segments with non-text content |
| agent_regular, v1 | 62 segments (local only) | `IB/TMP/SYNC/.../regular_v1_s*` | same seed-1 task draw as v2 → **excluded** (same tasks; possible identity collision) |
| agent_regular, v2 | 11,313 segments (9,391 roots, 53.6M output tokens) | releases `teacher_regular_v2_20260927_s{0..3}` in `agent_teacher_regular` (CSAIL, S3) | exported 17:05Z; 0 non-text segments; root finish: submitted 9,431 / max_duration 1,754 / no_tool_call 96 / max_tool_errors 32 segments; 10,522 main + 791 subagent |

v1 cache count (ORCD job 24096068): 676 rows / 720 segments / 2.8M output tokens; same tasks as v2; not exported.

Old campaign segments by root finish reason: submitted 11,257; max_duration 627; max_tool_errors 117; no_tool_call 87;
max_turns 8; trajectory_cap 7. v2 rows at 11:43Z: submitted 5,046; max_duration 310; no_tool_call 52; max_tool_errors 17;
task_unavailable 7 (no recorded turn → no segment).

### v2 "AC parts" finding

Earlier I reported that ~40% of v2 rows "have flat AC parts". That count came from `rowcheck2.py`, which counts any
dict holding `messages` plus `kind`/`tools`. The v2 teacher ran **without an AC model** (plain Qwen3.8-27B-FP8 in vLLM;
base teacher only, sequential/parallel multi-agent weights 0). Long tool outputs record an unrendered
`activation_content` description inside `tool_results` ("recorded whether or not it was rendered",
`agent_config.py` step record); the agent itself saw the truncated text (`agent_tools.py`, "full output written to
<path>"). `Agent.render_activation` returns `[]` without an AC model, so compaction resumes are text
(`COMPACTION_TEXT_INSTRUCTIONS`). The export takes `prompt_messages` + each step's `messages` and never `tool_results`,
so it already yields what the agent saw. A hard assertion (no `activation_context` item in any exported message) makes
this explicit.

## 3. Training examples and time estimate

Multi-input packing (`pack_multi_targets`): `groups = round(rows × f / (1 + 1.5 f))`, f = 0.1, alternating 2/3 inputs →
rows ≈ 1.15 × examples. Every row once.

| Family | Training rows | Examples | Views | Target tokens (est.) | s/example (est.) | Hours |
|---|---|---|---|---|---|---|
| RAG QA | ≈98k (100k − 1,900 reserved − panel) | ≈85k | ≈59.6k ctx / 17.0k unctx / 8.5k multi | ≈4.3M (≈44/row) | 0.68 single, ≈0.96 multi | ≈16.7 |
| Trajectory QA | ≈111.8k | ≈97k | ≈68k / 19.5k / 9.7k | ≈5.6M (≈50/row) | 0.78, ≈1.2 | ≈22.2 |
| Compaction | ≈65.9k | ≈57k | ≈51.6k single / 5.7k multi | ≈73.5M (≈1,116/row) | 0.97, ≈1.6 | ≈16.4 |
| Regular agent | ≈23k segments | ≈23k | text only | ≈119M (old-campaign mean 5,188/segment) | 1.2–1.5 (unmeasured) | ≈7.7–9.6 |
| Evidence QA | 10k | 10k | text only | ≈4.5M | 0.64 | ≈1.8 |
| **Total** | | **≈272k (≈34k updates at batch 8)** | | **≈207M** | | **≈64.8–66.7** |

Overheads: panels ≈1h (20 panels × ~3 min on 256 short items), checkpoint saves ≈1h (every 250 updates, ~25 s each
inferred from the 75M log), startup and fixed references ≈0.3h → **≈67–69h**; plausible range 60–80h until the M4
probe. Calibration: step-time model `0.50 s + 5.5e-5 s × (reader tokens + encoder tokens / 4.5)` from the 2×H200
canonical mixed fixture (5.53 s/update, 108 items) and the 75M run (4,736 updates, 19,606 s of step time = 0.52 s/example;
25,044 s wall). Multi groups come from the shortest sources, so the multi costs above are conservative. Agent segments
(mean 12,570 tokens, p90 36,207, max 61,820) were never timed.

Fixture per-family sizes (reader / encoder / target tokens): trajectory_qa 1,379 / 16,429 / 50; compaction 4,346 / 18,930 /
1,116; agent_teacher 2,561 / 0 / 446; rag_qa 1,107 / 10,072 / 44.

Loss normalization: each example's objective is its token-mean cross-entropy, then `× item.weight / batch`
(`ac_model_training.py:774`, `:1480`). A 5k-token agent segment counts as one example, like a 44-token QA answer. With
natural weights the loss shares by example count are RAG 31%, trajectory QA 36%, compaction 21%, regular agent 8.5%,
evidence QA 3.7%, although agent SFT carries ~58% of target tokens.

## 4. Recipe (canonical, unchanged)

Literal gold-token SFT; AC encoder/head LR 5e-5, reader LoRA LR 2e-5; AdamW; 8 examples per update; 50-update linear
warm-up; cosine to 10%; joint gradient clip 1.0; base-reader no-context KL penalty (weight 1 on a random 25% of
AC-bearing examples, scaled by 1/0.25); `teacher_cache_block_examples=32`; `logits_chunk_tokens=64`; gradient
checkpointing on; `max_example_tokens=72,000`; encoder part capacity 65,536. Task ratios sampled uniformly in
[1/20, 1/10]; panel fixed at 1/16. Initialization: the 75M pair
(`$R/run_75m_cont/AC_MODELS/rag_poc/epoch_001`, `$R/run_75m_cont/LORAS/rag_reader/epoch_001`; 2B encoder / 9B reader),
fresh optimizer (the runner's prepare path), new schedule over the mixed pass (decision D4).

Changes from the runner's current defaults: `reporting_interval` 0.1 → 0.05; `checkpoint_every_updates` 50 → 250;
family shares → decision D1.

### Penalty / extra-term audit for the text-only families (verified)

- Evidence QA: `make_item` raises on `ac_model_name`, compactions or subagent results (`run_channel_mixture.py:162`).
- Regular agent: `ActivationContextTrainingItem(ident,'agent_sft',None,messages,0,start,targets,...)` — no
  `in_context_messages`, no part requests.
- No-context penalty: built only when an item has AC parts (`ac_model_training.py:420`).
- No KL teacher: `loss_kind='sft'`.
- No encoder call (`_example_loss` encodes only with `part_requests`) → no encoder gradient; only the reader LoRA learns.
- Not multi-grouped (`pack_multi_targets` passes `AGENT_FAMILIES` through), no conditioning views.
- Targets: every assistant turn after `target_start`, reasoning and tool calls included (`enable_thinking=True`, canon).
- Eval: their no-context and full-context reference views are identical (panel cost only; irrelevant now since no
  agent panel group).

## 5. Held-out rule and panel

Canon: task-aware QA holds out **questions**, not source contexts; no unseen-context claim; no routine
reconstruction-overlap audits. The sources' split fields (`row['split']`, the 1% `question_held` hash, the regular
export's 4% root split) are ignored.

- Draw, seeded (`digest((SEED,'task_panel',identity))` order) over eligible rows: N RAG questions (excluding the 1,900
  reserved) and N trajectory QA rows. N = 64 (D5: the panel only). The `panel` phase writes the drawn
  identities to `panel/held.json`; `prepare` reads it and excludes exactly those (one source of truth).
- Held RAG questions: never train through RAG **or** evidence QA (shared question bank; normalized text match).
- Held trajectory QA rows: excluded by row id. Other questions on the same history, and compaction of that history,
  may train (canon: shared source facts allowed). Drop the `root_splits` "Trajectory root crosses train/held" check.
- Reserved RAG questions: excluded from RAG and evidence QA training (unchanged).
- Compaction, regular agent, evidence QA: no held split; everything eligible trains.
- Live panel = the first 64 drawn rows per QA family, each under both views: `rag_qa/contextual`, `rag_qa/uncontextual`,
  `trajectory_qa/contextual`, `trajectory_qa/uncontextual` → 256 items, ratio 1/16. Headline = mean over the 256 items;
  groups reported separately with the fixed initial-reader no-context and full-context references.
- Dropped panel groups: multi, recursive, compaction, reconstruction, agent. Their training NLL stays visible through
  `training_nll_by_group` (last ≤100 training objectives per group) in the report, and through the fixed
  `--learning-items 100` training sample (seeded, spread over the training groups) that every panel re-evaluates.

## 6. Milestones

### M1 — Agent SFT data: v2 export and union

After CAMPAIGN_END (17:19Z) and the shards' clean stop: one ORCD CPU job (`teacher_regular_v2/export.sbatch`,
`mit_normal`, 8 CPUs, 64G, 3h) reads the four v2 shard caches (`$R/sync/ROLLOUTS/teacher_regular_v2_s{0..3}/rollouts.jsonl`)
on a compute node and runs `export_agent_teacher_segments.py --tokenizer Qwen/Qwen3.8-27B-FP8 --model-id
Qwen/Qwen3.8-27B-FP8` into release folders `regular_v2_s{0..3}_<stamp>`. Add the hard
assertion (no `activation_context` item in any exported message). Copy to the container (guarded helper, one transfer),
upload to `s3://general-bucket-aln/activation-context-data/datasets/agent_teacher_regular` from the container, stage on
CSAIL `$R/data/agent_teacher_regular` next to `campaign_12h_20260911`, update the collection manifest. v1 is not
staged. Check: segment count, duplicate `segment_id`s = 0, AC-part assertion passed, sampled token check (200 segments
per source: chat template of the exported messages with the teacher tokenizer reproduces the recorded
`prompt_token_ids` + step token ids; report the mismatch rate, investigate if systematic).

**Done Sept 27.** Export job 24110915 (1.6 min; the exporter's AC-part guard passed), tar sha256 `955d744a…`, copied to
the container, uploaded 17:11Z (S3 collection manifest lists 7 releases: the 3 earlier + 4 v2), staged on CSAIL by
`mixed_v1/stage_v2.sbatch` (job 2363967: tar and every shard checksum verified, collection manifest appended
atomically → `campaign_12h_20260911` + 4 v2 releases). Duplicate `segment_id`s: 0 of 11,313. Fidelity check
(`check_exported_segments.py`, instead of a template re-render, which differs by design since training writes earlier
reasoning as `[Reasoning]` text): every text of an exported message (prompt, tool output, assistant text, reasoning,
string call arguments) must appear verbatim (NFC) in the tokens the teacher read or sampled at that point. Old campaign:
250 segments (25 per benchmark), 12.6M characters, 0 misses; negative controls (extended tool output, changed prompt,
dropped message, extra assistant text) are all caught. v2 (ORCD job 24112783, 25 per benchmark and release): 1,000
segments, 48.4M characters over all 10 benchmarks, 0 misses. Receipt: `AGENT_REGULAR_V2_EXPORT.json`.

### M2 — Runner changes (`run_channel_mixture.py`) — code done Sept 27 ~15:20Z; review 1 applied ~15:50Z (re-check running)

1. `TRAIN_FAMILIES` (this run: no reconstruction) separate from `FAMILIES` (every family the packer/reports know; the
   reconstruction runner's shared `pack_multi_targets` iterates it); `SHARES=None` (D1: every example weight 1).
2. `VIEWS=('question','summary')` separate from the `MODES` report labels; `conditioning()` = 7/9 contextual, 2/9
   uncontextual among singles (→ 70/20/10 of examples after multi packing).
3. `draw_held()` (seeded over usable rows; RAG by normalized question text, one row per text, reserved skipped; trajectory
   Q/A by row id) + `row_split(family,row,reserved,held)` (section 5) + `load_held()` (held.json checked against
   panel.json's sha256); `select_rows()` keeps train rows only and fails if a drawn held row is missing from its bank; the
   root-crossing check is removed.
4. `agent_regular`: raises on any AC part; drops roots per D2 (`DROPPED_ROOT_FINISH`); old-wording rewrite kept (D3).
5. `build_panel()`: 2 QA families × 2 views × 64 (the same questions under each view); writes `held.json`; the overlap
   corpus, diagnostics and nested groups are gone (`multi_groups`/`source_texts`/`overlaps` removed).
6. `prepare()`: no prepared held set (`held: []`); `reporting_interval=.05`, `checkpoint_every_updates=250` (flags);
   every build validated (reader ≤72,000, targets >0, part ≤65,536, no nesting, no parts on agent items); recipe records
   the panel identity, exclusions, D-choices; `--scan` (failures recorded, `scan.json` + `probe_ids.json`, no pickles),
   `--subset`, `--exclude`, `--probe` (heaviest first, no shuffle).
7. `train()`: headline over the task panel when the packet has no held items (`headline_ids`), in `reference()` and
   `panel()`; panel rows record `held eval seconds`, `learning eval seconds`, `callback seconds`, UTC.
8. CLI defaults: `--qa-questions`/`--summary-histories`/`--regular-segments` 10^9, `--teacher-histories 10000`, token caps
   off, `--per-group 64`, `--checkpoint-every 250`, `--reporting-interval .05`, `--snapshot-percent 10`.
9. Snapshots (D8): `snapshot_percent()` + `snapshot()` in `panel()` (main rank, no collectives) save
   `SNAPSHOTS/epoch_001_<pct>pct/{ac,reader}` (update number in `snapshot.json`) + `snapshot.json` at every 10% boundary below 100%; taken ones are skipped
   after a resume, an interrupted one (no snapshot.json) is removed and redone; the trainer's partial pruning never sees them.
10. `export_agent_teacher_segments.py · segment_rows()`: raises on an AC part in what the agent saw.
11. New: `bench/check_mixed_run.py` (CPU checks, 11 pass: held draw/selection incl. case/space variants and evidence QA,
    failed roots, duplicates, held.json receipt, 7:2 views over 90k ids, snapshot boundaries vs the trainer's 5% reports
    for 34,000/33,987/1,000/40/13 steps, both AC-part guards); `bench/check_snapshot.py` (reload a snapshot → panel NLL
    by group, fidelity vs the run's panel row at the same update, decoded answers with EM/F1).

Review 1 (general-purpose reviewer, ~15:45Z): held-out logic, headline, snapshot collectives/resume and exporter guard
verified correct. Fixed: (a, high) `train()` now requires the task panel for a packet without held items (default:
`recipe.held.panel`) and refuses a panel whose `sha256`/`held_sha256` differ from what prepare excluded — before any
reference is written (a missing panel used to crash after the references and poison `fixed_references*.json`);
(b) a snapshot failure (disk/NFS/quota) is logged to `snapshot_failures.jsonl` and the panel row instead of stopping the
run, and a half-written snapshot folder is removed; (c) the held check compares question/id sets, not row counts;
(d) `--scan` also records RAG material/conditioning and multi-packing failures (with constituent ids); (e) RAG always
uses the same retrieval pool (every usable training row) with per-item passage order, so probe packets rebuild the
production items exactly. Accepted: a stop exactly at a 10% boundary delays that snapshot to the next report (its
receipt records the real step).

Validation still on real data (M3/M4): multi fraction ≈10% of examples, no nested parts, penalty tokens 0 on agent items,
`excluded.teacher_held_qa` ≈ 40 (64 held questions × 66k/100k; 0 would mean the question texts do not match).

### M3 — Build panel and training data on CSAIL

Folder `/data/scratch/ngom/activation/mixed_v1/` (staged 15:17Z): `source/` (working tree), `data` → `$R/data`,
`initial_pair/` + `initial_pair.json` (copy of the 75M pair, fresh optimizer), `env.sh`, `prep.sbatch` (CPU only, 32 CPUs,
`--mem=512G --time=03:00:00`, one forked prepare worker per CPU, a 30 s memory log `logs/prep_<job>.mem`, `MODE=scan|probe|prepare`; variables from the submitting environment, since
`sbatch --export=ALL,VAR=…` is cancelled by root on quanta), `probe.sh`, `logs/`. Pre-v2 scan: job 2363708 (15:25Z).
Steps: `panel` phase (writes `held.json`), then
`prepare --task-panel`. Gates (all must pass):

- counts receipt within ±3% of section 3 per family; multi fraction ≈10%; contextual/uncontextual ≈70/20 of examples;
- nested-part audit = 0 for training and panel items; agent items part-free and encoder-free;
- panel/reserve questions absent from RAG, trajectory QA (row ids) and evidence QA training; reserved questions absent;
- reader ≤72,000 and encoder part ≤65,536 (prepare already raises);
- prepared file sizes and per-rank RAM (both ranks unpickle `prepared_examples.pkl`; estimate 60–100 GB at ~14 bytes/token
  over ≈4.3G tokens) within the job memory; CSAIL quota headroom (1,024 GiB);
- reviewer pass on the receipts.

Archive the 75M pair to S3 (`checkpoints/reconstruction-75m-flat-20260927/`) the same way as the 6459 pair, if not done.

### M4 — Probes for time, memory and interruptions (pathological cases)

1. CPU scan (`prepare --scan`, CSAIL CPU job): build every training item/example as prepare does, recording per-item
   exceptions instead of raising; report build failures, zero-target items (NaN loss), reader >72,000, encoder part
   >65,536, nested parts, AC parts in agent items; exact per-family reader/encoder/target/penalty token totals
   (`ac_execution_records`) → hours from the step-time model; prepared bytes (pickle a sample) → the launch preflight's
   RAM demand `world × (32 GiB + 3 × max(items.pkl, prepared_examples.pkl))`; heaviest items per family → probe ids.
2. GPU stress probe (2×H200 via `launch_channel_mixture.sbatch`; packet from `prepare --subset probe_ids.json --probe`:
   heaviest first, shuffle off, checkpoint every few updates, a panel within the probe): per-rank peak memory on the
   longest agent segments / largest encoder inputs / heaviest-only updates incl. penalty forwards and next-block
   reference preparation; slowest update, rank imbalance and main-rank-only phases (initial references, panel,
   learning sample) against the 30-minute NCCL timeout (`init_process_group(timeout=30 min)`); checkpoint save time/size;
   a 10% snapshot saved, reloaded through `run_qa.harness` and evaluated; PAUSE → exact resume; SIGUSR1 (allocation end)
   → clean stop; hard kill (scancel) → resume from the last checkpoint with identical step/LR/optimizer; checkpoint and
   snapshot disk growth vs quota; storage preflight.
3. Report: projected hours (no trimming), panel time, memory to request, failed cases and fixes; delete probe folders.

#### M4 results so far (Sept 27, CSAIL quanta-h200-1, 2×H200)

GPU probe `mixed_v1/probe1` (jobs 2363768 → 2363817 → 2363828): 464 examples / 58 updates from the provisional heavy
ids (`heavy_ids.py`: per family the 40 longest usable rows by characters + 60 seeded; built in 6 min, every build valid,
multi ≈10% per family), heaviest first, checkpoint every 8, reports every 25%, production launch script.

- Interruptions, all recovered with the same packet and weights:
  - SIGUSR1 (allocation end) at step 14 → PAUSE_ALLOCATION → finished step 15, partial saved, `duration limit`,
    self-requeue (`scontrol requeue`; Slurm restarts after ~2 min) → resumed at step 16.
  - `scancel` (hard kill) during step 25 → resubmitted → resumed from the step-24 partial (steps 25+ re-run).
  - PAUSE touched at step 39 → stopped after step 41 (NFS shows the file within ~1–2 updates), exit 0, no requeue →
    PAUSE removed, resubmitted → resumed at 42.
  - SIGUSR1 during the 70% panel (step 44): the panel eval aborted (no partial row), partial saved at 44, requeued;
    signal → requeue in 40 s (the launch script leaves 900 s).
  - A planted half-written snapshot folder (no `snapshot.json`) was replaced at the 50% boundary.
  - A stop landing exactly on a report boundary (step 15 = 25%) skips that boundary's report; the resumed attempt's
    baseline panel stands in for it at the same weights, and (fix) the snapshot is taken there too.
- Main-rank-only phases vs the 30-min NCCL timeout: two initial references (256 panel items each) ≤ ~3 min together;
  held (panel) eval 87 s; learning-sample eval 76 s; snapshot save 5–9 s; checkpoint with optimizer ≈15 s.
- Panel cost (D7): ≈2.7 min per 5% panel (87 s panel + 76 s learning sample) → 20 panels + baseline ≈ 1 h over the run.
- Sizes: snapshot 2.3 GB each (9 × 2.3 ≈ 21 GB); one partial with optimizer + final pair ≈ 7 GB; probe folder 16 GB.
- Update time: heaviest updates 46/43/40/37 s (agent segments, ~60–70k reader tokens), 12–16 s mid-packet, 2–6 s at
  the tail; 705 s for 58 updates. The planning model under-predicts the heaviest updates (~33 s predicted for the first);
  `probe_fit.py` fits the model on these 58 updates → the scan's token sums give the calibrated hours.
- Peak GPU memory: 40.5 GB/141 GB on the last (light) attempt; `result.json` kept only the latest attempt, so the runner
  now appends every attempt to `attempt_results.jsonl`. Probe2 (`mixed_v1/probe2`, job 2363888: the three heaviest
  updates alone, 24 agent segments, 513k target tokens) peaks at **60.3 GB per rank of 141 GB**; steps 45.9/42.0/39.7 s
  (as in probe1), checkpoint 18 s, attempt setup outside the steps 7 s. Time outside the logged steps (next-block
  reference preparation for penalty items, batching): probe1's last attempt 38 s over 14 light updates (≈2.2 s/update
  after the 7 s setup); the 75M run 3,058 s over 4,736 updates (0.65 s/update, 16% of step time). Budget: the 75M rate,
  ≈0.65 s/update (≈6 h over ≈34k updates), added to the fitted step time.
- Update-time fit (`probe_fit.py`, job 2363886, the 58 probe1 updates, R² 0.989):
  `s/update = 0.929 + 1.001e-4·reader + 4.00e-5·encoder + 2.05e-5·target − 1.87e-5·penalty` (sums over the 8 examples);
  the planning model under-predicts by 1.35×.
- Snapshot readout (`check_snapshot.py`, 1 GPU, 5.3 min): the reloaded 50% snapshot reproduces the in-training panel NLL
  at the same update to ≤5e-4 in every group; decoded answers on 16 items/group all submitted (probe weights: RAG token
  F1 ≈0.43, trajectory ≈0.30 — mechanism check only).
- Baseline panel NLL of the 75M pair with AC parts is 1.31 vs 0.67 with no context and 0.29 with full text (the
  reconstruction-trained pair hurts Q/A through its parts); the probe's heaviest-first 464 examples bring it to 0.52.
- Real-data failure found by the scan: a RAG answer stored decomposed (`p` + U+0307) fails `material_items`' tokenizer
  round-trip assertion (NFC normalizer) and aborted the scan (it would abort prepare). Fixed: RAG answers and held-question
  keys are NFC; every RAG row is pre-checked with material_items' own rendering and unrenderable ones are excluded and
  counted (`excluded.rag_answer_render`). Scan rerun: job 2363795 (pre-v2 banks); v2 segments scanned separately with
  `SUBSET=incoming/v2_ids.json` (job 2363968, `mixed_v1/scan_v2`; `prep.sbatch` takes optional SUBSET/EXCLUDE).
- CSAIL: `sbatch --export=ALL,VAR=…` is cancelled by root; `--wrap` runs /bin/sh (no `source`).
- Preparation time (Sept 27 18:00–18:50Z): the one-process build projected to 8.5–11 h for the whole mix (build-time
  sample, array 2363990; profile: ≈0.07 s per training example, ≈0.09 s per RAG row, JSON selection ≈55 s once). It is
  now a fork pool (`prepare --workers N`, default every CPU of the allocation; `ForkPool` in `run_channel_mixture.py`):
  the parent keeps the rows and sets `BUILD_CONTEXT`, workers fork once per stage (the parent's heap frozen out of the
  cyclic collector, `gc.freeze`) and build 64-row chunks returned in order — RAG items against the one BM25 index at the
  whole batch's fetch depth, the other families' items, and the examples (each chunk returns items with their examples,
  keeping `example.item` identity). Multi-input packing measures candidate groups ahead in parallel along the
  no-rejection path (a group is fixed by its start and size; the look-ahead doubles while groups fit and restarts at one
  key per worker after a rejection) and runs the exact serial walk over the measurements. Every row is seeded by its
  identity, so where a row is built does not matter. A token budget stops at the same row; failures past it never
  count, and without `--scan` the first failing row raises its own error. Packing workers return lengths only; the
  parent builds the groups the walk takes. A worker that dies outright (OOM kill, crash) breaks the pool: finished
  results are kept, the pool is forked again from the first unyielded chunk, and after a third break the stage
  finishes in one process. Deliberate change: under `--scan` a failing
  RAG row drops only itself (the one-process build dropped every RAG row with one `<material build>` failure).
- Equivalence (CSAIL `mixed_v1/equiv`, the five 400-row samples as one 2,000-row subset, against the deployed serial
  code kept as reference): 16 workers give the same recipe (sha), selection, item order and every item and example
  (`compare.json`); the pickles are 0.25% larger (the serial build shares code constants — the compaction handoff
  prompt, the submit_answer tool definition — across all items; the parallel one keeps one copy per chunk: ≤≈1.5 KB per
  example in the worst case, so pickle digests depend on the worker count). `--workers 1` is byte-identical (same
  pickle sha256). `--scan` with agent token budgets: identical failures, probe ids and stopping rows; only the byte
  estimate differs. Build stages 174 s serial → 43 s on 16 workers (few chunks per family at this size). Memory 5.8 GB →
  ≈8 GB with 16 workers (final code; pickle +0.21%). Reviewer (four rounds): crash paths (a failing row past a token
  budget, a failing packing group off the serial walk, a worker death seen at submit), items held twice until the
  pickles, torch threads reset by `harness()`, the freed-heap blow-up below — fixed; 26 CPU checks in
  `check_mixed_run.py`; equivalence rerun after every round (`mixed_v1/equiv/run1..4`, final in `equiv/`). First
  deploy Sept 27 18:49Z (old files in `source_serial_files_20260927.tgz`); final code deployed 20:25Z.
- Full scan (job 2364179, 32 workers, 1 h 16 min, `mixed_v1/scan_full`, local copy `scans/scan_full.json`):
  **274,805 examples, 34,351 updates, 0 failures.** Per family (examples / unique sources / multi): rag_qa 86,398 /
  99,336 / 8,638; trajectory_qa 97,376 / 111,943 / 9,734; compaction 57,947 / 65,924 / 5,733 (9 capacity rejections
  kept single); agent_teacher 10,000; agent_regular 23,084. Excluded: 600 reserved RAG questions, 44 teacher held-QA,
  22 unfinished teacher runs, 332 failed regular roots. Target tokens 184.8M (agent_regular 113.0M, compaction 58.9M).
  Maxima: reader 56,542 (agent_regular), encoder part 36,035 (compaction). Hours from the fit: 71.6 h of steps +
  6.2 h outside the steps (0.65 s/update) + ≈1.5 h panels and checkpoints ≈ **79 h** (range ≈67–80 h; request
  `--time=96:00:00`). Prepared examples ≈31.1 GB; the launch preflight asks 239 GiB for 2 ranks (request 320G).
  Stage times: RAG items 479 s, other items 160 s, packing 497 s, examples 3,137 s (≈88/s; agent segments are the
  slowest).
- Memory lesson (the same scan): the item stages ran at ≈93 GB with 32 workers, but the code then released the rows
  before packing, and packing peaked at 257 GB and the example stage at 445 GB (of 512 GiB). Forked workers allocate
  into the parent's freed, fragmented heap (glibc bins, pymalloc partially free pools) and materialize a private page
  wherever an allocation lands (≈2–5 MB/s per worker); `malloc_trim` cannot release the pymalloc part. Fix: the rows
  and the pre-packing items stay alive until the example workers are gone, as in the one-process build (the scan's
  results are unaffected: content-equivalent code).
- Preparation DONE (job 2364649, 64 workers, 59 min, peak ≈318 GB of 512 GiB; `mixed_v1/run`): 274,805 examples,
  recipe identical to the scan's (recipe sha 950eb25c…, every count equal); schedule 34,351 updates, checkpoint every
  250. Stage times: RAG items 482 s, other items 243 s, packing 616 s, examples 1,549 s. `items.pkl` 23.6 GB
  (sha 7197f45e…), `prepared_examples.pkl` 31.4 GB (sha e1811163…). CSAIL scratch 96% full (2.5T free).

### M5 — Launch and monitoring (after the launch go, D9)

**Launched Sept 28 02:21Z** (user go; job 2370827, `mixed_v1/logs/run_2370827.log`): 16 h pieces that continue
straight — at each limit the trainer saves and the launch script requeues the same job; ≈5–6 pieces for ≈79 h. Submitted
with `RUN=run MEM=320G TIME=16:00:00 bash $M/probe.sh` (the M4-tested path: production launch script, 2×H200, 32 CPUs,
panel `mixed_v1/task_panel`). Heartbeat: one guarded call per 50 min (csail-slurm rule for unattended loops).

`sbatch --partition=quanta-h200 --account=quanta --qos=quanta-main --gpus=2 --cpus-per-task=32 --mem=320G
--time=96:00:00 launch_channel_mixture.sbatch` (M4: ≈79 h, preflight 239 GiB) with `CHANNEL_RUN=mixed_v1/run`, `CHANNEL_TASK_PANEL=mixed_v1/panel`
(the script sets the trainer deadline to allocation end − 15 min). quanta-h200 allows 4-day jobs; one job suffices.
30-minute heartbeat (one guarded call per cluster per cycle); panel summary at every 5% in chat and STATE; PAUSE for a
clean stop; crash → resume from the latest checkpoint (≤250 updates lost). S3 archive of the pair at 50% and final.

### M6 — Readout (TBD)

Final vs 0% panel by group; comparison with the 75M start; training NLL by group trends; greedy answers on the 128 held panel questions and, on request, from any 10% snapshot
(canon: gold NLL is not generated-answer accuracy). Continuation/live agent checks are separate follow-ups.

## 7. Decisions (accepted in the review, Sept 27 14:55Z; interactions file cleared after integration)

- D1 weights: natural (every example weight 1; `SHARES=None`).
- D2 failed roots: drop regular-agent roots ending `max_tool_errors` or `no_tool_call` (`DROPPED_ROOT_FINISH`); keep
  max_duration / max_turns / trajectory_cap.
- D3 old compaction wording: keep the rewrite to `COMPACTION_TEXT_INSTRUCTIONS` (all 1,156 affected resumes carry the
  summary after it).
- D4 schedule: fresh optimizer, 50-update warm-up to 5e-5 / 2e-5, one cosine to 10% over the pass.
- D5 held out: 64 per QA family (panel only). User asked whether option 2 differed only by the end-only rows: yes.
- D6 GPUs: 2×H200.
- D7 panel cadence: 5%; user: "depends on the total time these probes take" → M4 measures panel time and reports it.
- D8 over 72h: trim nothing, run as is. Keep usable checkpoint pairs every 5–10% (not used during the run; usable on
  request, e.g. the 50% checkpoint's generated answers) → snapshots at every 10% boundary (9 × ≈2.4 GB: ac_modules.pt
  1.2 GB + side_lora ≈0.3 GB + reader LoRA 0.9 GB; optimizer.pt 4.7 GB excluded). CSAIL scratch filesystem is 95% full
  (2.7 TB free of 50 TB); our quota 1 TiB.
- D9 launch: hold; report after the probes; wait for the go.
- D10 v1 caches: delete only `teacher_regular_v1/sync/ROLLOUTS/teacher_regular_v1_s{0,1}/rollouts.jsonl` (57.6 + 55.6 GB);
  `teacher_regular_v1/hf` (v2's HF_HOME), the bundle symlink target, draws and reports stay. ORCD job 24104566.

Superseded: the 125M reconstruction extension and the 50M non-repeating reconstruction mix (reconstruction dropped);
the shared task panel's 202/626 nested items (new panel has none); AC-part support in regular agent SFT (never).

## 8. Risks

- Prepared data size and RAM (see M3); mitigation: measure, request more memory (node has 2 TB), else per-rank shards.
- Long agent segments (max 61,820 tokens) may exceed memory; M4 includes the longest; fallback: a reader-token cap
  with the dropped count reported.
- Agent cost unmeasured; M4 fixes the estimate.
- No reconstruction monitor in the panel (user choice); compaction training NLL is the nearest proxy.
- quanta-h200 is one shared node (7 H200): at 15:20Z another user held 6 GPUs, so the 2-GPU probe and launch wait for
  that group's jobs; the shared tier (`csail-shared-h200`, 24 h max, preemptible with 10 s notice) cannot host the 68 h run.
- v2 export must land before prepare; the timeline slips with it.

## 9. Timeline (tentative)

Now → 17:19Z: M2 code + CPU checks + reviewer. ~17:30–19:30Z: M1. ~19:30Z–Sept 28 ~01:00Z: M3. ~1h: M4. Launch after
the go; ≈67–69h of wall-clock → finish ~3 days after launch.
