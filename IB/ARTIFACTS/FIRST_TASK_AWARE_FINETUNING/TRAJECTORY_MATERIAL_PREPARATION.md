# Trajectory material preparation

Private compute-node producer: [prepare_trajectory_material.py](bench/prepare_trajectory_material.py). It selects histories before questions exist. No model weights or GPU are needed; it uses the existing dataset row normalizers, Qwen tokenizer and complete-turn training helpers.

Current source SHA256: `12ddb811893a7383cae18e7069eed25232d7e655e6b45b06f25d582291954a89` (V3). V2 added CPU parallelism (`de740149…`); V3 restores the existing Nemotron AIME exclusion. The earlier serial version frozen for ORCD job 23604378 was `f76fdd328ef8ac8d4aaab21b6d3703ae78d535d7867d04f770422ffd222eb76a`. Remote submission, source ownership and job replacement remain with the root agent.

## Sources and selection

The producer seeks 50% Open-SWE `v1.2/minisweagent`, 30% English S1-DeepResearch and 20% Nemotron-Math-v4 `tir` among accepted trajectories. It streams the existing HF sources; Nemotron reads only TIR-relevant shards 7–11. If a source ends, the summary records exhaustion and the remaining sources continue. This is a trajectory mix; window and target-token proportions depend on actual lengths.

Source-qualified root IDs group SWE rollouts by `instance_id`, S1 by original question, and Nemotron by original problem. A fixed root hash assigns 10% held-out before any windows or questions. Separate counts distinguish trajectories and roots. Source metadata gold/reference patches/expected answers are never copied; recorded assistant outputs remain recorded outputs, without a correctness claim.

V3 applies the existing Nemotron loader's 13-word exclusion against AIME 2024/2025 statements before preparing a math row. Statements are loaded once, then the same exclusion windows are shared with CPU workers. A new recipe marker prevents appending to an unguarded V1/V2 bank; use a fresh guarded bank. `math_exclusion_windows()` and `excluded_math_problem()` also expose the same predicate for any necessary filtering of previously published math-task text. Focused checks reject exact and embedded fixture statements while retaining unrelated problems.

For each trajectory, choose at most three chronological prefix windows ending at complete turn boundaries, bounded by 8k/16k/32k rendered tokens. The original task/system/tools count toward each budget. Short sources and boundary undershoots retain actual lengths; identical endpoints are emitted once. An incomplete terminal call may be excluded with an explicit omitted-message count. An indivisible over-budget call is never silently cut.

QA preserves original normalized messages. Compaction additionally uses the established public-assistant continuation conversion when needed; its separate optional view and synthetic-continuation count prevent questions being generated about invented Python continuation calls. Explicit `--max-assistant-tokens 8192` is recommended: a 1024-token cap excluded all initial research/math fixture rows because some complete call arguments were larger. Every compaction suffix contains maximal complete turns fitting 8192 rendered tokens across all roles and the 72k total guard.

## Portable bank

Read `manifest.json`, then only its listed `trajectories-NNNNN.jsonl` shards. Each line stores one trajectory:

```text
schema_version, id, source, root_id, split, system_prompt, tools,
messages, initial_end, source_meta,
qa_windows: [{id, start, end, target_tokens, source_tokens}],
compaction_cuts: [{id, window_id, start, end, target_tokens, source_tokens,
                  continuation_end, continuation_tokens,
                  assistant_target_tokens, assistant_turns, full_history_tokens}],
optional compaction_messages, compaction_tools, compaction_initial_end,
compaction_synthetic_continuation_turns, omitted_incomplete_tail_messages
```

Indices are half-open. A QA history is `messages[:initial_end] + messages[start:end]`, with `system_prompt` and `tools` supplied separately. Compaction indices refer to `compaction_messages` when present, otherwise `messages`; the suffix is `[end:continuation_end]`. Questions should reference the row/window IDs rather than duplicate entire histories. Target-token totals count supervision across examples, including any repeated source spans across cuts; they are not unique source-token counts.

Shards and small index files are immutable. Atomic replacement of the manifest commits a release and its consumed source cursors. Unlisted files after interruption are not published data. Resume reads compact indices, skips only committed source rows and appends new releases. A local producer lock prevents two writers. A changed preparation recipe requires a separate output directory, but existing published JSONL remains readable and reusable independently of recipe identity.

## Launch and scaling

Use the established environment (`datasets`, `transformers`, `torch`, project dependencies), eight CPU cores and approximately 48–64 GB RAM. Set the usual persistent HF cache. Example on ORCD:

```bash
PYTHONPATH="$TRAJ_SOURCE_ROOT" \
HF_HOME="$HOME/orcd/scratch/activation/hf" \
TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
"$HOME/activation_venv/bin/python" "$TRAJ_BENCH/prepare_trajectory_material.py" \
  --output "$TRAJ_ROOT/material_v1" --limit 50000 --release-size 100 \
  --max-assistant-tokens 8192 --workers 8 --seconds 10200
```

A first run may use `--limit 300` or `1000`; increasing that value resumes toward the larger total. `--workers` can change across resumptions without changing selection. SIGUSR1/SIGTERM or the time budget publishes completed work and returns 75 when more source material remains. The batch wrapper owns automatic requeue.

Workers prefetch only a bounded number of rows per source. Results are consumed in exactly the serial selection order, and published cursors never include unfinished prefetched rows. This avoids unbounded memory and preserves source mixing. There is also per-trajectory reuse of assistant target renderings. Large-scale throughput remains to be measured on compute nodes; do not infer it from the tiny local fixture.

## Validation

Cached public fixtures and the real locally cached Qwen3.5-9B tokenizer produced seven eligible trajectories, 19 QA windows, 17 compaction examples and 41,321 supervised assistant tokens. One non-English S1 fixture was excluded. Nine checks cover source metadata without gold, original QA evidence, complete boundaries, exact lengths/targets, maximal suffixes, sibling root identity and append/resume without duplicates.

Serial and two-worker outputs match exactly apart from preparation timings, including row order, split membership, source cursors and all counts. The seven-row test took 28.3 seconds serial versus 24.6 seconds with two workers, including library startup; this is a correctness smoke and only a rough timing observation.

A two-worker run stopped after three accepted trajectories with other rows already prefetched, then resumed with one worker. Its final material and committed cursors exactly match the uninterrupted serial run. The machine-readable private receipt is `trajectory_preparation_checks/result.json` under the task's TMP folder.

A separate focused review of `trajectory_qa_item` passed six real-tokenizer cases: depths 1/2 and three gold positions. Changing the future question leaves every encoded part unchanged; full-reference evidence, distractors and ordering match; original frames/tools survive; literal answer targets round-trip; caller/sibling inputs are not mutated. The legacy `generate_trajectory_qa_samples` still chooses question-centered windows and is not the new question-blind campaign entry point.

Private checks and material: `IB/TMP/FIRST_TASK_AWARE_FINETUNING/trajectory_preparation_checks/` and `trajectory_material_*`. No product source, TASK, STATE or remote jobs were modified by this preparation subtask.

## Real GPU gate — ready to run

[probe_trajectory_teacher.py](bench/probe_trajectory_teacher.py), SHA256 `dc9efc87b2bd067990c58178946695712c2c1e616582ebbda1940e168b39bb0f`, uses the actual campaign item builder and paired 2B→9B harness. One invocation launches two separate processes: two Adam updates and a real partial checkpoint, then reload and two further updates. A short example first allocates Adam moments; the heavy history runs second, so its memory result includes optimizer residency. A separate long-suffix example is included when the material lacks one example near both 32k history and 8k continuation.

The gate checks exact teacher-weight hashes before/after reader changes and reload; freshly regenerates teacher outputs through the actual bounded preparation path; records that no-context preparation forwards with `lora_name=None`; requires finite nonzero encoder and reader gradients; and compares restored optimizer moments, penalty RNG and teacher identity. Output comparisons permit 0.002 nats numerical tolerance while weight hashes must match exactly. It validates checkpoint state semantics; exact uninterrupted-versus-resumed training equivalence remains the separate tiny CPU check.

```bash
python "$TRAJ_BENCH/probe_trajectory_teacher.py" \
  --material "$TRAJ_ROOT/material_v3" \
  --initial-pair "$TRAJ_ROOT/strong_pair.json" \
  --output "$TRAJ_ROOT/gpu_teacher_gate_v1" --seconds 4800
```

Use the campaign environment and a fresh output folder. A 96 GB or larger GPU is preferable for the initial heavy backward gate. Local checks have passed syntax, scalar/BF16 optimizer hashing, exact source-row identity and construction of four real items with source/suffix lengths 7933/2014, 30763/4499, 16156/8176 and 8091/6600. **GPU execution is still pending**; the root agent owns submission and results.
