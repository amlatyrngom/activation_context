# Mixed run plan: QA, compaction and agent SFT from the 75M pair

One training pass of about 68 hours on 2×H200. It teaches the reader to answer from activation-context (AC) inputs — retrieval QA (RAG), trajectory QA and compaction summaries — and keeps ordinary agent behavior through text-only agent supervised fine-tuning (SFT). It starts from the finished 75M reconstruction checkpoint. Your review's decisions are integrated below. Code changes and probes are under way; the launch stays on hold until you give the go after the probe report.

## Executive Summary

### Goal

Fix what reconstruction-only training broke. On the task panel, RAG and trajectory QA got worse during the 75M run: RAG contextual went from 1.099 to 1.317 nats/token, and trajectory QA multi from 2.795 to 3.452. The fix is one mixed pass over every QA and compaction row we have, plus all regular agent segments and 10k evidence-QA trajectories.

Success means the panel's RAG and trajectory QA curves fall clearly below their starting values under both views, while the agent and compaction training curves stay healthy.

### What goes in

| Family | Bank | Training rows | Examples | Split of examples | Loss share¹ | Est. hours |
|---|---:|---:|---:|---|---:|---:|
| RAG QA | 100,000 | ≈98k² | ≈85k | 70% contextual · 20% uncontextual · 10% multi | 31% | ≈16.7 |
| Trajectory QA | 112,007 | ≈111.8k | ≈97k | 70 · 20 · 10 | 36% | ≈22.2 |
| Compaction | 65,924 | ≈65.9k | ≈57k | 90% single · 10% multi | 21% | ≈16.4 |
| Regular agent SFT | ≈23k segments³ | ≈23k | ≈23k | text only, exactly what the agent saw | 8.5% | ≈7.7–9.6 |
| Evidence QA | 66,146 | 10k | 10k | text only | 3.7% | ≈1.8 |
| **Total** | | | **≈272k (≈34k updates)** | | | **≈65–67h of training** |

1. With natural weights (accepted, D1), each example counts once, whatever its length. Agent SFT carries about 58% of the target tokens but 8.5% of the loss.
2. Minus 1,900 questions the bank reserves for existing evaluations, and the held-out panel questions.
3. 12,103 old-campaign segments plus about 11k v2 segments once the running campaign finishes (6,055 so far).

Each source row is used exactly once. A multi example packs 2–3 rows, the shortest intact sources first, so rows ≈ 1.15 × examples; nothing repeats. Multi stays size-aware: a 24,576-token grouped reader limit with a 3→2→single fallback and no truncation.

Not in this run: reconstruction (the uncontextual views take its place), the recursive parent-context view, AC parts in agent SFT, and the 62 v1 segments (same tasks as v2).

### How it flows

Data flow across milestones M1–M5 (the runner is `bench/run_channel_mixture.py`: phases panel → prepare → train):

```text
ORCD teacher v2 (ends 17:19Z) ── export, text only ──► container ──► S3 + CSAIL data/agent_teacher_regular     (M1)
                                                                                       │
CSAIL data/:  RAG 100k · trajectory QA 112k · compaction 65.9k · evidence QA 66k · regular agent ≈23k
     │ seeded draw of held questions (64 or 256 per QA family)          │ everything else, once
     ▼                                                                  ▼
 panel/  256 items: RAG + trajectory QA × contextual/uncontextual     prepare → ≈272k examples       (M2–M3)
     └───────────► timing probe (M4) ─► train on 2×H200 from the 75M pair (M5) ◄──────────────┘
                     panel every 5% · checkpoint every 250 updates · PAUSE / exact resume
```

### Recipe, held-out rule and panel

- **Recipe, unchanged from canon.** SFT on gold tokens. AC encoder learning rate 5e-5, reader adapter (LoRA) 2e-5, 8 examples per update, 50-update warm-up, cosine decay to 10%, gradient clip 1.0. The no-context penalty applies with weight 1 on a random 25% of AC examples. Compression ratios are sampled in [1/20, 1/10]; the panel uses 1/16.
- **Start:** the 75M pair (2B encoder, 9B reader) with a fresh optimizer and one new schedule over the pass (accepted, D4).
- **Three settings change.**
  - Panels run every 5% instead of 10%.
  - Checkpoints are saved every 250 updates (about 33 minutes) instead of every 50. At 50, saving would cost about 4.7h over 34k updates.
  - A usable checkpoint pair is kept at every 10% boundary (D8). These snapshots are never pruned, so you can later ask how the 50% checkpoint answers.
- **Held-out rule.** The banks' own split fields are ignored. A seeded draw picks the held RAG questions and trajectory QA rows, and those never train.
  - Held RAG questions are also kept out of evidence QA, which shares the question bank.
  - Canon holds out questions, not histories. A held trajectory question's history may still train through its other questions and through compaction.
- **Panel.**
  - 64 held rows per QA family (D5), each under both views: `rag_qa/contextual`, `rag_qa/uncontextual`, `trajectory_qa/contextual`, `trajectory_qa/uncontextual`. That's 256 short items, about 3 minutes per panel.
  - The headline is the mean over the 256. Each group also shows the fixed no-context and full-context references from the starting reader.
  - Compaction and agent SFT stay visible in two ways: the report's training loss by group, and the fixed 100-item training sample that every panel re-evaluates.
- **Text-only families, checked in code.**
  - Evidence QA and regular agent items carry no AC parts. So there is no encoder call, no no-context penalty, no KL teacher and no multi-grouping. Only the reader adapter learns from them.
  - Their targets are every recorded assistant turn, reasoning and tool calls included.

### Compute and time

| | |
|---|---|
| Where | CSAIL quanta-h200: 7 H200s, 4-day job limit, no QoS cap. One job, no chaining. |
| GPUs | 2×H200 (accepted, D6) |
| Estimate | Measured in M4: ≈72h of steps + ≈6h outside the steps + ≈1.5h panels and checkpoints ≈ **79h** (range ≈67–80h); request 96h |
| If the probe projects more than 72h | Run as is, trimming nothing (D8) |
| Monitoring | 30-minute heartbeat; a panel summary every 5% (about every 3.3h) in chat and STATE |
| Interruptions | PAUSE gives a clean stop and an exact resume; a crash loses at most 250 updates |
| Tentative timeline | Code, checks and review now → v2 export after 17:19Z → CPU scan and data build → GPU probes → report → **hold** until your go (D9) → finish about 3 days after launch |

### Accepted in chat

- Option (a) counting.
- RAG and trajectory QA: whole banks at 70/20/10 contextual / uncontextual / multi.
- Compaction: whole bank. Regular agent: all segments. Evidence QA: 10k.
- No reconstruction and no recursive view.
- The banks' held-out percentages are ignored.
- The panel is limited to flat single RAG and trajectory QA, with more items per group.
- Regular agent SFT never carries AC parts; it is exactly what the agent saw.
- Up to about 72h is acceptable with regular panel updates.

## Requested Decisions

No open decisions. Accepted in your review (September 27, 14:55Z):

| # | Decision | Accepted |
|---|---|---|
| D1 | Loss weights | Natural: every example weighs 1 |
| D2 | Unsuccessful regular-agent runs | Drop runs that ended with too many tool errors or with no tool call; keep time-limited runs |
| D3 | Old compaction sentence (1,156 segments) | Keep the correction |
| D4 | Schedule | Fresh optimizer, 50-update warm-up to the canonical peak, one cosine to 10% over the pass |
| D5 | Held out | 64 per QA family, the live panel only. Your question: yes, the 256 option differed only by 192 extra rows per family, checked at the end. |
| D6 | GPUs | 2×H200 |
| D7 | Panel cadence | Every 5%. The probe measures the panel's cost and I report it, so the cadence can be revisited from that number. |
| D8 | If the probe projects more than 72h | Trim nothing; run as is. Keep a usable checkpoint pair (encoder modules plus reader adapter, no optimizer) at every 10% boundary for later generated-answer checks, e.g. how the 50% checkpoint answers. About 2.4 GB each, about 22 GB for nine. |
| D9 | Launch | Hold. Report after the probes and wait for your go. |
| D10 | v1 caches | Delete the two v1 rollout files (113 GB). The v1 model cache and container bundle stay, because v2 uses them. ORCD job 24104566. |

Superseded by this plan, with no decision needed: the 125M reconstruction extension and the 50M non-repeating reconstruction mix (reconstruction is dropped), the shared task panel's 202 nested items (the new panel has none), and AC-part support in regular agent SFT (never).

## Milestones

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M1 — Agent SFT data: export v2 and stage it</span>
    <span class="card-oneliner">The v2 runs become text-only segments on S3 and CSAIL, next to the old campaign.</span>
    <span class="card-badge">Review</span>
  </summary>

#### What landed

| | v2 export (Sept 27, 17:05Z) |
|---|---|
| Segments | 11,313 from 9,391 root runs (10,522 main, 791 subagent); 53.6M output tokens |
| Root finish reason (segments) | submitted 9,431 · max_duration 1,754 · no_tool_call 96 · max_tool_errors 32 (the last two are dropped by D2) |
| Content | 0 segments with non-text content; the exporter's AC-part assertion passed; 0 duplicate segment ids |
| Releases | `teacher_regular_v2_20260927_s0` … `_s3`, one per shard (runs 2,500 / 2,404 / 2,426 / 2,071; shards 1–3 stopped at the campaign end) |
| S3 | uploaded 17:11Z; the collection manifest now lists 7 releases (the 3 earlier ones kept, v2 appended) |
| CSAIL | staged next to `campaign_12h_20260911`; the tar and every shard checksum verified; manifest updated atomically |
| Receipt | `AGENT_REGULAR_V2_EXPORT.json` |

#### Drifts, challenges, and unplanned steps

- **The token check changed shape.** Re-rendering the messages with a chat template cannot reproduce the recorded tokens, and the mismatch would be by design: training writes earlier reasoning as `[Reasoning]` text so the template cannot strip it. The new check, `activation/bench/canonical_training/check_exported_segments.py`, asks the question that matters instead. Does every text an exported message carries (prompt, tool output, assistant text, reasoning, string call arguments) appear verbatim in the tokens the teacher read or sampled at that point? This catches a full tool output where the agent saw a truncated one, a dropped or shifted message, or a rewrite.
- The sample is 25 segments per benchmark and release (1,000 for v2), not 200 per source.
- The old campaign was checked too (250 segments).
- The S3 collection manifest was merged by hand, because the exporter's own manifest lists only the v2 releases.

#### Validation

- Old campaign: 250 segments, 12.6M characters, **0 misses**. Negative controls on 5 segments were all flagged: a tool output extended, the prompt changed, a message dropped, assistant text extended.
- v2 (ORCD job 24112783): 1,000 segments over all 10 benchmarks, 48.4M characters, **0 misses**.

#### Planning Overview

After the v2 campaign stops (17:19Z), one ORCD CPU job reads the four shard caches on a compute node and exports them with the canonical exporter. The output is then copied to the container in one guarded transfer, uploaded from there to `s3://general-bucket-aln/activation-context-data/datasets/agent_teacher_regular`, and staged on CSAIL next to `campaign_12h_20260911`. v1 is not staged.

**What the agent saw.** I earlier reported that about 40% of v2 rows have flat AC parts. That came from a loose row check that counted any record holding messages plus a kind. The v2 teacher ran without an AC model.
- A long tool output keeps an unrendered description of itself inside the step's `tool_results`, but the agent was shown the truncated text.
- Compaction resumes are plain text.
- The exporter takes the recorded prompt and every step's messages, never `tool_results`, so it already produces exactly what the agent saw.
- A hard assertion makes this explicit: any AC part in an exported message fails the export.

**Checks:**
- segment count and zero duplicate segment ids;
- the assertion passes;
- a sampled token check on 200 segments per source: the teacher tokenizer renders the exported messages back to the recorded token ids. Any mismatch rate is reported, and a systematic one is investigated.

#### Planned Changes

`activation/bench/canonical_training/export_agent_teacher_segments.py · segment_rows()`

```diff-python
@@ segment_rows() @@
     for path, depth, segment in segments(row):
         messages = [dict(message) for message in segment["prompt_messages"]]
         target_start = len(messages)
         for step in segment["trajectory"]:
             messages.extend(step["messages"])
+        if any(is_ac_part(item) for message in messages if isinstance(message.get("content"), list) for item in message["content"]):
+            raise ValueError(f"{row['config_key']} {path}: an AC part in what the agent saw")   # the teachers ran without an AC model
         if not any(message.get("role") == "assistant" for message in messages[target_start:]):
             continue
⋯ unchanged lines omitted ⋯
```

`IB/TMP/FIRST_TASK_AWARE_FINETUNING/teacher_regular_v2/export.sbatch` (new)

```bash
#SBATCH --job-name=teacher_regular2_export --partition=mit_normal --cpus-per-task=8 --mem=64G --time=03:00:00
source /home/ngom/orcd/pool/activation/teacher_regular_v2/env.sh
args=()
for s in 0 1 2 3; do args+=(--cache $R/sync/ROLLOUTS/teacher_regular_v2_s$s/rollouts.jsonl --release regular_v2_s${s}_$STAMP); done
$PY -m activation.bench.canonical_training.export_agent_teacher_segments "${args[@]}" \
  --tokenizer Qwen/Qwen3.8-27B-FP8 --model-id Qwen/Qwen3.8-27B-FP8 --output $R/export/agent_teacher_regular
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M2 — Runner changes for this mix</span>
    <span class="card-oneliner">run_channel_mixture.py drops reconstruction and the recursive view, and applies the new held-out rule, views, panel, settings and 10% snapshots.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Planning Overview

The changes are all in `bench/run_channel_mixture.py`, plus the exporter assertion from M1:

1. Families without reconstruction. Weights per D1; natural means every example weighs 1.
2. Two Q/A views only. Among singles, 7 of 9 are contextual and 2 of 9 uncontextual. After 10% become multi groups, that gives 70/20/10 of examples. Multi constituents keep their own drawn view.
3. The held-out rule, with one source of truth. The panel phase draws the held identities and writes `held.json`, and prepare excludes exactly those. RAG held questions are also excluded from evidence QA, and reserved questions stay excluded everywhere. Trajectory QA is held by row id. The old "trajectory root crosses train/held" check goes: canon holds out questions, not histories.
4. Regular agent: assert that there is no AC part, and drop failed roots per D2. The old-wording rewrite is kept or removed per D3.
5. The panel builder keeps only the four flat single QA groups at 64 rows each, and writes the drawn identities to `held.json`.
6. Prepare builds no separate held set; the headline is the panel. Settings: `reporting_interval=.05`, `checkpoint_every_updates=250`. The recipe records the held identities, the exclusions and every D-choice.
7. CLI defaults for this run: whole banks, 10k evidence QA, no target-token caps, `--per-group 64`. No reserve (D5).
8. **10% snapshots (D8).** At every 10% panel boundary, the main rank saves the encoder modules and the reader adapter to `SNAPSHOTS/epoch_001_<percent>pct/{ac,reader}` with a `snapshot.json` that records the update. A failed snapshot is logged and training continues. There is no optimizer state, and the trainer's pruning never touches that folder.
9. Probe support for M4: `prepare --scan`, `--subset` and `--probe`. The reconstruction runner's shared packer keeps the full family list, so its behavior does not change.

**Validation:** CPU checks on small fixtures:
- the view shares over 9,000 seeded ids come out at about 7:2;
- held questions are absent from every training family, including evidence QA;
- there are no nested parts, and agent items carry no parts;
- the multi fraction is about 10% of examples;
- the wording and failed-root rules behave as decided.

Then a high-end reviewer checks the diff before M3.

#### Planned Changes

`IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/bench/run_channel_mixture.py · FAMILIES, SHARES, MODES, conditioning()`

```diff-python
@@ module constants @@
-FAMILIES=('rag_qa','trajectory_qa','compaction','agent_teacher','agent_regular','reconstruction')
-# Provisional until the mixed run is approved: agent_regular (unrolled regular-task teacher segments) joined after the gate.
-SHARES=dict(zip(FAMILIES,(.225,.225,.2,.1,.15,.10)))
+FAMILIES=('rag_qa','trajectory_qa','compaction','agent_teacher','agent_regular')   # no reconstruction: the uncontextual views replace it
+SHARES=None                                               # D1 natural mix: every example weighs 1; a dict restores explicit shares
+DROPPED_ROOT_FINISH=('max_tool_errors','no_tool_call')   # D2: regular-agent runs whose segments do not train
⋯ unchanged lines omitted ⋯
-MODES={'question':'contextual','parent':'recursive','summary':'uncontextual'}
+MODES={'question':'contextual','summary':'uncontextual'}  # depth 1: no recursive parent view
⋯ unchanged lines omitted ⋯
 def conditioning(ident):
- value=int(digest((SEED,'view',ident))[:8],16)%10
- return 'question' if value<4 else 'parent' if value<8 else 'summary'
+ # 70/20 contextual/uncontextual of all examples once 10% become multi-input groups: 7 of 9 singles see the question.
+ value=int(digest((SEED,'view',ident))[:8],16)%9
+ return 'question' if value<7 else 'summary'
```

`bench/run_channel_mixture.py · row_split(), select_rows()`

```diff-python
@@ row_split() @@
-def row_split(family,row,reserved):
- """(split, None) for a usable row; (None, reason) for an excluded one."""
- if family=='rag_qa':split='held' if normalized(row['question']) in reserved or question_held(row['question']) else 'train'
+def row_split(family,row,reserved,held):
+ """(split, None) for a usable row; (None, reason) for an excluded one. The banks' own split fields and hash rules are
+ ignored: 'held' is exactly the drawn panel identities (held.json); everything else trains."""
+ if family=='rag_qa':
+  question=normalized(row['question'])
+  if question in reserved:return None,'rag_reserved_qa'
+  split='held' if question in held['rag_qa'] else 'train'
  elif family=='agent_teacher':
   if normalized(row['task']['question']) in reserved:return None,'teacher_reserved_qa'
+  if normalized(row['task']['question']) in held['rag_qa']:return None,'teacher_held_qa'   # a held RAG question never trains through evidence QA
   if not row['quality'].get('submitted') or row['run'].get('finish_reason')!='submitted':return None,'teacher_unfinished'
-  # No lexical-F1 filter: correct paraphrases are deliberately retained.
-  split='held' if question_held(row['task']['question']) else 'train'
- # agent_regular: every segment, finished or not; the export's split is by root task, so every segment of a task shares it.
- else:split=row['split']
- if split not in ('train','held'):raise ValueError('Unassigned source split: '+family)
+  split='train'                                          # no lexical-F1 filter: correct paraphrases are deliberately retained
+ elif family=='trajectory_qa':split='held' if row['id'] in held['trajectory_qa'] else 'train'
+ elif family=='agent_regular' and row['root']['finish_reason'] in DROPPED_ROOT_FINISH:return None,'regular_failed_root'
+ else:split='train'                                       # compaction, agent_regular: every row trains (the export's root split is ignored)
  return split,None
@@ select_rows() @@
-   if family in ('trajectory_qa','compaction'):
-    provenance=row['provenance'];root=provenance.get('generation',provenance)['root_id']
-    if root_splits.setdefault(root,split)!=split:raise ValueError('Trajectory root crosses train/held: '+root)
+   # Questions are held out, not histories (canon): a held question's history may train through other questions and compaction.
⋯ unchanged lines omitted ⋯
-  if not chosen['train'] or not chosen['held']:raise ValueError('No available train/held rows: '+family)
+  if not chosen['train'] or (family in ('rag_qa','trajectory_qa') and not chosen['held']):raise ValueError('No available train/held rows: '+family)
```

`bench/run_channel_mixture.py · make_item() — agent_regular`

```diff-python
@@ make_item() @@
  elif family=='agent_regular':
-  # One unrolled teacher segment (main run, compacted segment or subagent), text only; targets are its recorded turns.
-  from activation.common.ac_parts import COMPACTION_TEXT_INSTRUCTIONS
+  # One unrolled teacher segment exactly as the agent saw it (recorded prompt + step messages), text only; targets are its recorded turns.
+  from activation.common.ac_parts import COMPACTION_TEXT_INSTRUCTIONS,is_ac_part
   tools=deepcopy(row['tools']) or None;kwargs={'enable_thinking':True}
   messages=deepcopy(row['messages']);start=row['target_start']
+  if any(is_ac_part(x) for m in messages if isinstance(m.get('content'),list) for x in m['content']):
+   raise ValueError('Regular agent segment carries an AC part: '+ident)   # never: the teachers ran without an AC model
   # The 12h campaign's text-only resumes still said "compacted into the activation context above": use the text-mode wording.
⋯ the rewrite stays (D3 recommended) or is removed (D3 verbatim) ⋯
```

`bench/run_channel_mixture.py · build_panel()`

```diff-python
@@ build_panel() — groups @@
- rows=held['rag_qa'][:args.multi_pool]
+ rows=held['rag_qa'][:args.per_group]                     # held = the seeded draw written to held.json
 ⋯ RAG material and matched BM25 distractors unchanged ⋯
 for mode,name in MODES.items():add('rag_qa/'+name,[condition_rag(deepcopy(item),row,'held',mode) for item,row in zip(base[:args.per_group],rows)])
- add('rag_qa/multi',multi_groups(study,[condition_rag(deepcopy(item),row,'held') for item,row in zip(base,rows)],args.per_group,24_576,builder))
- rows=held['trajectory_qa'][:args.multi_pool]
+ rows=held['trajectory_qa'][:args.per_group]
 for mode,name in MODES.items():add('trajectory_qa/'+name,[tagged('trajectory_qa',row,make_item(study,'trajectory_qa',row,'held',mode=mode)) for row in rows[:args.per_group]])
- add('trajectory_qa/multi',...)
- ⋯ compaction/single, compaction/multi, reconstruction/{flat,mixed,transport,balanced,multi}, agent/<kind>, agent/evidence_qa removed ⋯
+ # Fundamentals only: flat single RAG and trajectory Q/A under both views; the drawn identities are the run's only held-out rows.
+ write(out/'held.json',{'rag_qa':{'ids':⋯,'questions':⋯},'trajectory_qa':{'ids':⋯}})
```

`bench/run_channel_mixture.py · prepare(), train()`

```diff-python
@@ prepare() @@
- for split,items in [('train',rag_train),('held',rag_held)]:
+ for split,items in [('train',rag_train)]:               # held questions live in the task panel only: the panel is the headline
⋯ unchanged lines omitted ⋯
- for item in output['train']:item.weight=SHARES[item.info['family']]*total/actual[item.info['family']]
+ for item in output['train']:item.weight=SHARES[item.info['family']]*total/actual[item.info['family']] if SHARES else 1.0
  config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,learning_rate_target_lora=2e-5,examples_per_update=8,⋯
-  schedule_total_updates=math.ceil(total/8)*args.epochs,checkpoint_every_updates=50,reporting_interval=.1)
+  schedule_total_updates=math.ceil(total/8)*args.epochs,checkpoint_every_updates=250,reporting_interval=.05)
@@ train() — reference() and panel() @@
+ panel_headline=not [i for i in items['held'] if i.item_id not in panel_ids]   # no prepared held set: the task panel is the headline
-    summary=trainer.eval(AC,view);core=[r for r in summary['per_item'] if base_id(r['item_id']) not in panel_ids]
+    summary=trainer.eval(AC,view);core=[r for r in summary['per_item'] if panel_headline or base_id(r['item_id']) not in panel_ids]
⋯ the same condition in panel()'s headline ⋯
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M3 — Build the panel and the training data on CSAIL</span>
    <span class="card-oneliner">One reviewed source tree; the panel, then the ≈272k prepared examples, each gated before any GPU training.</span>
    <span class="card-badge">Planning</span>
  </summary>

#### Planning Overview

A new folder, `/data/scratch/ngom/activation/mixed_v1/`, holds:
- the reviewed source tree with manifest hashes;
- `data/`, which links to the staged collections after M1;
- `initial_pair.json`, which points to the 75M pair (`run_75m_cont/AC_MODELS/rag_poc/epoch_001` and `LORAS/rag_reader/epoch_001`).

The `panel` phase runs first and writes the drawn held identities to `held.json`; `prepare` then excludes exactly those. Both run as one CPU-only job, in the same pattern as the 75M preparation.

**Gates, all must pass before M4:**
- The counts receipt is within ±3% of the table per family, with the multi fraction at about 10% and contextual/uncontextual at about 70/20 of examples.
- No training or panel item has an AC part nested inside another (depth-1 rule). Agent items have no parts and no encoder work.
- Held questions and reserved questions are absent from RAG, trajectory QA and evidence QA training.
- Every example fits the reader limit (72,000 tokens) and the encoder part limit (65,536); prepare already raises otherwise.
- **Size and memory.** Both training ranks load the whole prepared file. At about 14 bytes per token over about 4.3G tokens, that could be 60–100 GB. Measure it and set the job memory; the node has 2 TB. It must also fit the CSAIL quota of 1,024 GiB. If it is too large, split it per rank. That would be a code change, reported first.
- A reviewer pass on the receipts.

The 75M pair is also archived to S3 the same way as the 6459 pair, if not already done.

#### Planned Changes

`/data/scratch/ngom/activation/mixed_v1/prep.sbatch` (new, CSAIL)

```bash
#SBATCH --job-name=mixed_prep --partition=quanta-h200 --account=quanta --qos=quanta-main --cpus-per-task=16 --mem=512G --time=12:00:00
# CPU only (no GPU requested), like the 75M preparation; the runner's preparation is a separate CPU command.
source /data/scratch/ngom/activation/mixed_v1/env.sh
$PY $B/run_channel_mixture.py panel   --output $M/panel --data $M/data --initial-pair $M/initial_pair.json --per-group 64
$PY $B/run_channel_mixture.py prepare --output $M/run   --data $M/data --initial-pair $M/initial_pair.json --task-panel $M/panel \
  --teacher-histories 10000 --multi-target-fraction 0.1
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M4 — Probes for time, memory and interruptions</span>
    <span class="card-oneliner">A CPU scan of every item, then GPU stress and interruption probes on the heaviest real examples, before any launch.</span>
    <span class="card-badge">Implementing</span>
  </summary>

#### Results so far

GPU probes ran on CSAIL quanta-h200-1 (2×H200) through the production launch script. probe1: 464 real examples (58 updates), the heaviest first, with a checkpoint every 8 updates and a panel every 25%. probe2: the three heaviest updates alone (24 agent segments).

**Interruptions: every case recovered with the same packet and weights.**

| Case | What happened |
|---|---|
| Allocation-end signal (SIGUSR1) at step 14 | Finished step 15, saved, requeued itself; Slurm restarted it ~2 min later and it resumed at 16 |
| Hard kill (`scancel`) during step 25 | Resubmitted; resumed from the step-24 checkpoint |
| PAUSE at step 39 | Stopped after step 41 with exit 0 and no requeue; resumed at 42 once PAUSE was removed |
| SIGUSR1 during a panel | Panel aborted, saved, requeued within 40 s (the script leaves 15 min) |
| Half-written snapshot folder | Replaced at the next boundary |
| Stop exactly on a report boundary | **Bug, fixed:** that boundary's snapshot was skipped. The resumed run's baseline panel now takes it, at the same weights |

**Memory and time.**

| Measure | Result |
|---|---|
| Peak GPU memory, heaviest updates | **60.3 GB per rank of 141 GB** |
| Heaviest update | 46 s (agent segments, 60–70k reader tokens each) |
| One-rank phases vs the 30-min NCCL timeout | Initial references ≤3 min; panel 87 s; learning sample 76 s |
| Panel every 5% (D7) | ≈2.7 min each → **≈1 h over the run** |
| Checkpoint (with optimizer) / snapshot | 15–18 s / 5–9 s, 2.3 GB per snapshot |
| Snapshot usable (D8) | A 50% snapshot reloaded in a fresh process matches the in-training panel within 5e-4 and produces submitted answers |
| Step-time model | Fitted on the 58 real updates (R² 0.989); the old planning model was 1.35× too low |

**Real-data failure found by the scan, fixed:** one RAG answer stored in decomposed Unicode (`p` + combining dot) failed the tokenizer round trip and would have aborted preparation. RAG answers and held-question keys are now NFC, and unrenderable rows are pre-checked and excluded with a count.

**Preparation is now parallel.** Built in one process, preparing the whole mix would have taken 8.5–11 hours. It now forks one worker per CPU and builds 64-row chunks. Packing measures candidate groups ahead in parallel, then takes exactly the same groups as the one-process walk.

| Check (2,000-row mixed sample, against the old one-process code) | Result |
|---|---|
| 16 workers | Same recipe, selection, order and every item and example; pickle 0.25% larger (shared constants kept once per chunk) |
| 1 worker | Byte-identical pickles |
| Scan with agent token budgets | Same failures, probe rows and stopping rows |
| Build stages | 174 s → 43 s on 16 workers |
| Review | Four rounds: crash paths, a double-held item list, a thread setting and the memory blow-up below fixed; 26 CPU checks |

One deliberate change: in a scan, a failing RAG row now drops only itself instead of every RAG row.

**Full scan: 274,805 examples, 34,351 updates, no failures** (1 h 16 min on 32 CPUs).

| Measure | Result |
|---|---|
| Examples (unique sources) | RAG 86,398 (99,336), trajectory QA 97,376 (111,943), compaction 57,947 (65,924), teacher agent 10,000, regular agent 23,084 |
| Multi-input share | 10.0% per Q/A family; 9 compaction groups too long, kept single |
| Target tokens | 184.8M (regular agent 113.0M, compaction 58.9M) |
| Longest reader / encoder part | 56,542 / 36,035 tokens (limits 72,000 / 65,536) |
| Hours from the fitted step time | **≈79 h** (range ≈67–80 h); request 96 h |
| Launch memory | Prepared examples ≈31 GB; preflight asks 239 GiB for 2 ranks |

**Memory lesson from the scan.** Freeing the source rows before the later stages pushed the scan to 445 GB of its 512 GiB: forked workers allocate into the freed, fragmented heap and each touched page becomes a private copy. The item stages, with the rows alive, used about 93 GB. The rows now stay alive until the last workers exit, as in the one-process build.

**Preparation done** (59 min on 64 CPUs, peak ≈318 GB, `mixed_v1/run`): 274,805 examples with the same recipe as the scan, every count equal. `prepared_examples.pkl` is 31.4 GB and `items.pkl` 23.6 GB.

#### Planning Overview

The probes look for anything that could stop the ~70-hour run partway, and they pin the time estimate.

**1. CPU scan of the whole mix** (`prepare --scan`, no GPU). It builds every training item and example exactly as the real preparation does, but a failing row is recorded instead of aborting. So one bad row cannot kill an 8-hour preparation, or later the run. It reports:
- rows that fail to build;
- items with no target tokens, which would give a not-a-number loss;
- readers over 72,000 tokens and encoder parts over 65,536;
- nested parts, and AC parts in agent items;
- exact per-family token totals, which give an hours estimate from the real data rather than the fixture;
- the prepared-file size, and the host memory the launch check will then demand (2 × (32 GiB + 3 × file size));
- the heaviest items per family, which become the GPU probe's selection.

**2. GPU stress probe** (2×H200, a probe packet of the heaviest items first plus a seeded sample, same code path). It covers:
- **Worst-case memory:** the longest agent segments (up to about 62k reader tokens); the largest compaction, trajectory QA and RAG encoder inputs; the updates made only of the heaviest items, with their no-context penalty forwards and the next block's reference preparation. Peak memory is recorded per rank.
- **Worst-case time against the 30-minute collective (NCCL) timeout:** the slowest update and the imbalance between ranks; also the phases where one rank works while the other waits (the initial references, each panel, and the fixed learning sample).
- **Checkpoints:** save time and size. A 10% snapshot is saved, then reloaded into a fresh harness and used to evaluate, which proves it is usable.
- **Interruptions:**
  - PAUSE → clean stop at the exact step → exact resume (same step, learning rate and optimizer state);
  - the allocation-end signal the launch script gets 15 minutes before the job's end (SIGUSR1) → clean stop;
  - a hard kill → resume from the last checkpoint.
- **Disk:** checkpoint and snapshot growth against the quota, plus the storage preflight.

**3. Report:** projected hours (no trimming, D8), panel time (D7), the job memory to request, and any case that failed together with its fix. The launch stays on hold (D9). Probe folders are deleted afterwards.

#### Planned Changes

`bench/run_channel_mixture.py · prepare(--scan / --subset / --probe)`: `--scan` records per-item failures and sizes and writes `scan.json` instead of pickles. `--subset ids.json` builds only the listed training rows. `--probe` orders items heaviest first, turns shuffling off and checkpoints every few updates. All other code is shared with the real preparation. A small driver in `IB/TMP/FIRST_TASK_AWARE_FINETUNING/mixed_v1_probe/` runs the stress, snapshot and interruption steps through the real launch script.

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M5 — Launch and monitoring</span>
    <span class="card-oneliner">One ~96-hour allocation on quanta, a 30-minute heartbeat and a panel summary every 5%.</span>
    <span class="card-badge">Implementing</span>
  </summary>

**Launched Sept 28 02:21Z** on your go: job 2370827 runs 16-hour pieces that continue straight. At each limit it saves and requeues itself, about 5–6 pieces for the ≈79 hours. The heartbeat makes one guarded call every 50 minutes.

#### Planning Overview

This starts only after the launch go (D9). One job requests `--time=96:00:00` and `--mem=320G` (M4: ≈79 h, preflight 239 GiB); the launch script sets the trainer deadline to the allocation end minus 15 minutes.

**Monitoring:**
- A 30-minute heartbeat makes one guarded call per cluster per cycle.
- At every 5% boundary, a panel summary goes to chat and to STATE: headline, the four groups against their no-context and full-context references, and training loss by group.

**Stops and recovery:**
- PAUSE gives a clean stop at the exact step, and the exact resume is the tested path.
- After a crash, training resumes from the latest checkpoint, losing at most 250 updates.

The pair is archived to S3 at 50% and at the end.

#### Planned Changes

`/data/scratch/ngom/activation/mixed_v1/train.sh` (new, CSAIL)

```bash
source /data/scratch/ngom/activation/mixed_v1/env.sh
export CHANNEL_SOURCE=$M/source CHANNEL_PYTHON=$PY CHANNEL_RUN=$M/run CHANNEL_TASK_PANEL=$M/panel \
  CHANNEL_QUOTA_ROOT=/data/scratch/ngom CHANNEL_QUOTA_GIB=1024
sbatch --job-name=mixed_v1 --partition=quanta-h200 --account=quanta --qos=quanta-main --gpus=2 --cpus-per-task=16 \
  --mem=320G --time=96:00:00 --output=$M/logs/train_%j.log --export=ALL $M/source/⋯/launch_channel_mixture.sbatch
```

</details>

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">M6 — Readout</span>
    <span class="card-oneliner">What the pass achieved, per family, against its starting point.</span>
    <span class="card-badge">TBD</span>
  </summary>

#### Planning Overview

This comes after the run finishes. It covers:
- the final panel against the 0% panel by group, compared with the 75M start;
- training-loss trends by group for compaction and agent SFT;
- generated answers on the 128 held panel questions, and on request from any 10% snapshot (gold-token NLL is not answer accuracy).

Continuation quality and live agent checks are separate follow-ups.

</details>

## Risks

- **Prepared data size and memory.** 60–100 GB could be loaded per rank. M3 measures it; the fixes are more memory or per-rank shards.
- **Long agent segments.** The longest is 61,820 tokens and could run out of memory. M4 includes the longest ones; the fallback is a reported length cap.
- **Agent cost has never been measured.** The estimate for them is 1.2–1.5 s per example; M4 replaces it with a measurement.
- **No reconstruction monitor in the panel.** That was your choice; compaction training loss is the nearest proxy.
- **Shared node.** quanta-h200 has 7 H200s and other users run there; this plan requests 2.
- **Order dependency.** The v2 export must land before prepare, and the timeline slips with it.
