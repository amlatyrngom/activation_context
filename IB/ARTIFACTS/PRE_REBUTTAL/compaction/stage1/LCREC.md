# Long-context reconstruction phase for the compaction recipe (`R_cmp_lcrec`, 2026-11-16)

Question: stage 1 (KL on 20k-token agent prefixes, frozen reader) started from the shared warm-up pair `R_res_rq9fr` sat at
about 0.11 dev-75 gap closed for its first ~80 updates. `R_res_rq9fr`'s encoder only ever compressed FineWeb passages of
1,024–2,048 tokens, while the old compaction lineage had 1,000 updates of long-context reconstruction before any continuation
training. Does a reconstruction phase on the compaction prefixes themselves (frozen reader) help stage 1?

Every number below was read from the files named next to it (ORCD `cmp1/runs/<RUN>/`; S3 copies under
`s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1/`).
Gap closed = (NLL_none − NLL) / (NLL_none − NLL_full) on dev-75 (497,993 target tokens per group), with NLL_none 0.6449 and
NLL_full 0.5057 (the untuned 9B without / with the full prefix; `cmp1/data/proxy_start_pair.json`).

## Recipe

The recipe is now: warm-up pair `R_res_rq9fr` → **long-context reconstruction (`R_cmp_lcrec`)** → stage 1 KL.

- **Material `lcrec_v1`** (`build_lcrec.py`): one item per pool run of the 7 seed-0 pools in
  `IB/TMP/PRE_REBUTTAL/compaction_stage1/pools/` (six training-side pools + `pc_pool9_h6_eval`).
  - Encoder input: exactly the stage-1 AC part, `activation_part("compaction", flattened head + prefix view, tool definitions)`.
  - Reader prompt: the warm-up's reconstruction prompt (system "You are a careful assistant. You reproduce text exactly when
    asked.", user "Passage:\n" + part + "Reproduce the passage above exactly, word for word, and stop at its end.").
  - Target: the view as the encoder renders it (side chat template, inline `<think>` escaped as `[Reasoning]`), without the tool
    definitions, as plain text: role headers become `### system` / `### user` / `### assistant`, the template's empty think
    blocks and `<|im_end|>` are dropped, tool-call / tool-response tags are kept. SFT.
  - Cut: the eval's window (first non-assistant step boundary at or past 20,000 tokens, head included; prefix under 50,000).
    Nothing has to follow the cut, so trajectory-capped runs are used. Runs ending in error / task_unavailable are left out.
  - Size: r drawn per run from {1/10, 1/5} (seeded); rows = round(prefix tokens × r), prefix tokens = the reader template's
    tokens of head + prefix. 1/5 of a prefix longer than 20,480 tokens is clamped at the encoder's 4,096 rows.
  - Exclusion: the 322 eval-200 / dev-75 / frozen-100 tasks. The three index files of `build_stage1.sh` were checked equal to
    `kit/exclude_h7.json`; the pools hold 0 runs of those tasks and the output holds 0 (asserted).
  - Held reconstruction panel: 50 training-side runs (seeded), each at 1/10 and 1/5 (100 items); they never train.
- **Training `R_cmp_lcrec`** (`cmp_stage1_job.sh` with `CMP_LOSS=sft CMP_HELD=<packet>/held_panel.pkl CMP_KEEP_STEPS=all
  CMP_CKPT_EVERY=100 CMP_REPORT_EVERY=50`): from `R_res_rq9fr`, reader LR 0 (the saved reader adapter's sha256 stays
  `830ba665…`, identical to the start), encoder 5e-5, warm-up 50, cosine, batch 8, one pass = 241 updates (not capped),
  pairs kept at updates 100, 200 and the end.
- **Stage 1 after it**: `cmp_stage1_job.sh` unchanged except `CMP_INIT=<folder under cmp1/init/>` (default
  `R_res_rq9fr_final`; another start skips the u00000 proxy, the in-run panel's update-0 report covers it).
- `cmp_ckpt_proxy_job.sh RUN STEP`: the dev-75 proxy of one kept checkpoint on its own GPU while the run trains (same scorer call
  as the job's own proxy; the job then skips that checkpoint).

## Material counts (`packets/lcrec_v1/build.json`)

| | |
|---|---|
| Runs read | 3,086 (1,088 never reach 20k at a step boundary, 13 task_unavailable) |
| Training items = runs = tasks | 1,935 (1,654 training side, 281 evaluation side), 597 repositories |
| Finish of those runs | 1,257 submitted, 205 max_tool_errors, 473 trajectory cap |
| Languages | py 1,253, java 572, kt 110 |
| Sizes | 1/10: 1,008 items, 1/5: 927 (423 clamped at 4,096 rows) |
| Prefix tokens (head + prefix) | median 20,427, p90 21,605, max 30,282 (min 19,912) |
| Target tokens | median 19,780, max 29,641 |
| Reader tokens (rows + target) | median 23,431, max 33,789 |
| Updates per pass | 241 (7 items left over) |
| Held panel | 50 runs × 2 sizes = 100 items, 2,004,080 target tokens |
| Excluded tasks in the output | 0 of 322 |

## Results

### Reconstruction run `R_cmp_lcrec` (ORCD job 29263023)

241 updates, valid, stop reason complete; 97.4 s per update effective (89.7 median), peak 57.1 GB, 6.6 h wall
(`sync/R_cmp_lcrec.stats.json`). The final reader adapter has the start's sha256 (`830ba665…`).

Held reconstruction panel (`sync/R_cmp_lcrec.held.jsonl`, NLL per target token; 100 items, 1,002,040 tokens per size):

| Update | 1/10 | 1/5 | Token accuracy |
|---|---|---|---|
| 0 (`R_res_rq9fr`) | 0.5614 | 0.5952 | 0.8717 |
| 51 | 0.0309 | 0.0104 | 0.9935 |
| 101 | 0.0232 | 0.0059 | 0.9955 |
| 151 | 0.0200 | 0.0045 | 0.9962 |
| 201 | 0.0171 | 0.0035 | 0.9968 |
| 241 (end) | 0.0152 | 0.0031 | 0.9971 |

dev-75 proxy of the reconstruction checkpoints alone (`proxy/<step>.json`, teacher stored):

| Checkpoint | NLL 1/10 | NLL 1/5 | Gap closed 1/10 | Gap closed 1/5 |
|---|---|---|---|---|
| `R_res_rq9fr` (start) | 0.7135 | 0.7447 | −0.493 | −0.717 |
| step_00100 (side job 29278946) | 0.7418 | 0.7397 | −0.696 | −0.681 |
| step_00200 | 0.8009 | 0.7855 | −1.121 | −1.010 |
| final (u241) | 0.7920 | 0.7794 | −1.057 | −0.967 |

Reconstruction alone makes the continuation proxy worse: the rows now ask the frozen reader to reproduce text, while the
proxy asks it to continue the task. Stage 1 is what turns them into a continuation context.

### Stage 1 from the reconstruction checkpoint vs from `R_res_rq9fr`

Same packet `s1kl_both_c20_v1` (1,467 items, 183 updates), same recipe (KL, reader frozen, encoder 5e-5, warm-up 50, cosine,
batch 8); in-run dev-75 panel (`sync/<RUN>.held.jsonl`); the last row is also each run's own final proxy (`proxy/final.json`,
same numbers). `S1_lcrec100_kl_both_c20` (job 29280019) starts from `R_cmp_lcrec` step_00100 (staged as
`cmp1/init/R_cmp_lcrec_u100`); it was started before the reconstruction run ended, because reconstruction had converged by u51.

| Update | Baseline `S1_res_kl_both_c20` NLL 1/10 / 1/5 | gap closed | From recon u100 NLL 1/10 / 1/5 | gap closed |
|---|---|---|---|---|
| 0 | 0.7135 / 0.7447 | −0.493 / −0.717 | 0.7417 / 0.7396 | −0.696 / −0.681 |
| 41 | 0.6294 / 0.6301 | 0.111 / 0.106 | 0.5785 / 0.5654 | 0.477 / 0.571 |
| 81 | 0.6271 / 0.6274 | 0.128 / 0.125 | 0.5696 / 0.5553 | 0.541 / 0.643 |
| 121 | 0.6085 / 0.6073 | 0.261 / 0.270 | 0.5638 / 0.5495 | 0.582 / 0.685 |
| 161 | 0.5765 / 0.5638 | 0.491 / 0.582 | 0.5616 / 0.5472 | 0.598 / 0.702 |
| **183 (end)** | **0.5739 / 0.5603** | **0.510 / 0.608** | **0.5606 / 0.5460** | **0.606 / 0.710** |

KL to the full-prefix teacher at the end: 0.0627 / 0.0505 (baseline) vs 0.0501 / 0.0375 (from recon). Both runs valid and
complete; the recon-started run took 65.8 s per update median, 103.8 effective (panels included), peak 57.9 GB.

References (OLD_H8A_DEV75.md): the old H8a encoder read by the untuned frozen 9B, NLL 0.5917 / 0.5758 (gap 0.382 / 0.496);
H8a with its own trained reader LoRA, 0.886 / 0.962.

## Reading

- The long-context reconstruction phase lifts stage 1 at every matched update: +0.37 / +0.47 gap closed at u41, +0.41 / +0.52
  at u81, and +0.10 / +0.10 at the end (0.606 / 0.710 vs 0.510 / 0.608). The recon-started run passes the baseline's final at
  u41–81 and keeps improving slowly after.
- The baseline left its plateau only late (between u121 and u161); started from the reconstruction phase, stage 1 has no
  such plateau.
- With the frozen untuned reader, the new encoder now beats the old H8a encoder (0.606 / 0.710 vs 0.382 / 0.496).
- Reconstruction itself converges fast (held NLL 0.56 → 0.03 / 0.01 by u51), so the phase could be shorter; the clean recipe
  uses the full one-pass final (`cmp1/init/R_cmp_lcrec_final`), whose stage-1 behaviour is not measured here.
- One seed, one packet, the dev-75 teacher-forced proxy only (no agent runs yet).

## Where things are

- Material: `IB/TMP/PRE_REBUTTAL/compaction_stage1/packets/lcrec_v1/` = S3 `compaction/stage1/packets/lcrec_v1/` = ORCD
  `cmp1/data/packets/lcrec_v1/` (items.pkl sha256 `b45e35d0…`, held_panel.pkl `b590a815…`).
- Runs (ckpt final, held/stats/run json, proxies, logs; no optimizer or trainer folders): S3
  `compaction/stage1/runs/R_cmp_lcrec/` (also its step_00100, the start of `S1_lcrec100_kl_both_c20`) and
  `compaction/stage1/runs/S1_lcrec100_kl_both_c20/`; local mirrors under `IB/TMP/PRE_REBUTTAL/compaction_stage1/runs/`.
- Start pairs on ORCD: `cmp1/init/R_cmp_lcrec_u100`, `cmp1/init/R_cmp_lcrec_final` (hard links of the run's checkpoints + DONE).
