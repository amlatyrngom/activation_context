# State: after submission, before the rebuttal (2026-11-09)

The paper was submitted to MLSys 2027 on 2026-10-30 (`paper/` at `16a0260`). On 2026-11-09 work moved
from the VS Code agent container to the EC2 host and the project was consolidated.

- **History:** the full Oct 14–27 log is `IB/ARTIFACTS/HANDOFF_EC2/STATE_archive_20261109.md`
  (verbatim, 400 KB); what was worth keeping from it is `IB/ARTIFACTS/HANDOFF_EC2/STATE_DIGEST_20261109.md`.
  Older states: `IB/TMP/PAPER_COMPACTION/STATE_before_condense_20261014T1415Z.md` and
  `IB/TMP/FIRST_TASK_AWARE_FINETUNING/STATE_before_paper_mvp_20261013.md` (both also on S3, see below).
- **Canon:** `IB/CANON/ROOT_CANON.md` (guardrails, how the user works, cluster lessons),
  `IB/CANON/PAPER_CANON.md` (every figure/table → script → data → numbers, and the settings that differ
  between arms), plus the activation-context and agent-training canon.
- **The move:** `IB/ARTIFACTS/HANDOFF_EC2/` — `HANDOFF_EC2.md` is at `IB/HANDOFF_EC2.md`, the
  consolidation plan is `PLAN.md` (+ `PLAN.html`), the S3 index is `S3_INDEX.md`.

## Direction (user, 2026-11-09)

Pre-rebuttal work, in this order:

1. **Text + reader-LoRA baseline on the main workloads** (retrieval, compaction, multi-agent, corpus),
   not only in the §3.5 ablations. In §3.5 it beat every AC run (peak 72.5, final 70.5 vs 68.5), so do
   it carefully: same held-out set rules, same penalties, same caps as the AC arms.
2. **Standardise training and hyperparameters.** Every workload used a different, partly ad hoc
   recipe (rates 2e-5 to 1e-3, cosine vs constant + cooldown, stacked stages, selection on eval sets,
   one-seed results). Write one documented recipe per workload family and rerun runs whose settings
   were missing or undocumented.
3. **Fairer baselines.** Same decoding settings for all arms (retrieval: text/select had presence
   penalty 1.5, AC none), same generation caps (§3.5: AC 128 vs text 2,048), one answer parser per
   workload, trained text baselines where AC has a trained reader.
4. **Rerun candidate:** `S_kv_v4fr` (All-Layer KVs, frozen reader, no reconstruction) reached 71.5 on
   §3.5, above every reported number (one seed, 200 questions).

The user said minor caption and wording issues (Fig 5 difficulty definition, compaction timing
provenance, the "No Context" arm intro) can be ignored. The full list of gaps with sources is in
`PAPER_CANON.md`, sections "Settings that differ between arms or are undocumented" and "Also from the
October log". Reviews arrive 2027-01-18; responses are due 2027-01-21.

**Green-lit 2026-11-09 ≈21:30Z; plan: `IB/ARTIFACTS/PRE_REBUTTAL/PLAN.md`** (priority order: §3.5 KV vs text+LoRA,
then §3.5 in/res with least-squares init; retrieval 2B→9B text+LoRA and one recipe; uncapped compaction summary
rerun on any free GPU; math last). Hill-climbing is allowed as long as the final recipe is clean and explainable.
Corpus compression stays as submitted (user correction: "compaction needs no further work" meant corpus).

## Setup (2026-11-09)

| What | Where / how |
|---|---|
| Workstation | EC2 `i-0918b8f73e9a0fe03` (t3.2xlarge, 8 CPU, 30 GB RAM, **no GPU**), repo `/home/ubuntu/activation_context`. Never stop or terminate it. |
| Git | One branch, `main` (= the paper code + container patch + IB, merged over the old main). Old main = tag `archive/main-20260921`. Agents commit and push at milestones. |
| Paper | `paper/` submodule → github.com/amlatyrngom/activation_context_paper (git-synced with Overleaf; pull first). Build: `IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh [--pages]` (TinyTeX in `~/.TinyTeX`). Submitted build: body 10 pages (references from page 11), 15 pages total; appendix 3 pages. |
| Python | `.venv` from `uv.lock`; CPU tests: `.venv/bin/python -m pytest activation/tests` (GPU tests need `--gpu`). |
| GPUs | SkyPilot/EC2 (`.venv/bin/sky`), CSAIL, ORCD. No cluster or cloud job of ours is running. |
| Clusters | `IB/CANON/CANON_ARTIFACTS/cluster_access/{csail,orcd}_ssh.sh` over the shared connections the user opens with Duo (`ssh csail echo ok`, `ssh orcd echo ok`; up to 7 days). **CSAIL: our `quanta-main` QoS preempts `shared-if-available`/`lab-free` jobs on `quanta-h200-1`; submit there even when it looks full (ROOT_CANON).** |
| S3 | `s3://general-bucket-aln/activation-context-data/` (`aws` CLI installed via uv). See `S3_INDEX.md` for what moved there on 2026-11-09. |
| Credentials | `~/.aws`, `~/.ssh`, `.env` (with `HF_TOKEN`, `GITHUB_TOKEN`), `gh` login. Never commit or print them. |

## Consolidation (2026-11-09) — see `IB/ARTIFACTS/HANDOFF_EC2/PLAN.md`

- Done: SSH and cluster access; held CSAIL job 2741502 cancelled (ORCD had none); AWS CLI; one `main`
  pushed (remote branches `paper-mvp-20261013` and `ac-recon-night-20260917` deleted, both inside
  `main`); paper submodule and TeX build; STATE archived and condensed; canon updated.
- Also done: old container's unique data and all of `IB/TMP` archived to S3 and verified file by file
  (`IB/ARTIFACTS/HANDOFF_EC2/S3_INDEX.md`); container removed; pre-paper folders pruned; disk 402 → 248 GB used.

## Open items inherited from October (never settled; decide when relevant)

- Fig 7c framing now that text + LoRA ≥ AC in §3.5 (paper currently shows "Text (+ LoRA)").
- Paper trim candidates for a camera-ready: §2.5 Engine Implementation; the select → text baseline
  in §3.1 ("KEEP for now, but a good candidate for removal"); the §4.5 decode-baseline bullets.
- Strict vs lenient answer parsing per workload (recommended lenient in October; part of item 3 above).
- Known failing CPU test: `test_basic_dataset_loading.py::test_trajectory_dataset_loading` — a streamed Open-SWE
  trajectory loses message text when chunked (content check at line 229). Fails on the old main too (2026-11-09).
- Residual every-4th vs every layer for the Input + Residuals arm; unused v2 corpus data (v2s2 19,916
  items; v2s3 raw on ORCD only).

## Rules

The user often dictates: expect garbled words and read them sensibly (examples in ROOT_CANON "Working with the user").

The guardrails, autonomy rule and the user's working preferences are in `ROOT_CANON.md`
("Autonomy", "Machines, access and guardrails", "Working with the user"). In short: don't block on the
user except for important things; never start Duo; never stop this host; print main result tables in
chat; never state a number you did not read from a file; one job per GPU; back up checkpoints to S3 as
they appear.
