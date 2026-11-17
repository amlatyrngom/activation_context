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

**Status 2026-11-10 ≈02:00Z** (details and launch commands: PRE_REBUTTAL `PLAN.md` log):
- §3.5 rescore under the single scorer (dev1000, n = 1,000, greedy, cap 256, strict parse): text+LoRA u1000 74.1 / u2000 72.6 /
  u3000 72.8 / u4000 (final) 72.3; AC `S_kv_v4fr` 73.5, `T_in_v4` 73.4, `S_kv_v4` 72.6, `T_kv_v4fr` 71.7, `T_kv_v4` 71.5, `T_res_v4` 71.5;
  untuned 4B 29.7. The paper's 4-point gap (68.5 vs 72.5) is mostly scorer / eval-set noise: AC final ≥ text final, text peak 0.6 above.
- Running: CSAIL `S_kv_fr90k`; ORCD `X_text_90k`, `S_kv_fr_e1e4`, `S_kv_pen0`; EC2 spot rqa / rqc (retrieval 9B text arm, gold-ref vs
  answer-only target), rqb (retrieval AC R_res_rq9 → T_res_rq9), cmpsum2 (uncapped compaction summary: s2 140/200, s3 136/200 solved).

## Setup (2026-11-09)

| What | Where / how |
|---|---|
| Workstation | EC2 `i-0918b8f73e9a0fe03` (t3.2xlarge, 8 CPU, 30 GB RAM, **no GPU**), repo `/home/ubuntu/activation_context`. Never stop or terminate it. |
| Git | One branch, `main` (= the paper code + container patch + IB, merged over the old main). Old main = tag `archive/main-20260921`. Agents commit and push at milestones. |
| Paper | `paper/` submodule → github.com/amlatyrngom/activation_context_paper (git-synced with Overleaf; pull first). Build: `IB/ARTIFACTS/PAPER_WRITING/build_pdf.sh [--pages]` (TinyTeX in `~/.TinyTeX`). Submitted build: body 10 pages (references from page 11), 15 pages total; appendix 3 pages. |
| Python | `.venv` from `uv.lock`; CPU tests: `.venv/bin/python -m pytest activation/tests` (GPU tests need `--gpu`). |
| GPUs | SkyPilot/EC2 (`.venv/bin/sky`; up to 4 spot RTX PRO 6000; long work runs as `python -m activation.cloud.sky exec`, so idle autodown cleans up), CSAIL, ORCD. Running jobs: status block above. |
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

**2026-11-15:** §3.1 done (frozen-reader KL stage 2 = main row; held-out test1000 in `PRE_REBUTTAL/sec35/TEST.md`, figure
`sec35/figs/fig_retrieval_s31_test.*`). Next (user): clean linear recipes for compaction (frozen reader only) and math synthesis
(frozen first, reader-LoRA hedge). Draft plan awaiting the user: `IB/ARTIFACTS/PRE_REBUTTAL/CLEAN_RECIPES_PLAN.md`.

**2026-11-15 (vLLM Hidden Residuals):** deep AC inputs (rows + per-layer residual deltas) now run in vLLM 0.28 for the
Qwen3.5-9B reader via an out-of-tree model (`activation/vllm_residuals/`, plugin entry point; `engine_to_device` adds the
`hf_overrides` when a deep AC model targets the reader). Validated against HF hooks on `T_res_rq9fr_kI_KL` (top-1 16/16, greedy
13/16 compiled, 15/16 eager; eager zero-delta = upstream bit for bit), prefix cache keys on deltas, throughput −8 % (tokens/s) at 4,000
rows but a 9× request payload (304 MB). Branch `worktree-agent-a9ce275374672303c` (not merged); details
`IB/ARTIFACTS/PRE_REBUTTAL/vllm_res/README.md`.

**2026-11-15 ~23:30Z compaction stage 1 set up (not launched).** Code `IB/ARTIFACTS/PRE_REBUTTAL/compaction/stage1/` (+ `sec45_train
--kind packet`); data S3 `pre_rebuttal_20261109/compaction/stage1/`. Packets: `s1kl_c20_v1` (eval cut rule at 20k, 1,261 runs = items,
157 whole updates) and `s1kl_c162024_v1` (16k/20k/24k cuts, 3,796 items, 474 updates; not trained). New seed caches: drop their
rollouts.jsonl into `IB/TMP/PRE_REBUTTAL/compaction_stage1/pools/<cache id>/` and rerun `build_stage1.sh <new name>`. Dev-75 proxy
(`cmp_proxy_eval.py`): NLL full 0.5057, none 0.6449; start pair R_res_rq9fr gap closed −0.49 (1/10) / −0.72 (1/5); ORCD smoke (20
updates) +0.078 / +0.076. 69 s/update median, 100 s effective (teacher blocks included), peak 57.1 GB. Launch (ORCD):
`cd /home/ngom/orcd/scratch/activation/cmp1 && sbatch --export=NIL --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 jobs/cmp_stage1_job.sh S1_res_kl_c20 s1kl_c20_v1`.

**2026-11-16 split rule restated by the user (ROOT_CANON): items never repeat across train/dev/test; repositories should be shared.** The first stage-1 packets used `--sides training` (repository-level: no eval-side repository trains), so they break it. Next packet adds the evaluation-side tasks minus the 322 excluded (seed 0: 206 qualifying runs from `pc_pool9_h6_eval`, 91 repositories, 37 Kotlin; no eval_v3 run qualifies outside the 322) and the new seeds. The two running jobs finish as the cut comparison only.

**2026-11-16 ~02:30Z compaction diagnosis.** Stage 1 from R_res_rq9fr plateaus: dev-75 gap closed ≈0.11 at u41 and u81 (train KL flat after ~5 updates). Old H8a (`compaction/stage1/OLD_H8A_DEV75.md`): rows only 0.886 / 0.962 (1/10, 1/5, with its reader LoRA); its encoder into the untuned frozen 9B 0.382 / 0.496; its adapter alone without rows 0.022. So a frozen reader can read good rows; our encoder is the gap. R_res_rq9fr saw only 1–2k-token FineWeb passages; the old lineage had lc3i (8,300 updates) + 1,000 updates long-context reconstruction. Testing a clean long-context reconstruction phase on the training prefixes (agent, ORCD) before stage 1.

**2026-11-16 ~04:30Z math teacher test done (`math/TEACHER_TEST.md`, 992941c).** 200 hard training problems × 3 repeats, paired vs the 9B reading the three results only (77.67): 9B full runs 77.50 (−0.17 [−3.00, +2.50]); 27B full runs 78.83 (+1.17 [−1.83, +4.17], p 0.41); 27B results-only 78.00. No arm is significantly better; on mixed problems (k=1–2, n=53) the 9B full-runs teacher trails by 4.4 (it re-solves 5× less). Single solver 70.17, majority 74.33, any-of-3 82.00. Teacher choice is the user's (pending).

**2026-11-16 ~07:10Z compaction** stage 1, dev-75 NLL r10/r5 (gap closed): `S1_res_kl_both_c20` (seed 0, both sides, 183 updates from R_res_rq9fr, reader frozen) u81 0.6271/0.6274 (0.128/0.126), u121 0.6085/0.6073 (0.261/0.270), u161 0.5765/0.5638 (0.491/0.583), final u183 0.5739/0.5603 (0.510/0.608), KL 0.0627/0.0505. Training-side only: `S1_res_kl_c20` final u157 0.6260/0.6262 (0.136/0.134); cuts variant u241 0.6238/0.6243 (flat). Old H8a encoder + frozen untuned reader 0.382/0.496 (KL 0.0833/0.0689), H8a with its reader LoRA 0.886/0.962. Order is shuffled (ac_model_training.py), so the late jump is a learning transition; one run per arm, so eval-side repositories vs run-to-run timing is not yet separated (S1_lcrec100_kl_both_c20 on the same packet is a second sample). Next: all-seeds both-sides packet.

**2026-11-16 ~07:40Z long-context reconstruction works:** `S1_lcrec100_kl_both_c20` (stage 1 on s1kl_both_c20_v1 from R_cmp_lcrec u100, reader frozen) at u41: NLL 0.5785/0.5654, KL 0.0669/0.0551, gap closed 0.477/0.571, vs 0.111/0.106 for the R_res_rq9fr start at u41 (0.510/0.607 at its final u183). Compaction recipe = R_res_rq9fr (FineWeb recon) → long-context recon on the training prefixes (one pass, `R_cmp_lcrec`, 241 updates) → KL stage 1. Cuts variant S1_res_kl_c162024 cancelled at ~u250 (old start, training side only, flat). Larger run (all seeds, both sides, from R_cmp_lcrec final) next.

- 2026-11-16 09:50Z: larger compaction stage-1 run `S1_lcrec_kl_both_c20_s012` launched on ORCD (job 29301026, resume job 29301045 after the 11-17 maintenance): R_cmp_lcrec_final start, packet s1kl_both_c20_s012_v1 (4,470 items, seeds 0-2, both sides, 0 of the 322), 558 updates. Smaller recon-start run S1_lcrec100_kl_both_c20 at u121: gap closed 0.582/0.685 (dev-75 r10/r5).

- 2026-11-17 00:30Z: first compaction AGENT eval of a res pair via vLLM residuals (dev-75, seeds 2-4, 50k; `compaction/AGENT_EVAL_DEV75.md`): S1_lcrec100_kl_both_c20 final ac_4000 67.56 % solved (full 75.56, none 58.67, uncapped summary 70.22; −8.0 vs full p 0.006, −2.7 vs summary p 0.41), ac_2000 59.56 % (= none). Main defect: the reader garbles tool calls ("[/shell]" … without "<tool_call>", 43/225 and 51/225 runs; HF reproduces it, so the pair not vLLM; "</think>" is the competing token). Open hypothesis: the encoder view escapes reasoning as "[Reasoning]…[/Reasoning]" and the reader imitates "[/". Probe + larger-run dev-75 eval queued (agent).
