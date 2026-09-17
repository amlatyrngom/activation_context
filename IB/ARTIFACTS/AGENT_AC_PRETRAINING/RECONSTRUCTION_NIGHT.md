# Reconstruction night — restart note (written 2026-09-15, ~19:00 user time, before the green light)

If this session restarts, read this first, then `IB/STATE.md`, then the ledger below if it exists.

## What was agreed in chat (in-chat planning for the new POC round)

- Diagnosis of the RAG AC null result: the co-trained reader takes the no-context shortcut (learns the answer prior, memorizes question→answer), and the encoder channel learns very slowly even with the reader frozen. See `RAG_AC_NEXT_PLAN.tressoir.md` (Completed) and `rag_ac_poc/ABLATION_RESULTS.md`.
- New round: **mass reconstruction SFT** on text the model has not memorized (SWE-trace and deep-research tool outputs / page content, chunks 128–512 tokens, plus LOFT/Loong for length), plain assistant text (no `submit_answer` wrapper, minimal system prompt, no tool schema). New task class `reconstruction` in `ActivationContextStudyGenerator` with two variants: full reconstruction and cued continuation. No oracle, no teacher pass.
- **No-context penalty** as the default when the reader trains: `loss = task(AC view) + λ·KL(reader_adapted(no-context view) || base(no-context view))` at the target positions; no-context view = `ac_messages` with AC parts stripped; base reference cached once per dataset with the top-k machinery. Defaults: sample p = 1/4 of examples per update, scaled by 1/p (unbiased), λ = 1; raise λ to 2–3 if the held no-context NLL moves > ~0.02 nats from the base. Skipped when the reader is frozen or λ = 0.
- **Batching**: bring the agent trainer's packing (`packed_attention_masks`, `packing_spacer_tokens`, `check_packing_support`) to the AC trainer's reader side; batch the encoder over a micro-batch; a probe measures items/s and peak memory vs token budget per ratio and writes a tuning record (style of `activation/autotuners`) that the run reads. Tonight's runs were CPU-bound (GPU util 30–42 %, one core at 100 %).
- **Anchors (never touched)**: ratio 1/16 and ratio 1/4, reconstruction SFT, reader trainable with the penalty (λ = 1, p = 1/4), 2 epochs over ~30k unique chunks (budget script picks the count from the smoke run), panels every 1/8 epoch.
- **Hill-climbing with genuine methods** for ~14 h on up to 3 RTX PRO 6000 nodes (`--gpu-type rtx6000pro`, 96 GB): node 1 anchors + promoted-recipe reruns; node 2 single-variable side experiments; node 3 reserved for scale. Fair game: penalty weight/sampling, ratio curricula, reconstruction/continuation mix, packing, LRs/schedules, encoder/row-scale init, staged freezing, data mix/chunk lengths. Not fair: train/held overlap by source, changing metrics or panels after a run starts, eval-time tricks, unflagged memorization.
- **Frozen evaluation protocol** (write `rag_ac_poc/PROTOCOL.md` before hill-climbing starts): held-out novel panel (200 chunks, unseen sources), Wikipedia panel (100 NQ chunks) as memorization detector, no-context floor, fixed seeds; metrics: reconstruction NLL/token, greedy token accuracy and edit distance, no-context NLL (penalty monitor), train-vs-held gap. Promote a change only if it improves held novel-text fidelity on this protocol.
- **Ledger** `rag_ac_poc/LEDGER.md`: every run recorded before it starts (hypothesis, the one change, expected effect) and after (result).
- Pre-authorizations requested: up to 3 RTX PRO 6000 nodes for up to 14 h, teardown without asking; product changes on a tagged snapshot, nothing committed to main; kill/replace non-anchor experiments freely; old run folders moved aside, never deleted; pull best-recipe checkpoints (few GB cap) before teardown. User is switching to yolo mode; user returns ~7–9 am their time.
- Wake-up: a persistent Monitor emitting a status line every 15 min plus errors/result arrivals (worked tonight).

## Facts from tonight that carry over

- Helpers: `IB/TMP/AGENT_AC_PRETRAINING/rag_ac_next/{remote.sh, snapshot.py, summarize_arms.py, skyenv.sh}`. `sky exec` reserves every GPU of a node → concurrent runs start over ssh (`remote.sh start`), a sleeping sky job holds off the 30-min autostop (`remote.sh keepalive`), folders are pulled with `remote.sh watch` (`sky watch NAME REMOTE LOCAL --until-file result.json`). Do not edit a helper script while a long invocation of it runs; use a frozen copy.
- Launch: `uv run --frozen python -m activation.cloud.sky setup <name> --gpu-type rtx6000pro --gpus 4` (check `sky.py` for the exact flags); source skyenv.sh first; fallback image needs `uv sync --frozen` on the node (`remote.sh warm`); bench rsynced to `/root/activation_artifacts/rag_ac_poc/`, data root `/root/activation_artifacts/SYNC/<ROUND>/`.
- Baseline for this round: tag the working tree before the first product change (`snapshot.py` pattern, `rag-recon-baseline-<ts>`); `launch.py` verifies `training_source.json` on the node.
- Never read/edit `IB/TASK.md`; `.env` only holds GITHUB_TOKEN and is not needed on the node; `IB/TMP/SYNC` is large, never push it wholesale.

## Next step (as of this note)

1. Wait for the user's green light (they are switching to yolo mode).
2. Tag the baseline; write `PROTOCOL.md`; launch node 1 in the background while implementing: reconstruction generator → no-context penalty view + cached reference → reader packing + probe → bench (`run_recon.py` from `run_poc.py`) → CPU tests → independent review.
3. Probe, smoke (512 × 2, one GPU), budget, then the two anchors; start the ledger and the 15-min monitor; then side experiments on node 2.
4. At 14 h or earlier: summary page, results doc, STATE/plan/canon updates, pull best checkpoints, tear down all nodes, confirm `sky ls` empty.

## Night log additions (00:15 UTC 09-16)

- **ssh to the nodes stalls at the banner exchange whenever `SSH_AUTH_SOCK` points at the dead VS Code agent socket.** ssh consults the agent before offering the key and hangs. Every node command must run with the variable unset: `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/nssh.sh` (retrying ssh) and `remote_run2.sh` (frozen helper with the unset) do this. Never `pkill` ssh by pattern (it killed sky's log stream once).
- Preparation on ac-recon-1: 40,000 training windows, 200 held, 100 wiki (`material.json`, 47 MB, also pulled locally). Review of the delta: `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/IMPLEMENTATION_REVIEW.md`; D1 (top-k 256 cache for the penalty reference) and D2 (completion budget shorter than half the passages) fixed; snapshot tag `ac-recon-20260916T000008Z`.
- Smoke S started 00:01 on ac-recon-1 GPU 0; keepalive job (14 h sleep) and a 30 s watch run in the background from this session; 15-minute Monitor prints `status.py`.
- Node ac-recon-2 launched 00:02 for side experiments.

## State at 01:02 UTC 09-16

- **Running.** Node ac-recon-1: A16 (GPU 0), R0 (GPU 1), A4 (GPU 2), C5 (GPU 3). Node ac-recon-2: B (0), P0 (1), P3 (2), E3 (3). All started over ssh from the v2 root (`/root/sky_workdir_v2`, bench `/root/activation_artifacts/ac_recon_v2`, tag ac-recon-20260916T003825Z; anchors on 002847Z, identical `activation/`). Arms in `ac_recon/arms.json`; the calibration (21,888 anchor items) in `IB/TMP/SYNC/AC_RECON/preparation/calibration.json` and on both nodes.
- **Local mirrors.** `IB/TMP/SYNC/AC_RECON/<label>/` pulled every 30 s by `remote_v2.sh watch` processes started from this session (logs `watch_<label>.log` in `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/`); `status.py` summarises them; a 15-minute Monitor prints it. Keepalive sky jobs hold both nodes (14 h / 13 h).
- **Helpers.** `remote_v2.sh` (v2 root, `NODE=ac-recon-2` for node 2), `remote_run2.sh` (original root, used only for `budget`), `nssh.sh` (ssh with the agent socket unset and retries). Never `sky exec` on a node with a keepalive job: the wrapper reserves all GPUs and the job queues forever.
- **Results so far.** S: novel 1.905 → 1.629 gold NLL, floor 1.921 → 1.855, wiki 2.338 → 2.261, greedy fidelity 0.07. S4 (4 per forward, no checkpointing): equivalent panels at 2.3× the speed but 92 GB peak; S8 OOM twice. Round 1 side arms run one example per forward.
- **Next.** Round-1 first panels ~01:45–02:00; anchors' first panels ~02:00. Round 2 at ~05:00 on node 2 (B/P0/P3/E3 end) and node 1 GPUs 1, 3 (R0/C5 end ~03:30): combine winners; consider 4 per forward with checkpointing above 2000 batch tokens. Anchors end ~08:15. Then results doc, plan completion reports (`AC_RECON_PLAN.md` M2–M5), canon/STATE, pull best checkpoints, `sky down` both nodes, confirm `sky status` empty.

## State at 01:48 UTC 09-16

- **Node 3 (ac-recon-3) up since ~01:10**: G3 (GPU 0), E3F (1), R0F (2), G1 (3) — full anchor size, 3 examples per forward (`micro_batch=3`, no checkpointing; S3 probe: 1.58 s/step vs 3.47, 64 GB), started with `remote_v4.sh` (per-run checkpoint root `SYNC_RUNS/<label>`). G1/G3 use gold-inclusive penalty targets (`teacher_targets_include_gold`, tag ac-recon-20260916T013934Z).
- **Checkpoint collision hazard** (shared `SYNC/AC_MODELS/rag_poc/epoch_NNN`): `ckpt_mover.sh` runs on nodes 1 and 2 (`/root/ckpt_mover.log`), moving finished pairs to `SYNC/CHECKPOINTS/`. Any new run must use `remote_v4.sh`.
- **Base no-context floor on the held novel panel = 1.9209** (= the initial reader's). Floors below it are prior learning. Under the plain penalty (λ=1, 25 %) the floor drifts to ~1.82–1.85 within an epoch of 512 items (S, S4, S4b); the penalty's top-32 KL cannot see mass moved onto gold tokens outside the base's top 32 — hence G1/G3.
- Early panels: P0 (no penalty) novel 1.530 at epoch 0.375; R0 (frozen reader) 1.600 at 0.5 with the floor pinned. Anchors still in their reference pass at 01:42 (21,888 items).

## State at 03:40 UTC 09-16

- **Findings so far (half-size arms, epoch-1 floors).** No penalty (P0): floor 1.921 → 1.633, most of the novel gain is prior learning. Plain penalty: floor drift ≈ 0.04 at λ=1 (B 1.881) and λ=3 (P3 1.886) — the top-32 KL is blind to mass moved onto gold tokens outside the base's top 32 (G1/G3 test the gold-inclusive fix on node 3). Frozen reader R0 final: novel 1.422, floor 1.921, gap 0.499, no train/held gap. B's honest gap at epoch 1 (0.46) ≈ R0's at the same point; cued continuation (C5) and encoder LR 3e-4 (E3) hurt. A4 lags A16 badly early (1.542 vs 1.389 at epoch 0.5).
- **Running.** Node 1: A16 (0), F1 staged (1, started 03:38, `target_lora_frozen_updates=1368`, tag 033033Z), A4 (2), C5 (3). Node 2: B, P0, P3, E3. Node 3: G3 (0), E3F (1), R0F (2), G1 (3) — full size, 3 per forward, per-run checkpoint roots.
- **Timeline.** P0 ~03:55; B/P3/E3/C5 ~05:15; R0F ~05:00; G1/G3/E3F ~05:45–06:15; F1 ~07:00; anchors ~08:00–08:15. Round 3 slots open ~05:00 (node 2 + node 1 GPU 3 + node 3): decide from G1/G3 (penalty design) and R0F (frozen at scale).
- `table.py [epochs...]` prints the matched-epoch panel table; results doc to be written at `IB/ARTIFACTS/AGENT_AC_PRETRAINING/AC_RECON_RESULTS.md`.
