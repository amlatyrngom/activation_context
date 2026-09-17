# Reconstruction night — RESTART NOTE (auto-written 2026-09-16 17:27 UTC)

Read this first after any restart (new session or new account). It is regenerated every ~15 minutes by
`IB/TMP/AGENT_AC_PRETRAINING/ac_recon/checkpoint.py`; the hand-maintained plan is in `NEXT.md` next to it.
Chronology and results: `IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/LEDGER.md`. Protocol: `ac_recon/PROTOCOL.md`.
Longer background: `RECONSTRUCTION_NIGHT.md` (same folder as this file), `AC_RECON_PLAN.md`, `AC_RECON_RESULTS.md` (draft).

## Standing rules (from the user)
- Yolo mode; autonomous. Pre-authorized: up to 3 RTX nodes (the user asked at 06:50 UTC whether we are at ≤4 GPUs — see NEXT.md),
  teardown without asking, product changes on tagged snapshots only (nothing committed to `main`), kill/replace non-anchor
  runs freely, old run folders moved aside never deleted, pull best-recipe checkpoints (few GB cap) before teardown.
- Anchors A16 (ratio 1/16) and A4 (1/4) are never touched. Hill-climb only with genuine methods (no memorisation tricks).
- Never read or edit `IB/TASK.md`. `.env` (GITHUB_TOKEN only) is never synced to nodes. Never push `IB/TMP/SYNC` wholesale.
- Never edit `remote*.sh` while a long invocation runs (bash re-reads scripts); make a frozen copy instead.
- ssh to nodes needs `SSH_AUTH_SOCK` unset (`nssh.sh` does it). Never `pkill -f` a pattern that matches your own shell or the
  remote ssh shell; processes exec into `run_recon.py`, so match `[r]un_recon.py`.

## Infrastructure
- Nodes `ac-recon-1/2/3`: AWS eu-central-1a g7e.24xlarge, 4× RTX PRO 6000 96 GB each. `sky` via
  `uv run --frozen python -m activation.cloud.sky ...` (wrapper) or raw
  `uv tool run --no-config --python 3.13 --with pip --from "skypilot[aws]==0.13.0" sky ...` (raw sky fails with a uv temp-dir
  permission error in this session; use the wrapper). Autostop after 30 idle minutes is held off by sleeping `sky exec` jobs
  (keepalives: originals expire 13:11 / 13:30 / 14:01 UTC; 20,000 s renewals queued at 06:55 as jobs 6 / 4 / 3 on nodes 1/2/3).
  A keepalive job blocks any later `sky exec`, so runs start over ssh.
- Helpers in `IB/TMP/AGENT_AC_PRETRAINING/ac_recon/`: `remote_v5.sh` (use for ALL new runs: v2 root `/root/sky_workdir_v2`,
  bench `/root/activation_artifacts/ac_recon_v2`, per-run checkpoint root `SYNC_RUNS/<label>`, expandable segments;
  commands `push`, `sync-source`, `start LABEL GPU`, `watch LABEL MINUTES`, `keepalive SECONDS`), `remote_v2.sh` (keepalive),
  `nssh.sh` (`NODE=ac-recon-N bash nssh.sh 'cmd'`), `status.py` (one line per run from the local mirrors),
  `table.py [epochs]` (matched-epoch table), `snapshot.py` (tags the working tree `ac-recon-<ts>`, writes
  `training_source.json` that `launch.py` verifies on the node), `test_recon.py` + `test_batched.py`
  (`cd IB/TMP/AGENT_AC_PRETRAINING/ac_recon && uv run --no-sync python -m pytest -q test_batched.py test_recon.py -p no:cacheprovider`),
  `ckpt_mover.sh` (on nodes 1–2 as `/root/ckpt_mover.sh`), `checkpoint.py` (this note), `NEXT.md`.
- Data on nodes: `/root/activation_artifacts/SYNC/AC_RECON/preparation/{material.json (40k train), material_100k.json (node 1),
  calibration.json (train_items 21888), wiki_source.json, initial/}`; run folders `/root/activation_artifacts/SYNC/AC_RECON/<label>/`
  (`report_data.json`, `console.log`, `result.json` when done). Local mirrors `IB/TMP/SYNC/AC_RECON/<label>/` are pulled by
  `remote_v*.sh watch` processes (logs `watch_<label>.log` in the helper folder); after a restart, restart the watches for
  running labels: `NODE=ac-recon-N setsid nohup bash remote_v5.sh watch LABEL 600 > watch_LABEL.log 2>&1 &`.
- Source: product changes live in the working tree (`activation/ac_model/ac_model_training.py`, bench
  `IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/run_recon.py`, `arms.json`), tagged snapshots `ac-recon-20260916T132825Z, ac-recon-20260916T151845Z, ac-recon-20260916T155942Z, ac-recon-baseline-20260915T233324Z`
  (latest ). Nodes carry the latest snapshot (pushed with `sync-source` + `push`).
- Node ↔ run map (06:50 UTC): node 1 = A16 (GPU 0), F1 (1), A4 (2), RL5 (3); node 2 = B4, RL1, R0F4, A3E; node 3 = R0F3, G1b, P1F, A8.

## Key results so far (novel gold NLL / no-context floor; base floor 1.921)
B (half, λ=1) 0.770 / 1.882 · P0 (no penalty) 0.912 / 1.689 · R0 (frozen, half) 1.422 / 1.921 · R0F (frozen, full) 0.880 / 1.921 ·
P3 (λ=3) 0.839 · E3 (enc LR 3e-4) 1.528 · C5 (cued continuation) 1.426 · A16 ~1.00 at epoch 1.6 · A4 1.51 at 1.6 (no transition) ·
R0F4 (frozen, 1/4) 0.849 at epoch 1.0 · B4 0.713 at 2.75/4. Findings: penalty caps floor drift at ~0.04; no penalty is worse on
novel too; encoder is the channel; sharp transition after 1.5k–3k updates with replicate variance; at 1/4 the trainable reader,
not marker selection, is the bottleneck (R0F4 vs A4).

## Plan and next steps (hand-maintained; updated at each monitor wake)

- **Deadline**: user said "assume it's 2 pm EST" at ~06:35 UTC → stop ≈ 18:00 UTC 09-16. Teardown (`sky down` all nodes, confirm `sky status` empty) by 18:00 UTC, after pulling best-recipe checkpoints (few GB cap) and finishing the docs.
- **GPU budget (user, 07:05 UTC): the intent was 4 GPUs IN TOTAL, i.e. one node.** Cost is $28.18/h per node ($84.54/h for three; ~$580 spent on the three nodes by 07:10). Drain plan: no new launches on nodes 2–3; node 2 DOWN at 08:20 (B4 OOM at epoch 4, R0F4 done; RL1/A3E killed); node 3 DOWN at 09:15 (G1b done; A8/P1F killed) — CONFIRMED by the user at 07:12 UTC. Node 1 stays (A16, A4, RL5, then round 4).
- **Headline schedule (user, 07:15 UTC: think about when the headline 1–2 large runs start; late starts may exceed 18:00 if insights require).**
  Findings so far fix the recipe: gate group at 3e-3 (row scale climb was the silent phase), reader frozen/staged at 1/4. Plan:
  REVISED 11:20: node 1 = X40K (GPU 0, 1/16 HEADLINE: gold + skip init, 40k × 2, end ~18:30), K16 (GPU 1, gold + skip, anchor size, end ~15:00 — 1.105 at epoch 0.125!), A4K (GPU 2, A4 + gold + skip, end ~16:30), R4X (GPU 3, 1/4 headline, frozen reader, 40k × 2, end ~15:30). Then E4K on GPU 1 (after K16); A4r on GPU 2 only if A4K fails to transition. Gate group withdrawn (G16 trailed A16). Teardown of node 1 after X40K's result (~18:30–19:00, allowed by the user for late starts); micro node down after SKn (~12:30).
  (gates, 40k × 2 = the 1/16 HEADLINE, ends ~13:30–14:00) on GPU 2; ~09:30 decide the 1/4 HEADLINE recipe from G4's first
  1,000 updates vs A4 (gates alone vs staged F4X) and start it on GPU 3 (kill RL5, low value) → ends ~15:00–15:30.
  If the probe shows ≥1.5× at 8/forward, the headline runs use micro_batch 8.
- **Micro node** (user-authorized, 07:15 UTC): one L40S (`ac-micro`, g6e.xlarge, ~$1.9/h; setup log in the scratchpad) for S-size
  (512 × 2) checks of code changes: gate group, RowHead identity-skip init (review finding 2), extended LoRA targets for the
  GDN layers (finding 4), token-weighted loss (finding 7). Baseline S: novel 1.629 / floor 1.855.
- **Round 4** (arms in `ac_recon/arms.json`, material `material_100k.json` on node 1 = standard panels, training superset): start with `NODE=ac-recon-1 bash remote_v5.sh start <LABEL> <GPU>` as node-1 GPUs free (F1 ~07:25 → S8c throughput probe (~10 min, 8 per forward) then X40 with the probe's micro_batch if it fits; A16/A4 ~08:05 → R4X, F4X; RL5 ~13:00). Then `remote_v5.sh watch <LABEL> 600` in the background for the local mirror.
- **A4r** (A4 replicate, different seed) starts in the first free node-1 slot (~14:30 when A4G or K16 ends); it may finish ~19:00.
- **Results to collect** (record each in LEDGER.md; final table via `table.py` and `result.json`): F1 07:20, A16/A4 08:05, B4 08:10, RL1/R0F4 08:30–09:00, G1b/P1F/A8 09:00–09:30, R0F3 09:00, A3E 10:45, X40/R4X/F4X ~14:00–15:00.
- **Docs to finish before teardown**: `AC_RECON_RESULTS.md` (findings 1–5 drafted; final table), `AC_RECON_PLAN.md` M2–M5 completion reports, LEDGER, STATE, canon (ACTIVATION_CONTEXT_CANON) update, this restart file.
- **Code review** of the trainer/AC code (clipping, warmup, schedule, weight decay, padding, dtype) requested by the user at 06:50 UTC; a subagent is reviewing; fixes go on a tagged snapshot (`snapshot.py`), tests `test_batched.py test_recon.py` must pass, then `remote_v5.sh sync-source` + `push` to nodes before any new launch.
- **ORCD (free Slurm cluster, user's account)**: env at `~/activation_ws` + `~/activation_venv`, bench `~/activation_artifacts/ac_recon_v2`, data `~/activation_artifacts/SYNC/AC_RECON/preparation`. Submit with `bash orcd_run.sh LABEL [mit_normal_gpu|mit_preemptable] [h200|rtx_pro_6000|h100|l40s] [hours]`; pull with `bash orcd_pull.sh LABEL`. Caps: 2 GPUs on mit_normal_gpu (6 h), 4 on mit_preemptable. First job SKn (22823375) pending. Planned there: seed replicates (onset variance), ratio sweep (1/4, 1/8, 1/16, 1/32) with skip + gold, data scaling to 100k. Never copy the user's `.bashrc` (holds a token).


## 11:58 UTC — ORCD items (user: checkpointing → H200 utilisation → use right away; later: what fits 6 h / 2×6 h)
1. DONE code: resume from last completed epoch (trainer `resume_optimizer`, bench `resume_state`), tag ac-recon-20260916T115407Z on ORCD. `orcd_run.sh` has `--requeue`, AFTER_JOBID chains, shared AUTOTUNE.
2. LIVE TEST: SR (job 22823545, preemptable). When `~/activation_artifacts/SYNC_RUNS/SR/AC_MODELS/rag_poc/completed_epoch.json` appears: `scancel 22823545`, resubmit `bash orcd_run.sh SR mit_preemptable rtx_pro_6000 2`, confirm console shows "AC resume: ... restored" and result.json has resumed.epochs_done=1, updates ≈ 128.
3. H200: P8H (mb 8, no ckpt) estimated start 12:08 UTC; read its console/training_stats (throughput, OOM?) → choose H200 micro_batch/ckpt; copy the published FLA profile from `~/activation_artifacts/AUTOTUNE/profiles/*.json` into `activation/autotuners/configs/` as a preset.
4. Keep 4 preemptable slots busy (K16r1, K16r2, A32K, SR now); A8K waits on H200 (est. 22:10 UTC) — consider moving A8K to preemptable after SR finishes.
5. Every wake: `python3 checkpoint.py` (includes the queue snapshot). Waits so far: preemptable 0.2 min; normal_gpu H200 hours (Priority, ~350 pending).

## 12:50 UTC — after review 2 (both reports in ac_recon/CODE_REVIEW_2_{fork,fresh}.md; details in LEDGER 12:30–12:45)
- Node 1: `slot_waiter3.sh` (scratchpad, log `slot_waiter3.log`): K16 result → K16D on GPU 1; R4X result → K16G0 on GPU 3; A4K result → A4r on GPU 2 only if novel > 1.3. X40K stays on GPU 0 (panel 21 % contaminated; quote clean numbers via `clean_panel.py`).
- ORCD preemptable: midi arms MK/MD/MC/MP/MA/MK0/MK4/MKT/MKP/MKG then A8K, as K16r1/K16r2/A32K free slots (~15:30 UTC). Read each by the epoch-2 panel vs MK; anything ≥ 0.05 better goes into the chunk recipe. P8H (H200 probe) pending on normal_gpu.
- ORCD normal H200 (when P8H gives the batch size): X40KD-clean = K recipe + decay + clip on deduplicated 40k material, 2 chained 6-h jobs (`ckpt_every 500`, AFTER_JOBID); then A4KD or R4XD after A4K's result.
- Code still open: live full-vocab penalty (F2), parallel/cached example building, shingle dedup in `prepare` (write material_40k_clean.json), lighter panels for chunks; results doc rewrite with clean/all numbers and the K16/A4K findings (finding 4 superseded: 1/4 fails only with the random row head).
- Every wake: `python3 checkpoint.py` (queue snapshot + wait policy), `status.py`, pull checkpoints before node-1 teardown (K16 epoch 2, K16D, A4K, X40K).

- 14:00 UTC: K16 DONE 0.190 (checkpoint pulled: CHECKPOINTS/K16_epoch2). K16D running on GPU 1 (watch_K16D.log). Remaining node-1 order: R4X result (~15:05) → K16G0 on GPU 3 (slot_waiter3); A4K result (~16:40) → A4r only if > 1.3 (it will not be). ORCD order: ND → NK → P8H → X40KD → MW/NW → other N arms; K16r1/r2 end ~16:00 UTC. MD (2,048 × 2 + cosine) 0.800 vs MK 1.581: the schedule matters; MW/NW separate warm-up from decay.

## Live run status (2026-09-16 17:27 UTC, from the local mirrors)

```
A16: RESULT novel 0.8777624069433659 acc 0.7797875542938709 | wiki 1.3487666273117065 | floor 1.8754533167928458 | seen 0.7398335693942499 | 431 min
A32K: training ep 2 of 2 upd 2966 of 5472 loss 0.6761 pen 0.0210 rem about 1h 25m | novel 0.783/0.788@1.0 wiki 1.123/0.708@1.0 floor 1.886/0.640@1.0 | stale 33s
A32X: AC student, held-out ep 1 of 1 upd 768 of 1024 loss 0.4103 pen 0.0118 rem about 9m 25s | novel 0.363/0.892@0.625 wiki 0.621/0.815@0.625 floor 1.894/0.640@0 | stale 39s
A3E: AC student, held-out ep 2 of 3 upd 3762 of 8208 loss 1.0354 pen 0.0211 rem about 3h 53m | novel 1.508/0.683@1.25 wiki 2.242/0.538@1.25 floor 1.863/0.643@1.0 | stale 33046s
A4: RESULT novel 1.5060649847611784 acc 0.6848618714511394 | wiki 2.2654837942123414 | floor 1.873317081257701 | seen 1.2177288306120317 | 434 min
A4G_killed: training ep 1 of 2 upd 1182 of 5472 loss 1.3992 pen 0.0111 rem about 6h 10m | novel 1.489/0.683@0.375 wiki 2.151/0.544@0.375 floor 1.921/0.640@0 | stale 22503s
A4K: RESULT novel 0.36396004233974966 acc 0.9033487756550312 | wiki 0.5621379348635673 | floor 1.8878304896131157 | seen 0.3084662548344568 | 270 min
A8: training ep 2 of 2 upd 4882 of 5472 loss 0.8191 pen 0.0087 rem about 29m 05s | novel 0.818/0.796@1.75 wiki 1.192/0.711@1.75 floor 1.874/0.643@1.0 | stale 29572s
B: RESULT novel 0.7699436894524843 acc 0.7972979700565338 | wiki 1.1560121056437493 | floor 1.8824599582701922 | seen 0.6500336277895258 | 238 min
B4: ERROR OutOfMemoryError('CUDA out of memory. Tried to allocate 110.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 105.75 MiB is free. Including non-PyTorch m
C5: RESULT novel 1.425965147903189 acc 0.6890717525780201 | wiki 2.070677575469017 | floor 1.8804417764768004 | seen 1.1969439056028932 | 231 min
CHECKPOINTS: no report yet ([Errno 2] No such file or directory: '/workspace/IB/TMP/SYNC/AC_RECON/CHECKPOINTS/report_data.json'); stale Nones
E3: RESULT novel 1.5278166274167597 acc 0.6805209615826606 | wiki 2.2716502261161806 | floor 1.8791744884476065 | seen 1.2886206265393412 | 229 min
E3F: ERROR OutOfMemoryError('CUDA out of memory. Tried to allocate 110.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 51.75 MiB is free. Including non-PyTorch me
E4G_killed: building training examples ep None upd None loss None pen None rem None | novel - wiki - floor - | stale 26987s
F1: RESULT novel 1.0559741858486087 acc 0.739073928296566 | wiki 1.5808101373910903 | floor 1.8890407748520375 | seen 0.9778942806078703 | 222 min
G1: ERROR OutOfMemoryError('CUDA out of memory. Tried to allocate 110.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 51.69 MiB is free. Including non-PyTorch me
G16: training ep 1 of 2 upd 1729 of 5472 loss 1.7241 pen 0.0065 rem about 4h 05m | novel 1.528/0.680@0.625 wiki 2.250/0.534@0.625 floor 1.921/0.640@0 | stale 28777s
G1b: RESULT novel 0.6538033827021718 acc 0.8286856289207936 | wiki 1.0342804667353631 | floor 1.8829276748746633 | seen 0.5331691870087525 | 256 min
G3: ERROR OutOfMemoryError('CUDA out of memory. Tried to allocate 110.00 MiB. GPU 0 has a total capacity of 94.97 GiB of which 51.75 MiB is free. Including non-PyTorch me
G4: AC student, held-out ep 1 of 2 upd 1026 of 5472 loss 1.4072 pen 0.0091 rem about 6h 53m | novel 1.563/0.675@0.25 wiki 2.237/0.542@0.25 floor 1.921/0.640@0 | stale 28775s
K16: RESULT novel 0.18962531184864928 acc 0.9413373273611069 | wiki 0.3095292370021343 | floor 1.8941744869574904 | seen 0.12055412309030089 | 237 min
K16D: RESULT novel 0.19928879773855443 acc 0.9392104609310628 | wiki 0.32193449916318057 | floor 1.889954088591039 | seen 0.12110655233891521 | 181 min
K16G0: training ep 2 of 2 upd 4890 of 5472 loss 0.0959 pen 0.0060 rem about 19m 04s | novel 0.247/0.928@1.75 wiki 0.377/0.889@1.75 floor 1.887/0.640@1.0 | stale 28s
MA: RESULT novel 1.6568016306683422 acc 0.6628295660018921 | wiki 2.2714583259820937 | floor 1.8286248916387557 | seen 1.5279850765218725 | 45 min
P0: RESULT novel 0.9117340958956629 acc 0.7690011358261108 | wiki 1.339593991935253 | floor 1.6893384470790624 | seen 0.7720996562566143 | 178 min
P1F: training ep 2 of 2 upd 5035 of 5472 loss 1.5528 pen 0.0198 rem about 23m 20s | novel 1.496/0.685@1.75 wiki 2.248/0.540@1.75 floor 1.870/0.642@1.0 | stale 29584s
P3: RESULT novel 0.8393218473624438 acc 0.7773902171850204 | wiki 1.26016275703907 | floor 1.8834749503433705 | seen 0.7428379332486656 | 235 min
R0: RESULT novel 1.4217465657927095 acc 0.6872145971655845 | wiki 2.0284389275312424 | floor 1.920878918990493 | seen 1.3808683255629148 | 174 min
R0F: RESULT novel 0.8800611328519881 acc 0.7764776746928692 | wiki 1.1951383045315742 | floor 1.9206868579983711 | seen 0.8082897251733812 | 149 min
R0F3: RESULT novel 1.5664437283575534 acc 0.6720166262984276 | wiki 2.2453208065032957 | floor 1.9206868579983711 | seen 1.4883230050909333 | 227 min
R0F4: RESULT novel 0.9055472472403199 acc 0.784847856760025 | wiki 1.2213534101843835 | floor 1.9206868579983711 | seen 0.8558127443684498 | 190 min
R4X: RESULT novel 0.4011680435761809 acc 0.9026191179454327 | wiki 0.6670549981296062 | floor 1.9206868579983711 | seen 0.38841213951673126 | 329 min
RL1: training ep 2 of 2 upd 3843 of 5472 loss 0.6564 pen 0.0104 rem about 1h 24m | novel 0.612/0.839@1.375 wiki 0.955/0.751@1.375 floor 1.873/0.642@1.0 | stale 33047s
RL5: RESULT novel 1.0442644807137549 acc 0.7518364714086055 | wiki 1.4975176331400872 | floor 1.876608537696302 | seen 0.8097673600932467 | 258 min
S: RESULT novel 1.6294464103505015 acc 0.6666094680130482 | wiki 2.2606801480054854 | floor 1.85453104455024 | seen 1.413934785683523 | 32 min
S4: RESULT novel 1.6250722105056048 acc 0.6663956373929978 | wiki 2.260695762634277 | floor 1.8518091997504234 | seen 1.4063919625332346 | 21 min
S4b: RESULT novel 1.6328062492609023 acc 0.6646943931281567 | wiki 2.239120020866394 | floor 1.8354934792220592 | seen 1.4216040738101583 | 25 min
S8c: RESULT novel 1.65595310267061 acc 0.662137272208929 | wiki 2.2268544149398806 | floor 1.83208950471133 | seen 1.555352614239382 | 12 min
SC: RESULT novel 1.6213945491053163 acc 0.6670084026455879 | wiki 2.2424638622999193 | floor 1.8531087334826588 | seen 1.400429507251829 | 44 min
SC_mb1: training ep 1 of 2 upd 3 of 128 loss 1.8997 pen 0.0002 rem about 5h 29m | novel 1.905/0.639@0 wiki 2.339/0.535@0 floor 1.921/0.639@0 | stale 32403s
SC_stopped: training ep 1 of 2 upd 15 of 128 loss 1.9943 pen 0.0422 rem about 1h 43m | novel 1.905/0.639@0 wiki 2.339/0.535@0 floor 1.921/0.639@0 | stale 34645s
SCn: RESULT novel 1.636770858746022 acc 0.666443464756012 | wiki 2.2834839391708375 | floor 1.8506358450651168 | seen 1.4220919032377424 | 45 min
SK: RESULT novel 1.4231517099030315 acc 0.680981851965189 | wiki 1.9295193028450013 | floor 1.84504284132272 | seen 1.256982864910242 | 44 min
SKn: RESULT novel 1.624595088325441 acc 0.6648772305250168 | wiki 2.218526729941368 | floor 1.8434735987335444 | seen 1.4374601174349664 | 44 min
SL: RESULT novel 1.6394286663271487 acc 0.6654450879991054 | wiki 2.2881404268741607 | floor 1.8495049080625177 | seen 1.390829287469387 | 55 min
X40K: training ep 2 of 2 upd 8579 of 10000 loss 0.0875 pen 0.0086 rem about 1h 01m | novel 0.171/0.949@1.625 wiki 0.259/0.921@1.625 floor 1.891/0.641@1.0 | stale 29s
X40_gates: no-context reference ep 0 of 2 upd None loss None pen None rem None | novel - wiki - floor - | stale 28791s
X40_noskip: no-context reference ep 0 of 2 upd None loss None pen None rem None | novel - wiki - floor - | stale 22490s
```

## ORCD (MIT Slurm; `ssh orcd`; helpers `orcd_run.sh LABEL [PART] [GPU] [HOURS] [AFTER_JOBID]`, `orcd_pull.sh LABEL...`, `orcd_queue.py [summary]`)
Caps: mit_normal_gpu 2 GPUs/user (6 h), mit_preemptable 4 GPUs/user (2 d, requeue on preempt). Runs resume from the last completed
epoch (bench `resume_state`, trainer `resume_optimizer`), so requeued/chained jobs continue. Queue log `orcd_queue_log.tsv`, waits `orcd_waits.tsv`.
Latest queue snapshot: `22836932	ac_WCa	mit_preemptable	2026-09-16T13:07:35	2026-09-16T13:12:36	5.0	RUNNING	00:15:22	node5101`
