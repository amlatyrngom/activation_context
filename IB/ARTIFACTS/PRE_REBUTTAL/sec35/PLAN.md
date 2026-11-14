
### 2026-11-13 ~04:50Z — KL plain-BM25 gain replicates on frozen reader
- T_res_rq9fr_20I_KL final: 20I dev1000 69.6 (frozen SFT 71.9).
- Plain BM25 k=100 (rqn stage-2 u0, b8): **57.3** vs frozen SFT 52.0 (side b32). LoRA: KL 59.6 vs SFT 51.9.
- LoRA KL 20I final plain BM25 k=20: 50.9 (SFT 52.0) — KL gain is at large k only (k=20 50.9, k=100 59.6, k=200 60.9).
- Stage-2 LoRA SFT u400 k=100 50.8 (u0 51.9); held answer_nll u1200 0.698 (u0 0.823).
- QA KL u10400 60.3.

### 2026-11-13 ~06:10Z — QA -> 20I KL queued on rqg
- QA KL (T_res_rq9L_QA_KL, no HotpotQA training) 20I dev1000: u10400 60.3, u11200 60.0, u12000 61.3 (ends u12500).
- Queued behind it on rqg (role qa_20i_kl, node script /root/rq20i_node_qa2.sh): T_res_rq9L_QA20I_KL = T_res_rq9L_20I_KL's recipe (KL, 5600 updates, 20I) from the QA KL final. Compare against T_res_rq9L_20I_KL (from R_res_rq9): 20I 70.5, plain BM25 k=20/100/200 50.9 / 59.6 / 60.9.
- Stage-2 LoRA SFT plain BM25: k=100 u0/400/800/1200 51.9 / 50.8 / 51.5 / 53.2; k=200 u0/400/800 49.8 / 52.2 / 52.3.
- Text 100I k=100: untuned 65.6, tuned 75.9; tuned 200I k=200 76.5.

### 2026-11-13 ~08:00Z — frozen KL plain BM25 (b32); text injected reads complete; KL injected reads queued
- T_res_rq9fr_20I_KL final plain BM25 (side scorer b32): k=100 57.5, k=200 57.5 (frozen SFT 52.0 / 51.4). k=100 read: 13 length stops, answered 98.7%.
- QA KL final (T_res_rq9L_QA_KL u12500) 20I dev1000 60.0; T_res_rq9L_QA20I_KL u0 60.0 (init OK), training on rqg.
- Stage-2 LoRA SFT plain BM25 k=200 u1200 55.4 (u0 49.8).
- Text injected reads done: untuned 100I/k100 65.6, 200I/k200 64.9; tuned 75.9 / 76.5.
- ORCD 28927579 (jobs/rq_kl_kI_orcd.sh): 100I k=100 and 200I k=200 for both KL 20I finals and the QA KL final.

### 2026-11-13 ~09:50Z — injected reads of KL finals; QA KL plain BM25; QA->20I KL early lead
- Frozen KL 20I final plain BM25 k=20 53.6 (frozen SFT 52.7). Full: frozen KL 53.6 / 57.5 / 57.5 at k=20/100/200.
- 100I k=100: LoRA KL 67.9 (SFT 61.5), frozen KL 66.2 (SFT 63.7), QA KL 56.0. 200I k=200: LoRA KL 67.1 (SFT 56.1). Text untuned 65.6 / 64.9, tuned 75.9 / 76.5.
- QA KL final plain BM25 k=20/100/200: 46.1 / 49.0 / 49.1 (general-QA KL alone gives no large-k gain).
- QA->20I KL (T_res_rq9L_QA20I_KL) 20I: u0 60.0, u800 64.1, u1600 67.7 vs from-base 19.6 / 62.2 / 63.4 (+4.3 at u1600).
- Stage-2 frozen SFT plain BM25: k=100 u800 51.5, k=200 u800 52.5. LoRA KL held answer_nll u400/u800 1.026 / 1.024.

### 2026-11-13 ~16:20Z — ORCD link restored; CSAIL incident check; stage-2 mid-run; rqm/rqn reclaimed
- ORCD/CSAIL shared ssh: dead ControlPath symlinks (~/.ssh/cm/ngom@{10.1.221.10,slurm-login.csail.mit.edu}:22 -> missing sockets) blocked the user's logins from becoming masters; removed; user logged in again.
- CSAIL security incident (TIG emails, Nov 11-12: backdoors in .bashrc, CLAUDE.md, shared scripts, conda envs on CSAIL NFS). Read-only check, nothing found:
  host startup files unchanged since 2026-08-25; no hooks in ~/.claude or repo .claude; CLAUDE.md / TRESSOIR.md last changed 2026-09-14; all 87 commits since Nov 9 ours;
  no file pulled CSAIL -> host this session (all transfers host -> CSAIL); every §3.1 checkpoint trained on EC2 (DONE hosts); CSAIL home has no startup/agent files and
  .ssh unchanged since 2026-09-18; CSAIL job scripts md5-identical to host copies or to commit 8070764; nothing in the CSAIL venv or code dirs newer than our Nov 9 pushes.
  Caveat: the code loads ac_modules.pt with torch.load(weights_only=False), so CSAIL-produced pairs (incl. the S3 archive csail_scratch_20261109/) must not be loaded unvetted.
- Stage-2 plain BM25 (side scorer b32), k=100 / k=200:
  LoRA SFT u1600 55.4 / 56.9, u2000 57.9 / 59.1; frozen SFT u1200 54.3 / 55.5, u1600 54.9 / -;
  LoRA KL u400 55.4 / 56.1, u800 57.1 / 58.5 (u0 59.6 / 60.9); frozen KL u400 54.8 / 57.3, u800 56.3 / 57.6 (u0 57.5 / 57.5).
- Frozen KL 20I final 200I k=200 63.9 (frozen SFT 59.3). QA->20I KL final 70.6 on 20I (from-base 70.5).
- rqm (T_res_rq9L_kI at u2000 slot) and rqn (T_res_rq9fr_kI_KL at u800 slot) reclaimed: frozen KL resumed on rqg (sky job), new rqo resumes LoRA SFT, new rqp runs the text stage-2 arm (X_rq9_text_t2).

### 2026-11-13 ~17:00Z — user: drop general-QA pre-training; the §3.1 key comparisons; scoreboard
- **User decision: drop the general-QA phase** (QA KL, QA -> 20I) unless something later justifies it (e.g. faster tuning). Basis: QA->20I KL final 70.6 = from-base 70.5; QA KL alone plain BM25 46.1 / 49.0 / 49.1.
- **User: the key §3.1 numbers.** (A) ideal retrieval (golds injected): AC vs UNTUNED text at the same setting; (B) plain BM25: AC at k=100 / 200 vs text at k=20
  (and higher), untuned and LoRA. Expectation: AC at least matches untuned text; vs LoRA text, compare AC k=100 with LoRA text k=20. Not expected to beat LoRA text at equal k.
- `sec35/scoreboard.py` (+ `scoreboard_fetch.sh`): re-scores every per-question output with rq_common.score (all match their summaries), paired deltas with
  95% bootstrap CI and exact McNemar p; writes `sec35/SCOREBOARD.md`. Stage-2 rows = latest scored checkpoint.
- Current (dev1000): A. KL 20I LoRA vs untuned: 20I +1.7 (p=0.11), 100I +2.3 (p=0.04), 200I +2.2 (p=0.07); SFT 20I LoRA 20I +3.5 (p=0.01) but 100I -4.1, 200I -8.8.
  B. KL 20I LoRA k=100 / 200 vs untuned k=20: +10.0 / +11.3; vs LoRA text k=20: +3.3 (p=0.02) / +4.6 (p=0.002); vs LoRA text same k: -1.7 / -3.3.
- Stage-2 ideal-retrieval reads: ORCD 28961025 `jobs/rq_fin_loop.sh` (u01200 midpoints + u02400 finals: 20I, 100I, 200I, plain k=20/100/200; text stage-2 final too);
  host `jobs/rq_fin_feed.sh` sends the EC2 finals (log IB/TMP/PRE_REBUTTAL/fin_feed.log).

### 2026-11-13 ~18:00Z — user: MVP tentatively met (table B); drive the recipe to the end; hill-climb on ideal retrieval vs text + LoRA first
- User: treat the scoreboard (esp. B) as MVP met; drive the recipe to its end with plots; the remaining gap = AC vs text + LoRA on ideal retrieval; any
  hill-climbing targets that first, without affecting what is running. Question: why did §3.5 (4B→4B) match text + LoRA and 2B→9B not — encoder capacity, or 45k vs 90k?
- Evidence (20I dev1000, SFT, reader LoRA; text + LoRA 77.1): 2B enc 44.8k 72.3 (-4.8, p<0.001); 2B enc 90k 75.0 (-2.1; text 90k stopped at u1600 75.1, flat);
  4B enc 44.8k 75.3 (-1.8, p=0.066, paired, scoreboard). §3.5 4B→4B (encoder = reader) 90k: dev 73.8 / test 75.4 vs text 75.7 / 76.1.
  => at 20I both levers each close ~60% of the gap; §3.5 had both (encoder as large as the reader, 90k). At 100I / 200I the gap is mostly length
  generalization (text + LoRA keeps 76-77 at 100I/200I; AC KL drops 2.6 / 3.4 from 20I, SFT 11 / 16) - stage 2 targets that.
- Zero-training 100I / 200I / plain read of the 4B-encoder SFT final: ORCD 28962078 (`jobs/rq_e4_kI_orcd.sh`; rq_eval_job.sh gained RQ_ENCODER, default 2B).
- ~18:30Z: user: own judgment on the 4B-encoder KL hill-climb → launched (likely to help: 4B SFT 20I 75.3 = within 1.8 of text + LoRA, not significant).
  **T_res_rq9e4_20I_KL** = T_res_rq9L_20I_KL (KL to the gold-view teacher, 20I, 44.8k, 5,600 × 8, reader LoRA) with ENCODER Qwen3.5-4B, INIT R_res_rq9e4,
  ENC_LR 2e-5 (5e-5 diverged on 4B); only differences from T_res_rq9e4_e2 = LOSS kl + TARGETS. ORCD 28964211 (mit_preemptable, rtx_pro_6000, --time 48:00:00).
  ORCD refs/refs_train.jsonl replaced by the S3 rq9_gold_refs_all copy (90,447 rows, md5 b4d6f8a2…; old 32,000-row copy kept as .32000_bak).
- First stage-2 ideal read: LoRA KL stage 2 u1200 100I 66.1 (start 67.9).
- ~19:30Z: **Stage-2 KL dips and recovers slowly** (LoRA KL u1200: plain k100/k200 57.5 / 58.0, 100I 66.1, 200I 65.4 vs start 59.6 / 60.9 / 67.9 / 67.1),
  while SFT rises toward KL's start (LoRA SFT u1200 100I 65.7 from 61.5; frozen 65.2 from 63.7). The text stage-2 arm dips too (k20 56.1 → 53.8 at u400).
  Cause hypothesis: stage 2 restarts at the FULL stage-1 peak LR (enc 5e-5, reader 2e-5, warm-up 50) on a converged model.
  User asked whether another stage-2 config is worth trying → **T_res_rq9L_kI_KL_lr4** (1/4 peak: enc 1.25e-5, reader 5e-6; else = T_res_rq9L_kI_KL),
  ORCD 28970512 (init T_res_rq9L_20I_KL final staged to $W/init). Checkpoints handed to the side scorer / final reads by host `jobs/rq_orcd_link.sh`.
- ~20:00Z: **4B-encoder SFT 20I final, zero-training reads** (ORCD 28962078): 100I **67.3**, 200I **65.3**, plain k=100 **55.9** vs 2B SFT 61.5 / 56.1 / 51.9
  (+5.8 / +9.2 / +4.0): the bigger encoder also cuts the long-context drop (20I→200I: 4B -10.0, 2B -16.2). Untuned text 65.6 / 64.9 / 54.8.
  Stage-2 midpoints (u1200): 20I LoRA SFT 68.4 (start 72.3), LoRA KL 67.3 (70.5); 200I LoRA SFT 64.3 (56.1), frozen SFT 63.9 (59.3), LoRA KL 65.4 (67.1).
  Frozen SFT stage 2 final plain k=100 57.1 (b8; start 52.0). Text stage 2 flat (k20 56.1 → 55.3 at u2000). 1/4-LR KL stage 2 u0 59.0 (b8), running.
- ~22:15Z: **Stage-2 SFT finals (late gains as LR decays):** frozen SFT final 100I **70.1** (start 63.7, u1200 65.2), plain k=100 57.1 (b8);
  LoRA SFT final 200I **71.7** (start 56.1, u1200 64.3), plain k=100 58.8 (b8; start 51.9). Text stage-2 final 100I 76.1 (stage-1 text 75.9), plain k=20 56.3 (start 56.1): flat.
  Frozen KL stage 2 u1200 100I 65.6 (start 66.2), plain k=100 / 200 56.5 / 57.7. LoRA KL stage 2 u2000 k=100 / 200 58.5 / 60.3.
  4B KL 20I u800 66.3 (2B KL 62.2). rqp torn down (text done); rqo gone after its run (final already sent).
- ~22:30Z **User framing (durable): a competitive FROZEN-reader AC is a strong paper argument** — frozen AC only changes the reader's INPUT, while
  text + LoRA modifies the reader's weights; so the like-for-like comparison for frozen AC is untuned text, and text + LoRA above us is acceptable.
  Consequences: show frozen arms beside LoRA arms in every table and plot; frame frozen AC against untuned text first. Current: frozen KL 20I final
  within 0.9-3.4 of the LoRA-reader KL; frozen SFT stage-2 final 100I 70.1 vs untuned 65.6. Open: a frozen-reader 4B chain (frozen-reader 4B R, then KL 20I).
- ~22:45Z **Frozen-reader 4B chain started (user: "starting now is better")** on new spot node rqq (role `e4_frozen_kl` in jobs/rq20i_node.sh):
  R_res_rq9e4fr (= R_res_rq9e4 with READER_LR 0; 2,000 x 8 FineWeb, micro 2) then T_res_rq9e4fr_20I_KL (= T_res_rq9e4_20I_KL with READER_LR 0,
  INIT R_res_rq9e4fr; ENC_LR 2e-5, KL, 20I, 44.8k, micro 1). Only difference from the LoRA-reader 4B chain: the reader is frozen. Est. R ~5 h + T ~14 h.
  LoRA KL stage 2 (rqj) finished training 22:26Z; its final reads follow via rq_fin_feed.sh.
- 2026-11-14 ~01:20Z **USER DECISION (durable): §3.1 reports FROZEN-reader AC as the main result.** The reader-LoRA AC arms move to the ablations;
  text + LoRA is shown (if at all) as a higher reference, not the comparison — the reader is not tuned, so "why not tune the reader other ways"
  no longer applies. MVP considered reached. The frozen 4B chain (rqq) stays valuable: matching text + LoRA (even at k=100/200) would be a bonus.
  Frozen SFT stage-2 final (u2400) complete: 20I 71.0 / 100I 70.1 / 200I 68.4 (vs untuned +2.2 p=0.105 / +4.5 p=0.001 / +3.5 p=0.016);
  plain k=20 52.0, k=100 56.9, k=200 58.4 (k=200 vs untuned k=20 +8.8, vs LoRA text k=20 +2.1 p=0.133).
  Frozen KL 20I (no stage 2): ideal retrieval matches untuned (+0.8 / +0.6 / -1.0, all n.s.); plain k=100 vs untuned k=20 +7.9, vs untuned k=100 +2.7 (p=0.023), vs LoRA text k=20 +1.2 (n.s.).
- 2026-11-14 ~01:45Z **User: prioritize the 4B FROZEN runs over the LoRA ones; wait for the frozen KL stage-2 final before choosing KL vs SFT.**
  The frozen 4B chain has its own spot node (rqq) and is not slowed by the ORCD 4B LoRA run. Reads of both 4B KL finals (they need RQ_ENCODER 4B, so
  the 2B final-reads loop cannot do them): jobs/rq_e4_read_orcd.sh PAIR (100I, 200I, plain k=100/200/20; one eval dir per pair, $W/rq_eval_e4x/PAIR),
  submitted by the host loop jobs/rq_e4_feed.sh (frozen checked and submitted first). Scoreboard: 4B KL rows (LoRA, frozen), 4B-vs-2B KL pairs, and
  frozen stage-2 SFT/KL vs frozen KL 20I pairs.
