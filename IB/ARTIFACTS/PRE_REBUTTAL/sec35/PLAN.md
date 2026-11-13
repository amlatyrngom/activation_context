
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
