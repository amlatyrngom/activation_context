
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
