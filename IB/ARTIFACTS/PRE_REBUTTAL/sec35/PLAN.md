
### 2026-11-13 ~04:50Z — KL plain-BM25 gain replicates on frozen reader
- T_res_rq9fr_20I_KL final: 20I dev1000 69.6 (frozen SFT 71.9).
- Plain BM25 k=100 (rqn stage-2 u0, b8): **57.3** vs frozen SFT 52.0 (side b32). LoRA: KL 59.6 vs SFT 51.9.
- LoRA KL 20I final plain BM25 k=20: 50.9 (SFT 52.0) — KL gain is at large k only (k=20 50.9, k=100 59.6, k=200 60.9).
- Stage-2 LoRA SFT u400 k=100 50.8 (u0 51.9); held answer_nll u1200 0.698 (u0 0.823).
- QA KL u10400 60.3.
