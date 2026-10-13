# ORCD H200 final960 readout

Prepared and CPU-checked; **not submitted**. New launcher `bench/evaluate_pi42k_merged32k_orcd_h200_v1.sbatch` uses versioned `evaluate_pi42k_merged32k_v2.py`. The launcher workflow differs from CSAIL only in manifest/output/worker names. New independent manifest: `bench/PI42K_MERGED32K_ORCD_H200_READOUT_MANIFEST_V1.json`; all previous helper pins remain unchanged.

The45-minute ORCD H200 allocation preserves the three-cell sequence: own final32k100 actual AC answers and three-view NLL (1850s), same-allocation parent42k native2k NLL (300s), then final42960 native2k NLL (300s). The existing exact input/final-panel replay, own complete120/all615Adam5370 paired-final guards, actual token-ID/EOS decoder and matched retention checks remain intact. Training designs/source and the originalV1 evaluator/CSAIL launcher/manifest are unchanged. V2 explicitly reuses the actual completed training result’s `autotuning.logits_chunk_tokens` for final32k, parent2k and final2k, validates the resolved value, and records it in identity/runtime policy. Missing, invalid or resolved-different values fail before scoring; all1e-3 replay gates remain unchanged.

It uses isolated root `/home/ngom/orcd/pool/activation/taskaware/pi960_h200_v1`, the shared `taskaware/bench` symlink and `taskaware/train_ws_night_v1`, and explicitly defaults HF_HOME to the complete `orcd/scratch/activation/hf` cache. Output: `pi960_h200_v1/reports/pi42k_merged32k_orcd_h200_cohort100/`.

Proposed submission after root stages/verifies the manifest's helper hashes:

```bash
sbatch --parsable --hold --dependency=afterok:23558386 \
  --export=ALL,HF_HOME=/home/ngom/orcd/scratch/activation/hf \
  --output /home/ngom/orcd/pool/activation/taskaware/logs/pi960_h200_final100_%j.log \
  /home/ngom/orcd/pool/activation/taskaware/bench/evaluate_pi42k_merged32k_orcd_h200_v1.sbatch \
  /home/ngom/orcd/pool/activation/taskaware/bench/PI42K_MERGED32K_FRESH960_ORCD_H200_DESIGN_V1.json
```

Root can inspect and release the held job. `afterok` does not bypass the evaluator's complete-final checks if training stops early.

39 focused checks and shell syntax pass: preserved workflow and original guard ASTs, helper hashes, actual frozen ORCD design/initial pair, paths/cache/allocation, retention, stale final clock, missing/invalid training chunks and resolved-policy mismatch. A CPU fixture executes the actual autotuner tail: automatic chunks change128→512 between8GiB/64GiB free in the fixture; an explicit256 stays256. Exact hashes, proposed argv and evidence are in [the preparation receipt](PI42K_MERGED32K_ORCD_H200_READOUT_PREPARATION_V1.json). No remote calls, checkpoint inspection or GPU work was performed.

Why the fix: the current autotuner derives automatic head chunks from live free CUDA memory (`free/16`, capped2GiB, then power-of-two rounding). The prior evaluator left this choice unbound. Training itself resolves before Adam restoration, and a large H200 may hit the same cap in both cases, so no actual mismatch is claimed. Pinning the measured training choice removes that avoidable numerical uncertainty.
