# Merged32k continuation from constant42k

Prepared for root review and CSAIL H200 staging: `pi42k_merged32k_fresh960_v1`.

Train960 fresh questions once from the exact constant42k paired weights and actual Adam5250. Finish at42,960 unique questions and Adam5370. Each example has one encoder sequence, up to32,768 evidence tokens/64 passages, ratio1/16. Logical batch8, microbatch1, encoder packing65,536, checkpoint threshold0, constant AC5e-6 / reader2e-6. Keep the existing penalty, clipping, frozen source and other scientific settings.

A small `subset.json` selects IDs from the unchanged493MB fresh8k material; no large material rewrite. Exact quotas: code288, math/arXiv115, math/textbook77, finance48, web192, technical web48, documents77, legal58, office57. All original learning32 remain included. Held data are exactly the original selected100 in their existing order,50 per generator; original gold and retrieval objects are reused. Only training rows are interleaved.

Report held100 and learning32 after each96 fresh questions. Measure the actual42k AC baseline before training; initial reader full/no-context controls remain frozen. Moving-reader full/no-context controls are measured at start/final. No inline12 decoding. Save own paired checkpoints every192 questions, and select only completed960×1 final weights. A separate readout manifest will pin the final100 evaluator without modifying the frozen training design.

The host-aware common launcher and CSAIL wrapper use the same command/config. CSAIL initialization changes only two checkpoint paths and adds deployment metadata. The six scientific files—including derived Adam5250—keep their original hashes; README/PRESERVE metadata may differ. Launcher verifies source, subset, model/optimizer bytes and inherited state on compute. `run_root` is explicit and checked.

Files: `bench/run_qa_pi42k_merged32k_v1.py`, `prepare_pi42k_merged32k_v1.py`, `train_pi42k_merged32k_v1.sbatch`, `train_pi42k_merged32k_csail_v1.sbatch`. Prepared design and subset are in [TMP preparation](/workspace/IB/TMP/FIRST_TASK_AWARE_FINETUNING/pi42k_merged32k_preparation/PI42K_MERGED32K_FRESH960_DESIGN_V1.json); root owns final review/freeze and submission. Expected host subset location: `/data/scratch/ngom/activation/taskaware/pi42k_merged32k_fresh960_v1_preparation/`.

Validation:39 focused CPU fixture/AST checks plus actual-source hash, selection, configuration and helper-hash checks passed. Python, shell and embedded launcher Python parse. No remote calls or GPU submissions were performed by this preparation agent. Trainer bound9,000seconds; allocation3h; actual setup/compute time remains unmeasured on H200.

[Receipt and hashes](PI42K_MERGED32K_PREPARATION_V1.json) · [focused delta](/workspace/IB/TMP/FIRST_TASK_AWARE_FINETUNING/pi42k_merged32k_preparation/delta.patch)
