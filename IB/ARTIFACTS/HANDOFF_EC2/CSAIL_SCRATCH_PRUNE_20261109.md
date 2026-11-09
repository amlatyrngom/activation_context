# CSAIL scratch prune, 2026-11-09

Folder: `/data/scratch-fast/ngom` on CSAIL (`nfs-prod-aia-1:/scratch/scratch`, 25 TB shared by everyone, 100 % full).
Goal: free at least 500 GB without losing anything that existed only there. The user asked for it ("Free up our scratch if needed").

**Free space on `/data/scratch-fast`: 59 GB at the start (53 GB just before the first delete) → 693 GB free at 23:40 UTC (`df -h`: 24T used of 25T, 98 %); NFS frees deleted blocks lazily, so the figure was still rising after.**
Because the filesystem is shared, free space also moves with other users' writes. Our own usage dropped from about 856 GB to 194 GB: `activation` 162 G and `sec45` 32 G. The new `sec35/` (29 G) is not counted; the main session created it during the prune (`du`).

New S3 prefix: `s3://general-bucket-aln/activation-context-data/csail_scratch_20261109/<folder>/<same relative path>`
(relayed through the EC2 host, because CSAIL has no S3 credentials). Per-file manifests are in
`csail_scratch_prune_20261109/<folder>.plan.tsv.gz` next to this file, with a copy at `csail_scratch_20261109/_manifests/`.
Their columns are `rel size inode action reason detail`. Actions:
- `relay`: now at `csail_scratch_20261109/<folder>/<rel>`;
- `del_s3`: `detail` gives the existing S3 key;
- `del_link` / `del_dup`: `detail` gives the relayed twin;
- `del_regen` / `del_inter`: regenerable;
- `keep`.

## Method

1. `find DIR -type f -printf '%P\t%s\t%i\t%n'` on CSAIL for every top-level folder (path, size, inode, link count), plus one
   S3 listing of the whole `activation-context-data/` root (913,642 objects).
2. **On S3**: a file counts as on S3 when an object has the same size and its key ends in the file's path from the run folder
   down (for example `sec45/runs/R_in/...` → `sec45_20261026/runs/R_in/...`). Three renamed copies were mapped explicitly and
   checked by exact key and size: `rq_v1/sync/tA_s2` → `rq_runs/20261021/trackA/tA_s2_epoch004/tA_s2/{AC_MODELS,LORAS}/epoch_004`,
   `paper_corpus_qa_v1/runs/K15/fresh_pair` → `corpus_hillclimb_20261019/K15_fresh_pair`, and the K15 `items.pkl` /
   `prepared_examples.pkl` → `container_archive_20261024/jobtmp_6f1c1373/hill_xfer/K15_packet_inputs/`.
3. **Hard links** (`latest/` = `epoch_N/`, `AC_MODELS`/`LORAS` = `ckpt/final`): covered when another link to the same inode is on S3
   or was relayed. Only one path per inode was relayed. On S3 it is usually the `latest/` path; the manifest lists the other names.
4. **Identical copies**: md5 was computed on CSAIL for the large derived files that repeat across runs (teacher targets, the
   `*_tables/final.pt` vs `special_inputs.pt`, `reader_final` vs `LORAS`). Only one copy per md5 was relayed; the manifest names it.
5. **Relay**: `rsync` over the user's existing CSAIL connection (the `csail_ssh.sh` transport options) into
   `/home/ubuntu/csail_relay_20261109/` on the EC2 host, `aws s3 cp --recursive`, then a check that every listed file is on S3 with the
   CSAIL size. The local copy was deleted after the check. Chunks were at most 45 GB, and the host disk stayed below 80 %.
6. **Delete on CSAIL** only after steps 2–5 had a clean result for every file in that folder. `squeue -u ngom` was checked first.

Regenerable material deleted without an S3 copy:
- `optimizer.pt`;
- intermediate `epoch_*_step_*` checkpoints, but only in runs whose final checkpoint is on S3 or was relayed;
- superseded partial attempts whose later attempt completed;
- `*.tmp` checkpoint folders;
- triton, vllm, tilelang and `AUTOTUNE` caches;
- unpacked code whose tarball is on S3.

## Per folder

| Folder | Before (du) | Already on S3 (prefix) | Relayed to `csail_scratch_20261109/` | Deleted on CSAIL | Left in place |
|---|---|---|---|---|---|
| `sec45/` | 204 G | `sec45_20261026/runs/` (R_in/kv/res, S_*, T_*, P0_*) and `host_ib_tmp_20261109/PAPER_SEC45/runs/` (all `*_v4*` runs); `sec45_20261026/{code,data}`, `container_final_20261109/.../sec45/hf_relay` | 126 small files (1 MB): the per-run `completed_epoch.json`/`partial_epoch.json`, the `S_in.killed_2863227` logs, all `resume_test` logs/JSON (incl. `resume_test/RESULT.json`), the `s45_*.out` Slurm logs, `orch_v4/*.submits` | `runs/` (every file on S3 or a hard link of one; the 22 small files missing from S3 relayed first), `triton_cache/`, the `resume_test` checkpoints (> 1 MB: 42 GB of `.pt`/`.safetensors`, optimizer and `.tmp` slots) | `hf/` (Qwen3-4B; the running §3.5 job reads it), `init/`, `data/`, `code/`, `jobs/`, `orch_v4/`, `home/`, top-level `*.out`, `resume_test/` small text (32 G total) |
| `paper_corpus_qa_v1/` | 15 G | `corpus_qa_20261019/K15/` (pair, packet JSON, logs, evals), `corpus_hillclimb_20261019/K15_fresh_pair/`, `container_archive_20261024/.../K15_packet_inputs/` | 7 small files: `gpu_*.csv`, `start_pair.json`, `packet/report/diagnostics/*`, `completed_epoch.json` | whole folder (its 5.3 GB `optimizer.pt` regenerable; `packet/SNAPSHOTS` was only a symlink to `/data/scratch/...`, which was not touched) | nothing |
| `am_ec2_out/` | 237 G | `am_lh5_20261019/run2/cqa-ec2b2/cont/` (`cont/C_*_tables`), `.../cqa-ec2r/` + `container_final_20261109/.../uall/` (`export/t0.1_fastest-u-all`), `rq_m1` data in `container_final`/`host_ib_tmp`, some job scripts | 38.6 GB, 215 files: finals of the CSAIL-trained runs **`R_deepS_all20_lr1e3`** (paper Fig. 4 "final R_deepS, CSAIL H200"), `L_G2n`, `L_deepS`, `L_rows` (`*_tables/final.pt` + `reader_final`, `sync_*/AC_MODELS/rag_poc/latest/{ac_modules,special_inputs,teacher_targets}.pt`, `side_lora`, `LORAS/.../latest`, `no_context_reference.pt`, run/stats JSON), all `curves/` evals, `.log`/`.out`, `jobs/`, `vfull/vfull39.out`, `rq_m1/{rq_data2.out,teacher_reads_nothink.jsonl}` | whole folder: 141 GB intermediate `epoch_001_step_*` tables/readers (the u500–u2000 curve points; their evals are in `curves/`) and superseded partial attempts (`sync_L_G2n_*_r1`, `sync_L_rows_*_r0`, `sync_L_deepS_*_r1`), 49 GB `optimizer.pt`, 24 GB md5-identical copies, caches | nothing |
| `rq_v1/` | 240 G | `rq_runs/20261021/` (`sync_in/real_gold` = `trackA_real/armG_final/real_gold`, `sync_in/recon2_s2`, `sync/recon2_s1_u1500`, `tA_s2` final), `container_final_20261109/.../pr/code/` and others for `code_rq*`, `jobs/`, `kit*/`, `refs/`, `out/` | 92.1 GB, 532 files: final checkpoint (`ac_modules.pt`, `side_lora`, reader LoRA, `no_context_reference.pt`, run/stats JSON) of every run in `sync/` (hc_*, speed*, tA_*, easy_*, real_same, sftd_dense_r10b, tA_trkB_*); `full_nt_p75m_e5e5_r20` (unfinished: its latest slot `epoch_002_step_00500` from `sync/` and `ckpt/`); all of `teacher/` (22 GB); one copy of each distinct `teacher_targets.pt` (5 copies, 11 GB, standing for 19 files); small logs/scripts (`out/`, `jobs/`, `kit*/`, `env3.sh`, `reads_nt/`, `tools/bin`) | whole folder: 103 GB `optimizer.pt`, 63 GB md5-identical `teacher_targets.pt` copies (the 9 hc/speed runs share one 7.17 GB file, the 6 tA_* runs one 1.01 GB file), `ckpt/full_nt.../epoch_002_step_00250`, triton cache, unpacked `code_rq*/` | nothing |
| `activation/` | 162 G | `node_final_20261018/pc-a/pc/pairs/*_final` (all 13 compaction pairs), `host_ib_tmp_20261109/PAPER_TRAJ_QA/round2/` (material, plans, `pairs_m1`), `container_final_20261109/.../{h7,h8a,h8full}` (material) | nothing | only the triton/vllm caches (0.3 GB) | **everything else (≈ 162 G)**; see below |
| `resubmit_fast.sh` | 5.5 K | | | | kept |

`sec35/` was created by the main session at 17:44 CSAIL time while this prune ran. Job 3010811 (`s45_S_kv_fr90k`) runs in it and
reads only `sec45/hf` from `sec45/`, so it was not touched. The md5 helper files in `/data/scratch-fast/ngom/.prune_tmp/` were removed at the end. Their results are in
`csail_scratch_prune_20261109/md5_csail.txt`.

### Why `activation/` was left in place

It holds paper evidence that exists only on CSAIL:
- the compaction packets `paper_compaction_v1/packets/*` (incl. `h8a_from_h5`, whose `recipe.json` PAPER_CANON lists as CSAIL-only);
- `material_h8*`;
- the trajectory-QA round-2 packets (`paper_traj_qa_v1/r2/packets`, 41 GB, not on S3);
- `synth_add_v1`.

Their final `AC_MODELS`/`LORAS` slots are separate files from the `pairs/*_final` on S3. Equal sizes do not prove equal content,
because every 2B LoRA has the same size. The 4.2 GB `optimizer.pt` per packet (34 GB in total) is safe to delete only once that
equality is checked.

The freed space already exceeded the target, so this folder was not relayed (about 150 GB). To prune it later:
1. md5 each packet's `AC_MODELS/rag_poc/latest/*` against its `pairs/<name>_final` on CSAIL.
2. Delete the `optimizer.pt` files and the `SNAPSHOTS` of packets whose final matches.
3. Relay the rest to `csail_scratch_20261109/activation/`.

## Totals

- Relayed: 130.7 GB in 880 files (am_ec2_out 38.6, rq_v1 92.1, small files from sec45 and paper_corpus_qa_v1 1 MB).
- Deleted on CSAIL: about 662 GB: 856 GB before, 194 GB after (excluding `sec35`). By folder: sec45 172, paper_corpus_qa_v1 15, am_ec2_out 237, rq_v1 240, activation caches 0.3 (du), across `sec45/{runs,triton_cache,resume_test checkpoints}`, `paper_corpus_qa_v1/`,
  `am_ec2_out/`, `rq_v1/` and the `activation` caches.
