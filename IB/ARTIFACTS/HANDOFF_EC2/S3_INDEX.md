# What moved to S3 on 2026-11-09

Root: `s3://general-bucket-aln/activation-context-data/` (≈1.97 TB before this move; older prefixes such as
`sec45_20261026/`, `rq_runs/`, `am_lh5_20261019/`, `container_archive_2026101{5,24}/` hold the October runs and are
indexed in `IB/CANON/PAPER_CANON.md`).

| Prefix | What | Size | Verified |
|---|---|---|---|
| `container_final_20261109/` | The VS Code container's unique data before it was removed: `home_claude/` (its `~/.claude`, incl. the paper session's job folder `jobs/6f1c1373/tmp`, 28 GB, and transcripts), `tmp_claude/-workspace/` (task outputs), `workspace/` (`TMP/`, `.claude/`, `kl_rows.json`, `RESEARCH_DESIGN.md`, `.devcontainer/`), `transcripts/paper_session_6f1c1373.jsonl.gz` | 46.8 GB, 218,851 files | every file present with equal size |
| `host_ib_tmp_20261109/` | All of `IB/TMP/` (same relative paths: `IB/TMP/X` → `host_ib_tmp_20261109/X`), symlinks skipped | 265.9 GB, 443,951 files | every file present with equal size (4 files edited afterwards were re-synced) |
| `ib_large_artifacts_20261109/` | The 11 `IB/ARTIFACTS/FIRST_TASK_AWARE_FINETUNING/*_COHORT100.{evidence.json,tressoir.html}` files over 5 MB (git-ignored; still on disk) | 229 MB | 11 objects |

Deleted locally after verification (small text files under 1 MB kept in place): `IB/TMP/SYNC/{AC_RECON, AGENT_TRAINING_TEST,
TASK_AWARE_QA, AC_RECON_ORCD, DAPO_BENCH, RAG_AC_POC}`, `IB/TMP/{AGENT_AC_PRETRAINING, INPUT_EMBEDS_UV, INPUT_EMBEDS_RESEARCH,
AGENT_ROLLOUTS}` (≈64 GB). The other `IB/TMP` folders stay on disk while the pre-rebuttal work uses them; prune them the same way
(`verify_prune.py`-style: list S3 sizes, compare every file, then delete) when they are no longer needed.

The old container (`affectionate_sammet`) and two stopped `hello-world` containers were removed, plus 5.3 GB of Docker build cache.
Disk went from 402 GB to 248 GB used.
