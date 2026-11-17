#!/bin/bash
# Compaction stage-1 RETRAIN packets (2026-11-17, corrected KL-teacher view: TEACHER_VIEW_FIX.md). CPU, this host.
# Material: build_stage1.sh's export step (export_pool_material.py of 2026-11-17, with prefix_assistant_token_ids) of all 13 pool caches
# (POOLS=pools_all_s012: seeds 0-2, both split sides, none of the 322) at the cuts 16k / 20k / 24k, written as
#   material/s1kl_A_c162024_s012_tv_v1_c{16,20,24}.jsonl   (build_stage1.sh s1kl_A_c162024_s012_tv_v1 "16 20 24"; its one-packet
#   prepare step ran out of host memory at 13,451 items, so the packets are prepared here one cut at a time, same arguments).
# Packets (full continuations, teacher_view recorded), one per cut:
#   packets/s1kl_tv_c{16,20,24}_s012_v1      prepare_agent_continuation.py --loss kl, r from {1/5, 1/10}, sides training+evaluation,
#                                            the 322 excluded, start pair R_cmp_lcrec_final, encoder 5e-5, reader 0, warm-up 50
# Phase B = packets/s1kl_tv_c20_s012_v1 as it is. Phase A = the three merged and cut to the first T continuation tokens:
#   truncate_packet.py --packet <the three> --tokens T --out packets/s1kl_A_tv_c162024_s012_t<T>_v1
#   bash build_retrain_packets.sh [CUTS]     (default "20 16 24"; a cut whose packet exists is skipped). CMP_WORKERS (default 3).
# Per-ratio packets (2026-11-17): CMP_R_VALUES=0.2 CMP_TAG=tv_r5 -> packets/s1kl_tv_r5_c{c}_s012_v1, every item at that one r (the same
#   prefixes, item ids and continuations as the {1/5, 1/10} packets; --r-values with one value draws it for every item).
set -euo pipefail
CUTS=${1:-20 16 24}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $HERE/../../../../.. && pwd)
T=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1; PY=$REPO/.venv/bin/python
NS=$REPO/IB/TMP/PAPER_COMPACTION/node_sync
EXCLUDE=($NS/prefixes_dev9_75/index.json $NS/pc-a_live/pc/prefixes_eval9c_200/index.json $NS/prefixes_eval_100/index.json)
I=$T/init/R_cmp_lcrec_final                     # s3://.../compaction/stage1/runs/R_cmp_lcrec/sync/ckpt/final (241/241 updates)
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1
for c in $CUTS; do
  NAME=s1kl_${CMP_TAG:-tv}_c${c}_s012_v1; M=$T/material/s1kl_A_c162024_s012_tv_v1_c$c.jsonl; O=$T/packets/$NAME
  [ -f $O/items.pkl ] && { echo "$NAME exists: skipped"; continue; }
  [ -e $O ] && { echo "$O exists without items.pkl: use a new name" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO CONTINUATION_BENCH_DIR=$REPO/IB/TMP/PAPER_CORPUS_QA/kit/bench $PY $HERE/prepare_agent_continuation.py \
    --material $M --output $O --loss kl --sides training evaluation --exclude-tasks "${EXCLUDE[@]}" \
    --initial-ac $I/ac --initial-reader $I/reader --held-repos 0 --r-values ${CMP_R_VALUES:-0.2 0.1} \
    --lr-ac 5e-5 --lr-reader 0 --train warmup_updates=50 --workers ${CMP_WORKERS:-3} > $T/logs/${NAME}_prepare.log 2>&1
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('$NAME:', d['training']['items'], 'items,', d['training_runs'], 'runs,', d['training_tasks'], 'tasks,', d['updates_per_pass'], 'updates per pass (batch 8), r', d['training_r_counts'], 'dropped', d['dropped'], 'excluded', d['excluded'])" $O/continuation_build.json
  aws s3 cp --recursive --quiet --exclude prepared_examples.pkl $O/ $S3/packets/$NAME/
done
