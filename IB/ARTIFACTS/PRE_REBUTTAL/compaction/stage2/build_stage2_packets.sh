#!/bin/bash
# Compaction stage-2 packets (README.md here, user-approved): plain cross-entropy on the shortest correct continuations, one packet per ratio.
#   bash build_stage2_packets.sh TAG MATERIAL [R ...]      CPU, this host; R default "0.2 0.1"
#     MATERIAL: select_material.py's output (IB/TMP/PRE_REBUTTAL/compaction_stage2/material/<name>.jsonl)
#     -> IB/TMP/PRE_REBUTTAL/compaction_stage2/packets/s2ce_<TAG>_r{5,10}/  (r 0.2 -> r5, 0.1 -> r10), each audited (audit_stage2_packet.py)
# Packet: prepare_agent_continuation.py --loss sft, every item at the one r (rows = 20,000 x r), sides training + evaluation, the 322
# (three frozen index files) and the 528 (dev_expand/exclude_h7_dev206.json) excluded, no training-side held panel (dev-75 is the
# in-run panel at launch), encoder 5e-5, reader 0, warm-up 50, cosine (trainer default), batch 8, one pass, checkpoint every 25.
# The packet is built against R_cmp_lcrec_final (CPU: tokenizers and row limits only, the same architecture as every stage-1 final);
# the training start pair is the ratio's stage-1 final, given at launch (CMP_INIT=<RUN>_final). CMP_WORKERS (default 4). S3 backup
# (compaction/stage1/packets/<NAME>/, where cmp_stage1_ec2.sh reads packets) with CMP_S3=1.
set -euo pipefail
TAG=${1:?TAG}; MATERIAL=${2:?MATERIAL}; shift 2; RS=${*:-0.2 0.1}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $HERE/../../../../.. && pwd)
T=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage2; PY=$REPO/.venv/bin/python; K1=$HERE/../stage1
NS=$REPO/IB/TMP/PAPER_COMPACTION/node_sync
X528=$T/exclude_h7_dev206_index.json            # the 528 task ids in the index format --exclude-tasks reads
[ -f $X528 ] || $PY -c "import json,sys; d=json.load(open(sys.argv[1])); json.dump({'source': sys.argv[1], 'prefixes': [{'task_id': t} for t in d['task_ids']]}, open(sys.argv[2], 'w'))" \
  $HERE/../dev_expand/exclude_h7_dev206.json $X528
EXCLUDE=($NS/prefixes_dev9_75/index.json $NS/pc-a_live/pc/prefixes_eval9c_200/index.json $NS/prefixes_eval_100/index.json $X528)
I=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1/init/R_cmp_lcrec_final
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1   # its packets/: where the EC2 / cluster jobs read packets
mkdir -p $T/packets $T/logs
for R in $RS; do
  case $R in 0.2) L=r5;; 0.1) L=r10;; *) echo "r $R: 0.2 or 0.1" >&2; exit 2;; esac
  NAME=s2ce_${TAG}_$L; O=$T/packets/$NAME
  [ -e $O ] && { echo "$O exists: use a new TAG" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES= PYTHONPATH=$REPO CONTINUATION_BENCH_DIR=$REPO/IB/TMP/PAPER_CORPUS_QA/kit/bench $PY $K1/prepare_agent_continuation.py \
    --material $MATERIAL --output $O --loss sft --sides training evaluation --exclude-tasks "${EXCLUDE[@]}" \
    --initial-ac $I/ac --initial-reader $I/reader --held-repos 0 --r-values $R \
    --lr-ac 5e-5 --lr-reader 0 --train warmup_updates=50 --checkpoint-every 25 --workers ${CMP_WORKERS:-4} > $T/logs/${NAME}_prepare.log 2>&1
  $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print('$NAME:', d['training']['items'], 'items,', d['training_runs'], 'runs,', d['training_tasks'], 'tasks,', d['updates_per_pass'], 'updates per pass (batch 8), r', d['training_r_counts'], 'dropped', d['dropped'], 'excluded', d['excluded'])" $O/continuation_build.json
  PYTHONPATH= $PY $HERE/audit_stage2_packet.py --packet $O --material $MATERIAL --r $R --exclude "${EXCLUDE[@]}" $HERE/../dev_expand/exclude_h7_dev206.json --out $O/audit_stage2.json
  if [ "${CMP_S3:-0}" = 1 ]; then aws s3 cp --recursive --quiet --exclude prepared_examples.pkl $O/ $S3/packets/$NAME/; fi
done
