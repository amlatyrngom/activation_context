#!/bin/bash
# Math recipe dev_hard (2026-11-18): the three untuned 9B solver runs of every dev_hard problem on a Sky node with the Podman image
# (dvA, the coordinator's eval node; one sky job, 1 GPU), the same setup that made dmhard and the M1 v2 pool: kit/run_tqa_solve_v2.py
# --caching-id ID --seeds 3 (Qwen/Qwen3.5-9B, single agent, thinking on, cap 40k, 40 agents per GPU, max_model_len 56k).
# Spot-safe: the rollout cache (ROLLOUTS/ID/rollouts.jsonl, one line per finished (task, seed); the rollout manager skips cached
# pairs on restart) is restored from S3 when the node has none and mirrored to S3 every 5 min (snapshot cut at the last full line);
# DONE on S3 when every (task, seed) is there. Started by devhard_launch.sh, which stages this folder's files on S3 first.
#   bash devhard_solve_node.sh ID EXPECTED_ROWS
set -uo pipefail
ID=$1; EXPECTED=$2
A=/root/activation_artifacts; D=$A/math_devhard; HERE=$D/code; SYNC=$A/SYNC
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/math/dev_hard
PY=/opt/activation/.venv/bin/python; S3PY=/root/s3venv/bin/python
X() { $S3PY $HERE/s3x.py "$@"; }
mkdir -p $D/logs $D/stage $SYNC/ROLLOUTS/$ID $SYNC/PAPER_TRAJ_QA/$ID
LOG=$D/logs/node.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }
F=$SYNC/ROLLOUTS/$ID/rollouts.jsonl
rows() { [ -s $F ] && grep -c . $F || echo 0; }
mirror() {
  [ -s $F ] && $S3PY -c "
import sys
data = open(sys.argv[1], 'rb').read(); data = data[:data.rfind(b'\n') + 1]
open(sys.argv[2], 'wb').write(data)" $F $D/stage/rollouts.jsonl && X put $D/stage/rollouts.jsonl $S3/ROLLOUTS/$ID/rollouts.jsonl > /dev/null 2>&1
  X putdir $D/logs $S3/logs/ > /dev/null 2>&1; }

# the sample: this host's frozen draw (sha256 checked against the copy launched with it)
[ "$(sha256sum < $HERE/sample.json | cut -c1-64)" = "$(cut -c1-64 $HERE/sample.json.sha256)" ] || { say "sample sha256 mismatch"; exit 2; }
if [ -f $SYNC/PAPER_TRAJ_QA/$ID/sample.json ]; then cmp -s $HERE/sample.json $SYNC/PAPER_TRAJ_QA/$ID/sample.json || { say "a different sample.json is already on the node"; exit 2; }
else cp $HERE/sample.json $SYNC/PAPER_TRAJ_QA/$ID/sample.json; fi
if [ ! -s $F ]; then X get $S3/ROLLOUTS/$ID/rollouts.jsonl $F > /dev/null 2>&1 && say "restored $(rows) rows from S3" || say "no S3 snapshot: fresh start"; fi
[ "$(rows)" -ge $EXPECTED ] && { say "already complete ($(rows) rows)"; mirror; echo done > $D/DONE; X put $D/DONE $S3/ROLLOUTS/$ID/DONE > /dev/null; exit 0; }

# environment as the paper compaction kit's node_env.sh (HF cache shared with the node's other jobs when present)
HFH=$A/pc/hf; [ -d $HFH ] || HFH=$D/hf
export HF_HOME=$HFH PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC VLLM_WORKER_MULTIPROC_METHOD=spawn ACTIVATION_SYNC_ROOT=$SYNC
export PYTHONPATH=/root/sky_workdir:$HERE PATH=/opt/activation/.venv/bin:/usr/local/cuda/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd /root/sky_workdir
command -v podman > /dev/null || { say "no podman on this node"; exit 2; }
say "start on $(hostname): $ID, $(rows)/$EXPECTED rows; GPU $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1)"
( while sleep 300; do mirror; say "mirror: $(rows)/$EXPECTED rows"; done ) &
MIRROR=$!
trap 'kill $MIRROR 2>/dev/null; mirror' EXIT
$PY $HERE/run_tqa_solve_v2.py --caching-id $ID --seeds 3 >> $D/logs/solve_$ID.log 2>&1
RC=$?
kill $MIRROR 2>/dev/null; mirror
say "solve exit $RC; $(rows)/$EXPECTED rows"
if [ "$(rows)" -ge $EXPECTED ]; then
  for f in $SYNC/PAPER_TRAJ_QA/$ID/*; do [ -f $f ] && X put $f $S3/PAPER_TRAJ_QA/$ID/$(basename $f) > /dev/null 2>&1; done
  echo "$(date -u +%FT%TZ) $(rows) rows" > $D/DONE; X put $D/DONE $S3/ROLLOUTS/$ID/DONE > /dev/null; say "DONE"
fi
exit $RC
