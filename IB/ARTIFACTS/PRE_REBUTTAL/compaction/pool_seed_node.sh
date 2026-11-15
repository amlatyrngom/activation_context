#!/bin/bash
# New seeds of the 9B training-side pools (seed 0 = pc_pool9_h6_trA/_trB/_trC1/_trC4 + pc_pool9_h8_D1/_D2) on one SkyPilot
# spot node (1x RTX PRO 6000), run as a sky job. Same command as the seed-0 pools (kit/pool9h6_node.sh -> kit/pool_node.sh ->
# kit/run_pool.py --model-id Qwen/Qwen3.5-9B --max-model-len 72000 --split split_train_shuffled.json, 40 agents per GPU),
# only --seed and the cache id differ, and the KV cache is bf16 (run_pool_seed.py). Resumable across spot reclaims: each
# cache is restored from S3 when the node has none, mirrored to S3 every 10 min (a snapshot cut at the last full line),
# and marked DONE on S3 when it holds every task of the shard.
#   pool_seed_node.sh <h0|h1> <seed> [seed ...]      h0 = training order 0-1337, h1 = 1338-2675
#   cache ids: pc_pool9_tr_s<seed>_<h0|h1>   ->  S3 .../compaction/pool_seeds/ROLLOUTS/<cache id>/rollouts.jsonl
set -uo pipefail
SHARD=$1; shift
case $SHARD in h0) START=0;; h1) START=1338;; *) echo "shard must be h0 or h1" >&2; exit 2;; esac
COUNT=1338
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/pool_seeds
A=/root/activation_artifacts; P=$A/pool_seeds; PY=/opt/activation/.venv/bin/python
mkdir -p $P/logs $P/inputs $P/stage $A/SYNC/ROLLOUTS $A/hf
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)   # the image can be older than the lock
[ -x /root/s3venv/bin/python ] || { /usr/local/bin/uv venv -q /root/s3venv && /usr/local/bin/uv pip install -q --python /root/s3venv/bin/python boto3; }
S3PY=/root/s3venv/bin/python
$S3PY -c "import boto3; boto3.client('s3').download_file('general-bucket-aln', 'activation-context-data/pre_rebuttal_20261109/compaction/pool_seeds/inputs/s3x.py', '$P/s3x.py')"
X() { $S3PY $P/s3x.py "$@"; }
X getdir $S3/inputs/ $P/inputs/
LOG=$P/logs/node_$SHARD.log
say() { echo "$(date -u +%FT%TZ) $SHARD: $*" | tee -a $LOG; }

export HF_HOME=$A/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TZ=UTC VLLM_WORKER_MULTIPROC_METHOD=spawn
export ACTIVATION_SYNC_ROOT=$A/SYNC PYTHONPATH=/root/sky_workdir:$P/inputs POOL_KV_CACHE_DTYPE=auto
export PATH=/opt/activation/.venv/bin:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset HF_TOKEN                                          # public model and dataset (as sec35/jobs/ec2_prep.sh)
for t in 1 2 3; do timeout 1800 $PY -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-9B', token=False)" > /dev/null && break; done

rows() { [ -s "$1" ] && $S3PY -c "
import json, sys
keys = set()
for line in open(sys.argv[1]):
    try: keys.add(json.loads(line)['config_key'])
    except Exception: break
print(len(keys))" "$1" || echo 0; }
# trim SRC DST: copy (or rewrite in place) up to the last complete line
trim() { $S3PY -c "
import sys
data = open(sys.argv[1], 'rb').read(); data = data[:data.rfind(b'\n') + 1]
open(sys.argv[2], 'wb').write(data)" "$1" "$2"; }
mirror() {                                              # snapshot -> S3 (never a half-written line), plus logs and pool.json
  local c=$1 f=$A/SYNC/ROLLOUTS/$1/rollouts.jsonl
  [ -s $f ] && mkdir -p $P/stage/$c && trim $f $P/stage/$c/rollouts.jsonl && X put $P/stage/$c/rollouts.jsonl $S3/ROLLOUTS/$c/rollouts.jsonl > /dev/null 2>&1
  [ -f $A/SYNC/PAPER_COMPACTION/$c/pool.json ] && X put $A/SYNC/PAPER_COMPACTION/$c/pool.json $S3/ROLLOUTS/$c/pool.json > /dev/null 2>&1
  X putdir $P/logs $S3/logs/ > /dev/null 2>&1
}

for SEED in "$@"; do
  C=pc_pool9_tr_s${SEED}_$SHARD; F=$A/SYNC/ROLLOUTS/$C/rollouts.jsonl
  if $S3PY $P/s3x.py get $S3/ROLLOUTS/$C/DONE $P/stage/$C.DONE > /dev/null 2>&1; then say "$C already DONE on S3"; continue; fi
  mkdir -p $A/SYNC/ROLLOUTS/$C
  if [ ! -s $F ]; then X get $S3/ROLLOUTS/$C/rollouts.jsonl $F > /dev/null 2>&1 && say "$C restored from S3: $(rows $F) rows"; fi
  [ -s $F ] && trim $F $F                               # a torn last line would hide every row appended after it
  ( while sleep 600; do mirror $C; done ) & LOOP=$!
  for attempt in 1 2 3; do
    say "$C attempt $attempt: tasks $START+$COUNT seed $SEED, $(rows $F) cached"
    $PY $P/inputs/run_pool_seed.py --split $P/inputs/split_train_shuffled.json --side training --start $START --count $COUNT \
      --caching-id $C --model-id Qwen/Qwen3.5-9B --seed $SEED --max-model-len 72000 --agents-per-gpu 40 >> $P/logs/$C.log 2>&1
    say "$C attempt $attempt exit $?, $(rows $F) rows"
    [ "$(rows $F)" -ge $COUNT ] && break
    [ -s $F ] && trim $F $F
  done
  kill $LOOP 2>/dev/null; mirror $C
  if [ "$(rows $F)" -ge $COUNT ]; then
    date -u +%FT%TZ > $P/stage/$C.DONE && X put $P/stage/$C.DONE $S3/ROLLOUTS/$C/DONE > /dev/null && say "$C DONE"
  else
    say "$C incomplete after 3 attempts"
  fi
done
say "all seeds ended"; X putdir $P/logs $S3/logs/ > /dev/null 2>&1
