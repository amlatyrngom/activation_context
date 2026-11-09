#!/bin/bash
# rq-fresh3 (≈11:25Z; ORCD's 2x H200 slot would only start Oct 23): Arm G here. Waits for GPU 0's reference shard (PID $1), builds the gold
# KL targets on GPU 0 (short views), waits for GPU 1's shard (PID $2), then trains (run_train_real.sh gold) with micro-batch 2; if that dies
# out of memory (RTX 96 GB: 20 x 20 at 1/10 never ran on RTX), retries once with micro-batch 1 (first sync root kept as .mb2).
set -u; Q=/root/rq; cd $Q
while kill -0 $1 2>/dev/null; do sleep 20; done
bash $Q/run_teach_real.sh gold 1
while kill -0 $2 2>/dev/null; do sleep 20; done
MB=2 bash $Q/run_train_real.sh gold > $Q/out/train_gold.log 2>&1
if [ $? != 0 ] && grep -q "out of memory\|OutOfMemoryError" $Q/out/train_gold.log; then
  mv $Q/sync/real_gold $Q/sync/real_gold.mb2; echo "=== $(date -u +%FT%TZ) mb2 OOM -> micro-batch 1"
  MB=1 bash $Q/run_train_real.sh gold > $Q/out/train_gold_mb1.log 2>&1
fi
echo "=== $(date -u +%FT%TZ) armG chain done"
