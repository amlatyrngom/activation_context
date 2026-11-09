#!/bin/bash
# rq-fresh3: KL targets for one Track A real-format arm. Usage: bash run_teach_real.sh VIEW (same | gold) [SHARDS (default 2, one per GPU)]
# merges refs/real_VIEW*.jsonl (shards) into refs/real_VIEW.all.jsonl if needed, then rq_teacher --refs ... --view VIEW over both GPUs,
# then merges the shard targets into teacher/real_VIEW.pt. Base 9B = the tA_s2 pair's target (no adapter).
set -u; Q=/root/rq; V=$1; N=${2:-2}; cd ~/sky_workdir
export RQ_THINKING=0 PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$Q/kit_c2:$Q/bench:$Q CQA_BENCH=$Q/bench
R=$Q/refs/real_$V.all.jsonl; mkdir -p $Q/teacher
if [ ! -e $R ]; then
  if [ $V = gold ]; then cp -n $Q/refs/real_gold.jsonl $R; else uv run --frozen python $Q/kit_c2/rq_refs.py --merge $R $Q/refs/real_same.sh?.jsonl; fi
fi
echo "=== $(date -u +%FT%TZ) teacher targets view $V from $R over $N GPUs"
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i uv run --frozen python $Q/kit_c2/rq_teacher.py --refs $R --questions $Q/data/hotpot_train_native.questions.jsonl --stages held real \
     --view $V --data $Q/data --pair $Q/tA_s2/AC_MODELS/epoch_004 $Q/tA_s2/LORAS/epoch_004 --out $Q/teacher/real_$V.$i.pt --shard $i/$N > $Q/out/teach_${V}_$i.log 2>&1 &
done
wait
uv run --frozen python $Q/kit_c2/rq_teacher.py --merge $Q/teacher/real_$V.pt $(for i in $(seq 0 $((N-1))); do echo $Q/teacher/real_$V.$i.pt; done)
echo "=== $(date -u +%FT%TZ) teacher targets view $V exit $?"
