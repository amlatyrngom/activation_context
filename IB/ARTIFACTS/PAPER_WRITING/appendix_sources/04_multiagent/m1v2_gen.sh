#!/bin/bash
# M1 v2 phase 2 (copy of m1_gen.sh with the v2 paths m1/v2/; coordinator 16:12Z: the H200 pair is held by H8a) on one pc GPU, synthesis fork Oct 16: the 27B (NVFP4) synthesis targets over the M1 triples (tqa_r2_generate.py, as arm B:
# 4 samples per mixed triple, 3 per unanimous; shard SHARD of 2, resumable by triple id). Waits for m1/v2/phase1_done_X (this GPU's phase 1)
# and for TRIPLES.ready (written after the triples file is copied and checked), then for the GPU < 2 GB, and launches in the same test.
# Touches m1/v2/gen_done_X when it ends; RELEASE (if given) is touched however it ends (pc-b: pc/m1_done_b0 hands GPU 0 back).
#   setsid bash m1v2_gen.sh <X> <GPU> <SHARD> <TRIPLES> <OUT> <DEADLINE e.g. 2026-10-17T03:00Z> [RELEASE]
set -uo pipefail
X=$1; G=$2; SHARD=$3; TRIPLES=$4; OUT=$5; END=$6; RELEASE=${7:-}
T=/root/activation_artifacts/tqa; K=$T/kit
stamp() { echo "$(date -u +%FT%TZ) m1v2_gen_$X: $*"; }
trap 'stamp "exit"; [ -n "$RELEASE" ] && touch "$RELEASE" && stamp "touched $RELEASE"' EXIT
trap 'exit 143' TERM INT HUP
. /root/activation_artifacts/pc/kit/node_env.sh
export PYTHONPATH=$PYTHONPATH:$K CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G; }
stamp "waiting for $T/m1/v2/phase1_done_$X and $TRIPLES.ready"
until [ -e $T/m1/v2/phase1_done_$X ] && [ -e $TRIPLES.ready ]; do sleep 30; done
mkdir -p $OUT
until [ "$(used)" -lt 2000 ]; do sleep 30; done
stamp "GPU $G at $(used) MiB; generate shard $SHARD of 2 over $TRIPLES -> $OUT (deadline $END)"
$PY $K/tqa_r2_generate.py --triples $TRIPLES --out $OUT --model unsloth/Qwen3.8-27B-NVFP4 --shard $SHARD --shards 2 \
  --deadline $END --samples-unanimous 3 --chunk 32 > $OUT/gen$SHARD.log 2>&1
stamp "generate ended with $?: $(cat $OUT/gen_$SHARD.jsonl 2>/dev/null | grep -c .) triples"
touch $T/m1/v2/gen_done_$X
