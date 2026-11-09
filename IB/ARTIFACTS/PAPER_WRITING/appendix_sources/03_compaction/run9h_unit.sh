#!/bin/bash
# A hill-climbed pair on the 200 evaluation prefixes at the 50k limit (Oct 15 ≈03:10Z: H5 = H0 + 155 9B-written pool runs passed the
# dev promotion rule: dev s0, 1/10, 57 solved vs H0 50, tokens −3,359). Same settings as run9d_unit.sh's AC step (50k, H0's encoder
# batch defaults, rows only), 32 agents (40+ thrashed with the encoder resident).
#   run9h_unit.sh <seed> <tag> <variants> <pair dir>      e.g. run9h_unit.sh 0 pc_r9h5_s0 ac_2000 /root/activation_artifacts/pc/pairs/h5_final
set -uo pipefail
A=/root/activation_artifacts/pc; SEED=$1; TAG=$2; VARS=$3; PAIR=$4
echo "$(date -u +%FT%TZ) run9h: seed $SEED tag $TAG variants $VARS pair $PAIR on GPU $CUDA_VISIBLE_DEVICES"
bash $A/kit/resume_node.sh prefixes_eval9c_200 --variants $VARS --budget 50000 --trajectory-cap 90000 --max-model-len 100000 --max-duration 6000 \
  --seed $SEED --tag $TAG --agents-per-gpu 32 --ac-checkpoint $PAIR/ac --reader-checkpoint $PAIR/reader --rows-only
echo "$(date -u +%FT%TZ) run9h: $TAG $VARS ended ($?)"
