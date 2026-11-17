#!/bin/bash
# dev-75 AC arms of S1_lcrec100_kl_both_c20 final on a node, seeds 2-4, 32 agents/GPU (one sky job per seed)
cd /home/ubuntu/activation_context
NODE=${1:-cdv1}; RUN=${2:-S1_lcrec100_kl_both_c20}; SHORT=${3:-lcrec100}
for s in 2 3 4; do
  bash IB/ARTIFACTS/PRE_REBUTTAL/compaction/agent_eval/launch.sh $NODE run cdv1${SHORT}_s$s $s ac_4000,ac_2000 32 $RUN > IB/TMP/PRE_REBUTTAL/cmp_agent_eval/exec_${SHORT}_s$s.log 2>&1
  echo "$(date -u +%FT%TZ) $SHORT seed $s exit $?" >> IB/TMP/PRE_REBUTTAL/cmp_agent_eval/ac_chain.log
done
