#!/bin/bash
# Run a list of seeds of one arm set on one node, sequentially (one sky job per seed), on a chosen dev set.
#   dev_chain.sh NODE DEVSET TAGPREFIX VARIANTS AGENTS PAIR_RUN|- SEED...
#   e.g. dev_chain.sh dvA prefixes_dev9_206 d206base full,none,summary_write,summary 40 - 2 4
#        dev_chain.sh s1sm1 prefixes_dev9_75 cdv1tvF ac_4000,ac_2000 32 S1_tv_kl_both_c20_s012 2 3 4
# Tags: <TAGPREFIX>_s<seed>; dev-75 tags keep the cdv1* prefix of the existing baselines (cdv1base_s*), dev-206 tags start d206.
cd /home/ubuntu/activation_context
NODE=$1; DEVSET=$2; PFX=$3; VARIANTS=$4; AGENTS=$5; RUN=$6; shift 6
case $DEVSET in prefixes_dev9_75) N=75;; prefixes_dev9_206) N=206;; *) echo "not a dev set" >&2; exit 2;; esac
L=IB/TMP/PRE_REBUTTAL/cmp_agent_eval; mkdir -p $L
for s in "$@"; do
  echo "$(date -u +%FT%TZ) $NODE $PFX seed $s start ($VARIANTS, $DEVSET)" >> $L/dev_chain.log
  DEVSET=$DEVSET DEVN=$N bash IB/ARTIFACTS/PRE_REBUTTAL/compaction/agent_eval/launch.sh $NODE run ${PFX}_s$s $s $VARIANTS $AGENTS $RUN > $L/exec_${PFX}_s$s.log 2>&1
  echo "$(date -u +%FT%TZ) $NODE $PFX seed $s exit $?" >> $L/dev_chain.log
done
