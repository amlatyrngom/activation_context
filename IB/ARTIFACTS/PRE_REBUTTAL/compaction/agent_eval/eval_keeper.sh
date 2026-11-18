#!/bin/bash
# Spot-safe queue of agent-eval / generation jobs on one node: (re)creates the node when it is gone and reruns each job until every
# variant cache of it has its DONE marker on S3 (dev75_node.sh resumes a cache from its S3 mirror, so a rerun continues).
#   eval_keeper.sh NODE "DEVSET TAG VARIANTS AGENTS PAIR SEED" ...       (TAG = full cache tag, e.g. s2g_s0_r_s22)
cd /home/ubuntu/activation_context
NODE=$1; shift
S3=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/agent_eval_dev75/ROLLOUTS
L=IB/TMP/PRE_REBUTTAL/cmp_agent_eval; mkdir -p $L; LOG=$L/eval_keeper_$NODE.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
done_all() { local tag=$1 v; for v in ${2//,/ }; do aws s3 ls $S3/${tag}_$v/DONE > /dev/null 2>&1 || return 1; done; }
ensure_node() {
  .venv/bin/sky ls 2>/dev/null | grep -q "^$NODE .* UP " && return 0
  for t in 1 2 3 4 5 6; do
    say "creating $NODE (attempt $t)"
    .venv/bin/python -m activation.cloud.sky setup --name $NODE --gpu rtx6000pro --spot --instance-type g7e.4xlarge >> $L/setup_$NODE.log 2>&1 && return 0
    sleep 300
  done; return 1
}
for job in "$@"; do
  read -r DEVSET TAG VARIANTS AGENTS PAIR SEED <<< "$job"
  case $DEVSET in prefixes_dev9_75) N=75;; prefixes_dev9_206) N=206;; prefixes_s2gen_c20_s0) N=1261;; prefixes_s2gen_c20_s1) N=1360;; prefixes_s2gen_c20_s2) N=1364;; *) say "bad set $DEVSET"; continue;; esac
  for a in 1 2 3 4 5 6 7 8; do
    done_all $TAG $VARIANTS && { say "$TAG DONE"; break; }
    ensure_node || { say "cannot create $NODE; giving up"; exit 1; }
    say "$TAG attempt $a on $NODE ($VARIANTS, $DEVSET)"
    DEVSET=$DEVSET DEVN=$N bash IB/ARTIFACTS/PRE_REBUTTAL/compaction/agent_eval/launch.sh $NODE run $TAG $SEED $VARIANTS $AGENTS $PAIR >> $L/exec_$TAG.log 2>&1
    say "$TAG attempt $a exit $?"
  done
done
say "queue finished"
