#!/bin/bash
# Priority scheduler for one agent-eval node (user, 2026-11-18: end-to-end dev evals at regular points of training, not only finals).
#   eval_scheduler.sh NODE
# Loop: pick the highest-priority unfinished job, wait until the node's sky queue is empty, run it through eval_keeper.sh (one job), repeat.
# Priority 1: intermediate / final dev-75 evals of the training runs, oldest point first (slot -> EVALSNAP pair via snap_eval_pair.sh).
# Priority 2: the stage-2 generation set.  Priority 3: the dev-206 baselines.  Nothing left: sleep and look again.
cd /home/ubuntu/activation_context
NODE=$1
AE=IB/ARTIFACTS/PRE_REBUTTAL/compaction/agent_eval
S3R=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1/runs
S3E=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/agent_eval_dev75/ROLLOUTS
L=IB/TMP/PRE_REBUTTAL/cmp_agent_eval; LOG=$L/eval_scheduler_$NODE.log
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
SKY="uv tool run --no-config --python 3.13 --with pip --from skypilot[aws]==0.13.0 sky"
# training runs: LABEL RUN ARMS ; points every 200 updates + final
RUNS=("M S1_tv_kl_both_c162024_s012 ac_4000,ac_2000" "R10 S1_tv_kl_r10_c162024_s012 ac_2000" "R5 S1_tv_kl_r5_c162024_s012 ac_4000")
POINTS_M="1120"; POINTS_PR="200 400 600 800 1000 1200"
done_all() { local tag=$1 v; for v in ${2//,/ }; do aws s3 ls $S3E/${tag}_$v/DONE > /dev/null 2>&1 || return 1; done; }
slot_ok() { aws s3 ls $S3R/$1/sync/AC_MODELS/rag_poc/epoch_001_step_$(printf %05d $2)/ac_modules.pt > /dev/null 2>&1 && \
            aws s3 ls $S3R/$1/sync/LORAS/rag_reader/epoch_001_step_$(printf %05d $2)/adapter_model.safetensors > /dev/null 2>&1; }
final_ok() { aws s3 ls $S3R/$1/sync/ckpt/final/meta.json > /dev/null 2>&1 && aws s3 ls $S3R/$1/sync/ckpt/final/ac/ac_modules.pt > /dev/null 2>&1; }
node_idle() { local q; q=$(timeout 300 $SKY queue $NODE 2>/dev/null) || return 0   # node gone: the keeper recreates it
  ! echo "$q" | grep -qE " (RUNNING|PENDING|SETTING_UP|STARTING) "; }
next_job() {   # prints "DEVSET TAG VARIANTS AGENTS PAIR SEED" or nothing
  local cand=() r label run arms u name tag s
  for r in "${RUNS[@]}"; do read -r label run arms <<< "$r"
    local pts=$POINTS_PR; [ $label = M ] && pts=$POINTS_M
    for u in $pts; do cand+=("$u $label $run $arms u$u"); done
    final_ok $run && cand+=("99999 $label $run $arms fin")
  done
  while read -r u label run arms pt; do
    [ -z "$u" ] && continue
    if [ $pt = fin ]; then name=$run; else name=EVALSNAP_${label,,}_$pt; [ $label = M ] && name=EVALSNAP_M_$pt; fi
    for s in 2 3 4; do
      tag=cdv1ev${label}${pt}_s$s; [ $label = M ] && [ $pt = u1120 ] && tag=cdv1evM1120_s$s
      [ $label = R10 ] && [ $pt = u400 ] && tag=cdv1evR10u400_s$s; [ $label = R5 ] && [ $pt = u400 ] && tag=cdv1evR5u400_s$s
      done_all $tag $arms && continue
      if [ $pt != fin ]; then
        aws s3 ls $S3R/$name/sync/ckpt/final/meta.json > /dev/null 2>&1 || { slot_ok $run $u || continue 2
          bash $AE/snap_eval_pair.sh $run $u $name >> $LOG 2>&1 && say "snapshot $name from $run u$u"; }
      fi
      echo "prefixes_dev9_75 $tag $arms 40 $name $s"; return
    done
  done < <(printf '%s\n' "${cand[@]}" | sort -n)
  done_all s2g_s1_r_s22 full || { echo "prefixes_s2gen_c20_s1 s2g_s1_r_s22 full 40 - 22"; return; }
  for s in 2 3 4; do done_all d206base_s$s full,none,summary_write,summary || { echo "prefixes_dev9_206 d206base_s$s full,none,summary_write,summary 40 - $s"; return; }; done
}
say "scheduler start on $NODE"
while true; do
  until node_idle; do sleep 120; done
  job=$(next_job)
  if [ -z "$job" ]; then sleep 900; continue; fi
  say "next: $job"
  bash $AE/eval_keeper.sh $NODE "$job"
done
