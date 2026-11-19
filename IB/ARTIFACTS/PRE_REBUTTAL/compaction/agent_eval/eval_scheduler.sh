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
# runs and points come from eval_runs.txt (re-read before every job): "LABEL RUN ARMS POINT..." per line, "#" comments
RUNS_FILE=$AE/eval_runs.txt
done_all() { local tag=$1 v; for v in ${2//,/ }; do aws s3 ls $S3E/${tag}_$v/DONE > /dev/null 2>&1 || return 1; done; }
slot_ok() { aws s3 ls $S3R/$1/sync/AC_MODELS/rag_poc/epoch_001_step_$(printf %05d $2)/ac_modules.pt > /dev/null 2>&1 && \
            aws s3 ls $S3R/$1/sync/LORAS/rag_reader/epoch_001_step_$(printf %05d $2)/adapter_model.safetensors > /dev/null 2>&1; }
final_ok() { aws s3 ls $S3R/$1/sync/ckpt/final/meta.json > /dev/null 2>&1 && aws s3 ls $S3R/$1/sync/ckpt/final/ac/ac_modules.pt > /dev/null 2>&1; }
node_idle() { local q; q=$(timeout 300 $SKY queue $NODE 2>/dev/null) || return 0   # node gone: the keeper recreates it
  ! echo "$q" | grep -qE " (RUNNING|PENDING|SETTING_UP|STARTING) "; }
next_job() {   # prints "DEVSET TAG VARIANTS AGENTS PAIR SEED" or nothing
  local cand=() r label run arms u name tag s
  while read -r label run arms pts; do
    [ -z "$label" ] && continue; [ "${label:0:1}" = "#" ] && continue
    for u in $pts; do cand+=("$u $label $run $arms u$u"); done
    final_ok $run && cand+=("0 $label $run $arms fin")   # finals first
  done < $RUNS_FILE
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
  [ "$1" = p1 ] && return   # p1: priority-1 (training-run evals) only
  done_all s2g_s1_r_s22 full || { echo "prefixes_s2gen_c20_s1 s2g_s1_r_s22 full 40 - 22"; return; }
  for s in 2 3 4; do done_all d206base_s$s full,none,summary_write,summary || { echo "prefixes_dev9_206 d206base_s$s full,none,summary_write,summary 40 - $s"; return; }; done
}
# 2026-11-19: each finished eval leaves 11-21 GB of saved_tensors on the node (S3 keeps only rollouts.jsonl + DONE); dvA's 485 GB disk
# filled after 26 evals and evals failed (exit 100). After every job, drop saved_tensors of the evals whose DONE + rollouts.jsonl are on S3.
clean_node() { local d n=0
  for d in $(timeout 120 ssh -o ConnectTimeout=20 $NODE 'ls /root/activation_artifacts/SYNC/ROLLOUTS' 2>/dev/null); do
    aws s3 ls $S3E/$d/DONE > /dev/null 2>&1 && aws s3 ls $S3E/$d/rollouts.jsonl > /dev/null 2>&1 || continue
    timeout 300 ssh -o ConnectTimeout=20 $NODE "rm -rf /root/activation_artifacts/SYNC/ROLLOUTS/$d/saved_tensors" 2>/dev/null && n=$((n+1))
  done
  say "clean_node: checked $n S3-done rollout dirs; $(timeout 60 ssh -o ConnectTimeout=20 $NODE 'df -h / | tail -1' 2>/dev/null)"; }
say "scheduler start on $NODE"
while true; do
  until node_idle; do sleep 120; done
  job=$(next_job)
  if [ -z "$job" ]; then sleep 900; continue; fi
  say "next: $job"
  case "$job" in
    *" s2g_"*|*" d206base_"*)   # low priority: runs preemptibly; every 10 min, if a training-run eval is ready, cancel it (it resumes later from S3)
      setsid bash $AE/eval_keeper.sh $NODE "$job" & kp=$!
      while kill -0 $kp 2>/dev/null; do
        sleep 600; kill -0 $kp 2>/dev/null || break
        p=$(next_job p1)
        if [ -n "$p" ]; then say "preempting low-priority job for: $p"; kill -- -$kp 2>/dev/null
          timeout 300 $SKY cancel -y -a $NODE >> $LOG 2>&1; say "cancel returned $?"; sleep 60; break; fi
      done; wait $kp 2>/dev/null;;
    *) bash $AE/eval_keeper.sh $NODE "$job";;
  esac
  clean_node
done
