#!/bin/bash
# Results + Summary on CSAIL (Oct 18), one GPU, one DeepMath set: summaries (gen_summaries_vllm.py) then the single-call reads over
# summaries + results (run_synth1call_v2.py --conditions synth_summary_result), one tag per read seed. Every step skips what is
# already written, so a preempted and requeued job resumes. Submitted with SET=dmh|dmx (and MODE=smoke for a few tasks) exported.
S=/data/scratch/ngom/activation/paper_traj_qa_v1/sumres_20261018
case $SET in
  dmh) AT3=tqa_dmhard; SEEDS="0 1 2 3 4";;
  dmx) AT3=tqa_dmext_solve_v1; SEEDS="0 1 2";;
  *) echo "SET must be dmh or dmx"; exit 2;;
esac
SUM=tqa_sum_$SET; N=300; PRE=tqa_r2${SET}_sumres_s; LIM=""; SLIM=""
if [ "${MODE:-full}" = smoke ]; then N=2; PRE=tqa_smoke_${SET}_sumres_s; SUM=tqa_smoke_sum_$SET; LIM="--limit 2"; SLIM="--limit 6"; SEEDS=0; fi
DL=$(date -u -d "+9 hours 20 minutes" +%Y-%m-%dT%H:%MZ)
stamp() { echo "$(date -u +%FT%TZ) sumres $SET: $*"; }
done_ids() { python3 -c "import json,sys;print(len({json.loads(l)['id'] for l in open(sys.argv[1]) if l.strip() and json.loads(l).get('finish')!='deadline'}))" "$1" 2>/dev/null || echo 0; }
cd $S/kit
stamp "start (mode ${MODE:-full}, job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} on $(hostname), deadline $DL)"
$PY gen_summaries_vllm.py --prompts $S/prompts/tqa_sum_${SET}_summary_prompts.jsonl --tag $SUM --sync-root $S/SYNC $SLIM || { stamp "summaries failed ($?)"; exit 1; }
W=$(grep -c . $S/SYNC/ROLLOUTS/${SUM}_summaries/rollouts.jsonl); stamp "summaries rows $W"
for s in $SEEDS; do
  O=$S/SYNC/PAPER_TRAJ_QA/$PRE$s
  [ "$(done_ids $O/gen_0.jsonl)" -ge $N ] && { stamp "$PRE$s complete, skipped"; continue; }
  stamp "$PRE$s start"
  $PY run_synth1call_v2.py --sync-root $S/SYNC --at3-id $AT3 --summaries-tag $SUM --out $O --conditions synth_summary_result --seed $s --deadline $DL $LIM
  stamp "$PRE$s ended ($?): $(done_ids $O/gen_0.jsonl) done"
done
stamp "chain done"
