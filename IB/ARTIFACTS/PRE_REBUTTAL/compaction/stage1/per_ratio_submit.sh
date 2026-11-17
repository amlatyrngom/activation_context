#!/bin/bash
# Host side (2026-11-17): stage and submit one per-ratio stage-1 run (S1_tv_kl_r5_c162024_s012 / S1_tv_kl_r10_c162024_s012) on CSAIL or ORCD.
#   bash per_ratio_submit.sh CLUSTER RUN PACKET          CLUSTER = csail | csail-gh200 | orcd; PACKET = a local folder name under compaction_stage1/packets/
# Recipe = run M's (cmp_stage1_job.sh: KL to the untuned 9B in the recorded teacher view, reader frozen, encoder 5e-5, warm-up 50, cosine,
# batch 8, one pass, Hidden Residuals) from R_cmp_lcrec_final, code bundle cmp_csail_code_v2 (= main 7488e17; its 128 files equal v1's),
# dev-75 corrected panel (panel_dev75_tv, 150 items, r10 + r5) every 80 updates, recovery slot every 40, final proxy vs proxy_ref_tv.
#   csail: csail/csail_stage.sh put (sha256 of every file checked on arrival) + jobs + sbatch (its pre-job check first):
#          --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=1 --requeue --time=4-00:00:00 (pending until a group GPU frees)
#   csail-gh200: the same on --partition=quanta-gh200 --qos=quanta-main through csail/csail_gh200_job.sh (its first step builds a fresh
#          aarch64 venv from uv.lock in the job, then cmp_stage1_job.sh with that Python)
#   orcd:  cluster_lib.sh (shared connection only; exit 97 if down): the same inputs uploaded and sha256-checked, then
#          --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 (W=/home/ngom/orcd/scratch/activation/cmp1, its rq_v1 venv)
# Appends "RUN CLUSTER JOBID" to compaction_stage1/per_ratio_jobs.txt (csail for both CSAIL copies) for per_ratio_arbiter.sh.
# Prints the job id. The copy that starts first is the run; cancel the other by id (per_ratio_arbiter.sh does it).
set -euo pipefail
CL=${1:?csail|orcd}; RUN=${2:?RUN}; PACKET=${3:?PACKET}
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $HERE/../../../../.. && pwd)
T=$REPO/IB/TMP/PRE_REBUTTAL/compaction_stage1; BUNDLE=cmp_csail_code_v2
[ -f $T/packets/$PACKET/items.pkl ] && [ -f $T/packets/$PACKET/audit_ratio.json ] || { echo "packet $PACKET or its audit_ratio.json missing" >&2; exit 2; }
grep -q '"ok": true' $T/packets/$PACKET/audit_ratio.json || { echo "audit of $PACKET not ok" >&2; exit 2; }
EXPORT="NIL,CMP_INIT=R_cmp_lcrec_final,CMP_BUNDLE=$BUNDLE,CMP_PANEL=panel_dev75_tv,CMP_TEACHER=teacher_dev75_tv.pt,CMP_REF=proxy_ref_tv.json,CMP_REPORT_EVERY=80,CMP_CKPT_EVERY=40"
case $CL in
  csail|csail-gh200)
    S=$HERE/csail/csail_stage.sh; W=/data/scratch-fast/ngom/cmp_csail
    bash $S put $T/packets/$PACKET data/packets/$PACKET
    bash $S jobs
    if [ $CL = csail ]; then ARGS=(--partition=quanta-h200 --cpus-per-task=8 --mem=96G --output=$W/logs/${RUN}_h200_%j.out jobs/cmp_stage1_job.sh)
    else ARGS=(--partition=quanta-gh200 --cpus-per-task=16 --mem=200G --output=$W/logs/${RUN}_gh200_%j.out jobs/csail_gh200_job.sh); fi
    out=$(bash $S sbatch $BUNDLE --export=$EXPORT --account=quanta --qos=quanta-main --gpus=1 --requeue --time=4-00:00:00 --job-name=$RUN "${ARGS[@]}" $RUN $PACKET)
    echo "$out"; id=$(grep -oE "Submitted batch job [0-9]+" <<<"$out" | awk '{print $4}')
    [ -n "$id" ] && echo "$RUN csail $id" >> $T/per_ratio_jobs.txt && echo "job $id ($CL)";;
  orcd)
    source $REPO/IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd; W=/home/ngom/orcd/scratch/activation/cmp1
    put() {   # put LOCAL REMOTE_REL: rsync, then every file's sha256 compared (no find on the login node: sha256sum of the listed names)
      local L=$1 R=$W/$2
      if [ -d $L ]; then remote 60 "mkdir -p $R"; xfer -a $L/ orcd:$R/
        local names; names=$(cd $L && ls -1 | tr '\n' ' ')
        [ "$(cd $L && sha256sum $names)" = "$(remote 1800 "cd $R && sha256sum $names")" ] || { echo "MISMATCH under $R" >&2; exit 1; }
      else remote 60 "mkdir -p $(dirname $R)"; xfer -a $L orcd:$R
        [ "$(sha256sum < $L)" = "$(remote 600 "sha256sum < $R")" ] || { echo "MISMATCH $R" >&2; exit 1; }; fi
      echo "ok $R"; }
    put $T/code/$BUNDLE.tar.gz code/$BUNDLE.tar.gz
    remote 300 "[ -d $W/code/$BUNDLE ] || { mkdir $W/code/$BUNDLE.tmp && tar xzf $W/code/$BUNDLE.tar.gz -C $W/code/$BUNDLE.tmp && mv $W/code/$BUNDLE.tmp $W/code/$BUNDLE; }"
    for x in "proxy_ref_tv/proxy_ref_tv.json data/proxy_ref_tv.json" "proxy_ref_tv/teacher_dev75_tv.pt data/teacher_dev75_tv.pt" \
             "panel_dev75_tv data/panel_dev75_tv" "packets/$PACKET data/packets/$PACKET"; do put $T/${x%% *} ${x#* }; done
    for d in ac reader; do put $T/init/R_cmp_lcrec_final/$d init/R_cmp_lcrec_final/$d; done
    put $T/init/R_cmp_lcrec_final/meta.json init/R_cmp_lcrec_final/meta.json; put $T/init/R_cmp_lcrec_final/DONE init/R_cmp_lcrec_final/DONE
    put $HERE/cmp_stage1_job.sh jobs/cmp_stage1_job.sh
    remote 60 "mkdir -p $W/logs && cd $W && sbatch --export=$EXPORT --partition=mit_preemptable --requeue --gres=gpu:rtx_pro_6000:1 --time=${ORCD_TIME:-2-00:00:00} \
      --cpus-per-task=8 --mem=96G --job-name=$RUN --output=$W/logs/${RUN}_%j.out jobs/cmp_stage1_job.sh $RUN $PACKET" | tee /dev/stderr \
      | grep -oE "Submitted batch job [0-9]+" | awk -v r=$RUN '{print r, "orcd", $4}' >> $T/per_ratio_jobs.txt;;
esac
