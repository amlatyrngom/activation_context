#!/bin/bash
# Stage the section 4.5 inputs on a cluster (run in THIS container):  bash jobs/stage.sh csail|orcd [--jobs-only]
#   code   $S3/code/$SEC45_BUNDLE (default sec45_code_v3.tar.gz, + .sha256), fetched here with this container's credentials,
#          verified, rsynced to W/code/ and verified there
#   jobs   every jobs/generated*/ folder -> W/jobs/ (--jobs-only: this step alone, e.g. after make_jobs.py --probes)
#   data   data/fineweb_recon_v1/ and data/hotpot_tasks_v1/ (this folder) -> W/data/
#   model  Qwen/Qwen3-4B-Instruct-2507 in the cluster's HF cache: checked; on CSAIL downloaded on the login node if missing
#          (the cluster venv's huggingface_hub, token=False, no HF token in the environment); ORCD already has it (checked only)
# W = /data/scratch-fast/ngom/sec45 (CSAIL; /data/scratch/ngom is over quota) or /home/ngom/orcd/scratch/activation/sec45 (ORCD). Exit 97: the shared ssh connection is down.
# SEC45_DRY_RUN=1 prints the remote commands without any ssh.
set -euo pipefail
source "$(dirname "$0")/cluster_lib.sh"
[ $# -ge 1 ] || { sed -n 2,11p "$0"; exit 2; }
BUNDLE=${SEC45_BUNDLE:-sec35_code_v2.tar.gz}
cluster $1
stage_jobs() {
  echo "== $H: job scripts"
  remote 60 "mkdir -p $W/jobs"
  xfer -a --no-owner --no-group $KIT/jobs/text_lora_job.sh $KIT/jobs/rescore_job.sh $H:$W/jobs/
  for g in $KIT/jobs/generated*/; do [ -d "$g" ] || continue; xfer -a --no-owner --no-group --exclude '*.tmp' $g $H:$W/jobs/$(basename $g)/; done
  remote 60 "ls $W/jobs/*"
}
if [ "${2:-}" = --jobs-only ]; then stage_jobs; exit 0; fi
CACHE=$REPO/IB/TMP/PRE_REBUTTAL/stage_cache; mkdir -p $CACHE
for f in $BUNDLE $BUNDLE.sha256; do [ -f $CACHE/$f ] || aws s3 cp --quiet $S3/code/$f $CACHE/$f; done
(cd $CACHE && sha256sum -c $BUNDLE.sha256)
for d in fineweb_recon_v1/material.json.gz hotpot_tasks_v2/train.jsonl hotpot_tasks_v2/dev200.jsonl hotpot_tasks_v2/dev1000.jsonl hotpot_tasks_v2/test1000.jsonl hotpot_tasks_v2/hotpotqa.corpus.jsonl.gz rq_tasks_v1/train.jsonl rq_tasks_v1/dev200.jsonl rq_tasks_v1/eval500.jsonl rq_tasks_v1/hotpotqa.corpus.jsonl.gz; do
  [ -f $DATA_ROOT/$d ] || { echo "local data $DATA_ROOT/$d is missing (s3x getdir $S3/data/... first)" >&2; exit 2; }
done
echo "== $H: folders"
remote 60 "mkdir -p $W/code $W/data $W/init $W/runs $W/home $W/triton_cache && df -h $W | tail -1"
echo "== $H: code bundle"
xfer -a --no-owner --no-group $CACHE/$BUNDLE $CACHE/$BUNDLE.sha256 $H:$W/code/
remote 60 "cd $W/code && sha256sum -c $BUNDLE.sha256"
echo "== $H: data"
for d in fineweb_recon_v1 hotpot_tasks_v2 rq_tasks_v1; do
  xfer -aL --no-owner --no-group --exclude '*.tmp' --exclude '*.part' $DATA_ROOT/$d/ $H:$W/data/$d/
done
remote 60 "ls -l $W/data/fineweb_recon_v1 $W/data/hotpot_tasks_v2"
echo "== $H: $MODEL in $HF"
CHECK="env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HOME=$HF HF_HUB_OFFLINE=1 $PY -c \"from huggingface_hub import snapshot_download as s; print(s('$MODEL', local_files_only=True))\""
if remote 120 "$CHECK"; then echo "model present"
elif [ $H = csail ]; then
  echo "model missing on CSAIL: downloading on the login node (public weights, token=False)"
  remote 3600 "env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HOME=$HF HF_HUB_DISABLE_TELEMETRY=1 $PY -c \"from huggingface_hub import snapshot_download as s; print(s('$MODEL', token=False))\""
  remote 120 "$CHECK"
else
  echo "model missing on ORCD in $HF: not downloaded by this script (ask the coordinator)" >&2; exit 3
fi
stage_jobs
echo "staged on $H under $W"
