#!/bin/bash
# Compaction stage 1 on one SkyPilot spot node (g7e: 1 x RTX PRO 6000, as the ORCD stage-1 runs), run as a sky job by ec2_launch.sh.
#   cmp_stage1_ec2.sh RUN PACKET            (CMP_* options as cmp_stage1_job.sh, passed in the environment by ec2_launch.sh)
# Inputs from S3 (compaction/stage1/): code/<CMP_BUNDLE>.tar.gz (sha256-checked), runs/R_cmp_lcrec/sync/ckpt/final (the start pair; DONE =
# its meta.json), panel_dev75_tv/, proxy_ref_tv/ (when it exists), packets/<PACKET>/. Reader/encoder weights from the HF hub.
# Spot reclaims: runs/<RUN>/ is restored from S3 before the job (recovery slot with optimizer.pt included), mirrored to S3 every 15 min
# and at the end (S3X_ALLOW_OPTIMIZER=1), so a relaunch on a new node resumes (sec45_train --resume-if-possible).
# The corrected dev-75 references, when this run computes them (CMP_REF missing on S3), go to S3 proxy_ref_tv/.
set -uo pipefail
RUN=${1:?RUN}; PACKET=${2:?PACKET}
S3C=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1
W=/root/cmp1; PY=/opt/activation/.venv/bin/python; BUNDLE=${CMP_BUNDLE:-cmp_csail_code_v1}; J=$W/jobs
export HF_HOME=/root/.cache/huggingface CMP_BUNDLE=$BUNDLE CMP_WORK=$W CMP_PY=$PY
mkdir -p $W/code $W/data/packets $W/init $W/runs $W/logs
say() { echo "$(date -u +%FT%TZ) [ec2 $RUN] $*" | tee -a $W/logs/node.log; }
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)
S3PY=/root/s3venv/bin/python                                   # made by ec2_launch.sh's boot line
X() { $S3PY $J/s3x.py "$@"; }
C=$W/code/$BUNDLE
if [ ! -f $C/MANIFEST.json ]; then
  X get $S3C/code/$BUNDLE.tar.gz $W/code/$BUNDLE.tar.gz > /dev/null && X get $S3C/code/$BUNDLE.tar.gz.sha256 $W/code/$BUNDLE.tar.gz.sha256 > /dev/null
  (cd $W/code && sha256sum -c $BUNDLE.tar.gz.sha256) || { say "bundle sha256 mismatch"; exit 2; }
  mkdir -p $C.tmp && tar -xzf $W/code/$BUNDLE.tar.gz -C $C.tmp && mv -T $C.tmp $C
fi
I=$W/init/R_cmp_lcrec_final
[ -f $I/DONE ] || { X getdir $S3C/runs/R_cmp_lcrec/sync/ckpt/final/ $I/ > /dev/null && grep -q '"update": 241, "planned_updates": 241' $I/meta.json && cp $I/meta.json $I/DONE; } \
  || { say "start pair R_cmp_lcrec_final incomplete"; exit 2; }
[ -f $W/data/panel_dev75_tv/panel_ac.pkl ] || X getdir $S3C/panel_dev75_tv/ $W/data/panel_dev75_tv/ > /dev/null
X getdir $S3C/proxy_ref_tv/ $W/data/ > /dev/null 2>&1 || true          # proxy_ref_tv.json, teacher_dev75_tv.pt/.json when they exist
P=$W/data/packets/$PACKET
[ -f $P/items.pkl ] || X getdir $S3C/packets/$PACKET/ $P/ > /dev/null
[ "$(sha256sum $P/items.pkl | cut -c1-64)" = "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['items_sha256'])" $P/prepared.json)" ] \
  || { say "packet $PACKET: items.pkl sha256 differs from prepared.json"; exit 2; }
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 env -u HF_TOKEN $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done; done
D=$W/runs/$RUN; mkdir -p $D
X getdir $S3C/runs/$RUN/ $D/ > /dev/null 2>&1 || true
say "inputs ready on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)); restored $(find $D -type f | wc -l) run files from S3"
export S3X_ALLOW_OPTIMIZER=1
up() {
  X putdir $D $S3C/runs/$RUN/ --exclude triton_cache > /dev/null 2>&1
  local R=$W/data/${CMP_REF:-proxy_start_pair.json}
  if [ "${CMP_REF:-}" = proxy_ref_tv.json ] && [ -f $R ]; then mkdir -p $W/proxy_ref_tv && cp -u $R $W/data/teacher_dev75_tv.pt $W/data/teacher_dev75_tv.json $W/proxy_ref_tv/ 2>/dev/null
    X putdir $W/proxy_ref_tv $S3C/proxy_ref_tv/ > /dev/null 2>&1; fi
}
( while sleep 900; do up; done ) & LOOP=$!
bash $J/cmp_stage1_job.sh $RUN $PACKET; RC=$?
kill $LOOP 2>/dev/null; up
say "exit $RC; run folder on $S3C/runs/$RUN/"
exit $RC
