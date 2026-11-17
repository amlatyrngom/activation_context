#!/bin/bash
# The "[/" vs "</think>" probe (compaction/agent_eval/probe_think.py) of a stage-1 run's final pair on the corrected dev-75 panel
# (panel_dev75_tv), on a SkyPilot node, in the code bundle's environment. Run by ec2_launch.sh with EC2_SCRIPT=probe_ec2.sh:
#   EC2_SCRIPT=probe_ec2.sh ec2_launch.sh NODE RUN - [PROBE_REFERENCES=1]
# PROBE_REFERENCES=1 also scores the pair-independent views of the frozen reader (dev75/none and the corrected teacher view "full").
# Output: S3 compaction/stage1/runs/RUN/probe/probe_final.json (+ .positions.jsonl), log next to it.
set -uo pipefail
RUN=${1:?RUN}
S3C=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1
W=/root/cmp1; PY=/opt/activation/.venv/bin/python; BUNDLE=${CMP_BUNDLE:-cmp_csail_code_v1}; J=$W/jobs
export HF_HOME=/root/.cache/huggingface
say() { echo "$(date -u +%FT%TZ) [probe $RUN] $*"; }
(cd /root/sky_workdir && UV_PROJECT_ENVIRONMENT=/opt/activation/.venv /usr/local/bin/uv sync --frozen -q)
X() { /root/s3venv/bin/python $J/s3x.py "$@"; }
C=$W/code/$BUNDLE
if [ ! -f $C/MANIFEST.json ]; then
  mkdir -p $W/code && X get $S3C/code/$BUNDLE.tar.gz $W/code/$BUNDLE.tar.gz > /dev/null && X get $S3C/code/$BUNDLE.tar.gz.sha256 $W/code/$BUNDLE.tar.gz.sha256 > /dev/null
  (cd $W/code && sha256sum -c $BUNDLE.tar.gz.sha256) || { say "bundle sha256 mismatch"; exit 2; }
  mkdir -p $C.tmp && tar -xzf $W/code/$BUNDLE.tar.gz -C $C.tmp && mv -T $C.tmp $C
fi
[ -f $W/data/panel_dev75_tv/panel_ac.pkl ] || X getdir $S3C/panel_dev75_tv/ $W/data/panel_dev75_tv/ > /dev/null
Q=$W/probe_pairs/$RUN; O=$W/probe_out/$RUN; mkdir -p $O
[ -f $Q/ac/ac_modules.pt ] || { X getdir $S3C/runs/$RUN/sync/ckpt/final/ac/ $Q/ac/ > /dev/null; X getdir $S3C/runs/$RUN/sync/ckpt/final/reader/ $Q/reader/ > /dev/null; }
[ -f $Q/ac/ac_modules.pt ] && [ -f $Q/reader/adapter_model.safetensors ] || { say "final pair of $RUN not on S3"; exit 2; }
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 env -u HF_TOKEN $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done; done
REF=; [ "${PROBE_REFERENCES:-0}" = 1 ] && REF=--references
say "probe of $S3C/runs/$RUN/sync/ckpt/final $REF"
(export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
   QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
 cd $C && $PY $J/probe_think.py --pair $Q/ac $Q/reader --label ${RUN}_final --panel $W/data/panel_dev75_tv --out $O/probe_final.json $REF) > $O/probe_final.log 2>&1
RC=$?; say "probe exit $RC"
X putdir $O $S3C/runs/$RUN/probe/ > /dev/null; say "uploaded to $S3C/runs/$RUN/probe/"
exit $RC
