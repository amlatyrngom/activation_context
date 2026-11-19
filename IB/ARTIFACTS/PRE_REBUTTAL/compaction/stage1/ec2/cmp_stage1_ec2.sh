#!/bin/bash
# Compaction stage 1 on one SkyPilot spot node (g7e: 1 x RTX PRO 6000, as the ORCD stage-1 runs), run as a sky job by ec2_launch.sh.
#   cmp_stage1_ec2.sh RUN PACKET            (CMP_* options as cmp_stage1_job.sh, passed in the environment by ec2_launch.sh)
# Inputs from S3 (compaction/stage1/): code/<CMP_BUNDLE>.tar.gz (sha256-checked), runs/R_cmp_lcrec/sync/ckpt/final (the start pair; DONE =
# its meta.json), panel_dev75_tv/, proxy_ref_tv/ (when it exists), packets/<PACKET>/. Reader/encoder weights from the HF hub.
# Spot reclaims: runs/<RUN>/ is restored from S3 before the job (recovery slot with optimizer.pt included), mirrored to S3 every 15 min
# and at the end (S3X_ALLOW_OPTIMIZER=1), so a relaunch on a new node resumes (sec45_train --resume-if-possible).
# CMP_TVCOMPARE=N (2026-11-17): before training, agent_eval/teacher_view_compare.py on the packet's first N runs (old escaped vs recorded
# teacher view, read by the frozen reader: first-turn "[/" mass), -> runs/<RUN>/proxy/tvcompare.json.
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
if [ "${CMP_INIT:-R_cmp_lcrec_final}" != R_cmp_lcrec_final ]; then   # 2026-11-18 (stage 2): CMP_INIT=<RUN>_final = runs/<RUN>/sync/ckpt/final, complete
  I2=$W/init/$CMP_INIT; R0=${CMP_INIT%_final}
  [ -f $I2/DONE ] || { [ "$R0" != "$CMP_INIT" ] && X getdir $S3C/runs/$R0/sync/ckpt/final/ $I2/ > /dev/null && \
    $PY -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m['update'] == m['planned_updates'] and m['name'] == sys.argv[2] else 1)" $I2/meta.json $R0 && cp $I2/meta.json $I2/DONE; } \
    || { say "start pair $CMP_INIT incomplete (S3 runs/$R0/sync/ckpt/final)"; exit 2; }
fi
[ -f $W/data/panel_dev75_tv/panel_ac.pkl ] || X getdir $S3C/panel_dev75_tv/ $W/data/panel_dev75_tv/ > /dev/null
X getdir $S3C/proxy_ref_tv/ $W/data/ > /dev/null 2>&1 || true          # proxy_ref_tv.json, teacher_dev75_tv.pt/.json when they exist
P=$W/data/packets/$PACKET
[ -f $P/items.pkl ] || X getdir $S3C/packets/$PACKET/ $P/ > /dev/null
[ "$(sha256sum $P/items.pkl | cut -c1-64)" = "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['items_sha256'])" $P/prepared.json)" ] \
  || { say "packet $PACKET: items.pkl sha256 differs from prepared.json"; exit 2; }
for m in Qwen/Qwen3.5-9B Qwen/Qwen3.5-2B; do
  for t in 1 2 3; do timeout 1800 env -u HF_TOKEN $PY -c "from huggingface_hub import snapshot_download as s; s('$m', token=False)" > /dev/null && break; done; done
D=$W/runs/$RUN; mkdir -p $D
export S3X_ALLOW_OPTIMIZER=1                                   # 2026-11-17: before the restore too (s3x skips optimizer.pt otherwise: resume refused)
restore() {   # 2026-11-19: the run folder without the older recovery slots: only the slot partial_epoch.json points to (pointers are
  # uploaded after their slot: complete) + everything outside the slot folders (ckpt/final, logs, panels); older slots stay on S3.
  # Any doubt (no boto3, a state.json pointer, an error) -> the full getdir as before. Fetched files get the S3 object's mtime.
  $S3PY - $J $S3C/runs/$RUN/ $D <<'PYEOF' || X getdir $S3C/runs/$RUN/ $D/
import json, os, re, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import s3x
s3, config = s3x.client(); bucket, prefix = s3x.split(sys.argv[2].rstrip("/") + "/"); local = Path(sys.argv[3])
when = {it["Key"][len(prefix):]: it["LastModified"].timestamp() for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix) for it in page.get("Contents", [])}
rel = list(when)
if any(Path(r).name == "state.json" for r in rel): sys.exit(1)
keep = set()
for r in rel:
    if r.startswith("sync/AC_MODELS/") and Path(r).name == "partial_epoch.json":
        p = json.loads(s3.get_object(Bucket=bucket, Key=prefix + r)["Body"].read())
        keep |= {"sync/" + p[k].strip("/") + "/" for k in ("ac_checkpoint", "target_lora_checkpoint") if p.get(k)}
slot = re.compile(r"^sync/(AC_MODELS|LORAS)/[^/]+/epoch_\d+_step_\d+/")
fetched = left = 0
for r in rel:
    m = slot.match(r)
    if m and m.group(0) not in keep: left += 1; continue
    if Path(r).name in s3x.NEVER: continue
    if s3x.get(s3, config, f"s3://{bucket}/{prefix}{r}", local / r):
        fetched += 1; os.utime(local / r, (when[r], when[r]))   # mtime = the S3 copy's: the mirror does not send it back (M: ~200 GB re-sent)
print(f"s3x restore: {fetched} files fetched; latest slot {sorted(keep)}; {left} older-slot files left on S3")
PYEOF
}
say "$(restore 2>&1 | tail -n 1)"
say "inputs ready on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)); restored $(find $D -type f | wc -l) run files ($(find $D -name optimizer.pt | wc -l) optimizer.pt) from S3"
export S3X_ALLOW_OPTIMIZER=1
up() {
  X putdir $D $S3C/runs/$RUN/ --exclude triton_cache final.json > /dev/null 2>&1   # proxy/final.json (= done) only at the end, after ckpt/final
  local R=$W/data/${CMP_REF:-proxy_start_pair.json}
  if [ "${CMP_REF:-}" = proxy_ref_tv.json ] && [ -f $R ]; then mkdir -p $W/proxy_ref_tv && cp -u $R $W/data/teacher_dev75_tv.pt $W/data/teacher_dev75_tv.json $W/proxy_ref_tv/ 2>/dev/null
    X putdir $W/proxy_ref_tv $S3C/proxy_ref_tv/ > /dev/null 2>&1; fi
}
( while sleep 900; do up; done ) & LOOP=$!
if [ -n "${CMP_TVCOMPARE:-}" ] && [ ! -f $D/proxy/tvcompare.json ]; then   # CMP_TVCOMPARE=N: old vs recorded teacher view on the packet's first N runs
  mkdir -p $D/proxy $D/logs; say "teacher_view_compare on the first $CMP_TVCOMPARE runs of $PACKET"
  (export PYTHONPATH=$C:$C/kit:$C/rq_kit:$C/bench CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit SEC45_KIT=$C/kit RQ_THINKING=0 \
     QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128 SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_LINEAR_SKIP=1 QA4B_KV=0 \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
   cd $C && $PY $J/teacher_view_compare.py --pair $I/ac $I/reader --packet $PACKET=$P/items.pkl --out $D/proxy/tvcompare.json --limit $CMP_TVCOMPARE) \
    > $D/logs/tvcompare.log 2>&1; say "teacher_view_compare exit $?"; up
fi
bash $J/cmp_stage1_job.sh $RUN $PACKET; RC=$?
kill $LOOP 2>/dev/null; up
# 2026-11-19: proxy/final.json marks the run done (ec2_keep, stage2_after_stage1, the gates): upload it last, after a second pass
# confirms everything else (ckpt/final) is on S3 (M's final.json landed 30 s before its ckpt/final weights).
[ -f $D/proxy/final.json ] && X putdir $D $S3C/runs/$RUN/ --exclude triton_cache final.json > /dev/null 2>&1 && \
  X put $D/proxy/final.json $S3C/runs/$RUN/proxy/final.json > /dev/null 2>&1
say "exit $RC; run folder on $S3C/runs/$RUN/"
exit $RC
