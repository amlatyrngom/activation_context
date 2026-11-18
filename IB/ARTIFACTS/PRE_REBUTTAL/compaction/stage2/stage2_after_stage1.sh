#!/bin/bash
# Host side (start with nohup ... &): when a per-ratio stage-1 run has finished on its EC2 node, probe its final pair there, then start
# that ratio's stage-2 run on the same node under ec2_keep.sh (spot reclaims: resume from the recovery slot, every 25 updates).
#   stage2_after_stage1.sh L NODE PACKET          L = r5 | r10;  e.g.  stage2_after_stage1.sh r10 prr10 s2ce_v1_r10
#   S1RUN=S1_tv_kl_both_c162024_s012 S2RUN=S2_ce_Mfin_r10_c20_s012 stage2_after_stage1.sh r10 cmpM s2ce_v1_r10   (stage 2 from M's final)
#     stage 1:  S1_tv_kl_<L>_c162024_s012      (done = S3 runs/<it>/proxy/final.json, the job's last step)
#     stage 2:  S2_ce_<L>_c20_s012 on PACKET (compaction/stage1/packets/ on S3, audit_stage2.json ok), start pair = stage 1's ckpt/final
#               (CMP_INIT=<S1 RUN>_final: cmp_stage1_ec2.sh fetches and checks it), loss sft, dev-75 corrected panel every 80 updates,
#               recovery slot every 25, bundle v2, the stage-1 settings otherwise (encoder 5e-5, reader 0, warm-up 50, cosine, batch 8, one pass).
# The probe (probe_ec2.sh, PROBE_REFERENCES=1) runs first on the same node; its failure is logged, not fatal (stage 2 starts anyway).
set -uo pipefail
L=${1:?r5|r10}; NODE=${2:?NODE}; PACKET=${3:?PACKET}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd $HERE/../../../../.. && pwd); E=$HERE/../stage1/ec2
S1=${S1RUN:-S1_tv_kl_${L}_c162024_s012}; S2=${S2RUN:-S2_ce_${L}_c20_s012}   # S1RUN / S2RUN: another start run / name (2026-11-18: S2_ce_Mfin_r10_c20_s012 from M)
S3C=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1
LOGS=$ROOT/IB/TMP/PRE_REBUTTAL/compaction_stage1/logs/ec2; mkdir -p $LOGS
say() { echo "$(date -u +%FT%TZ) [stage2 $L] $*"; }
grep -q '"ok": true' $ROOT/IB/TMP/PRE_REBUTTAL/compaction_stage2/packets/$PACKET/audit_stage2.json || { say "packet $PACKET: no ok audit"; exit 2; }
aws s3 ls $S3C/packets/$PACKET/items.pkl > /dev/null || { say "packet $PACKET not on S3"; exit 2; }
until aws s3 ls $S3C/runs/$S1/proxy/final.json > /dev/null 2>&1; do sleep 300; done
say "$S1 finished (final proxy on S3)"
cd $ROOT
if ! aws s3 ls $S3C/runs/$S1/probe/probe_final.json > /dev/null 2>&1; then
  .venv/bin/python -m activation.cloud.sky status $NODE 2>&1 | grep -q "^$NODE .* UP " || \
    { say "node $NODE not up: creating it"; .venv/bin/python -m activation.cloud.sky setup --name $NODE --gpu rtx6000pro --spot || say "setup failed"; }
  say "probe of $S1 on $NODE"
  EC2_SCRIPT=probe_ec2.sh bash $E/ec2_launch.sh $NODE $S1 - PROBE_REFERENCES=1 > $LOGS/${S1}_probe.log 2>&1; say "probe returned $?"
fi
say "starting $S2 on $NODE (packet $PACKET, start pair ${S1}_final)"
exec bash $E/ec2_keep.sh $NODE $S2 $PACKET CMP_INIT=${S1}_final CMP_LOSS=sft CMP_PANEL=panel_dev75_tv CMP_TEACHER=teacher_dev75_tv.pt \
  CMP_REF=proxy_ref_tv.json CMP_REPORT_EVERY=80 CMP_CKPT_EVERY=25 CMP_BUNDLE=cmp_csail_code_v2
