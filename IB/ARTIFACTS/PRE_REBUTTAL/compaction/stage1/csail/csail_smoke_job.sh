#!/bin/bash
#SBATCH --job-name=cmp_smoke
#SBATCH --cpus-per-task=8 --mem=96G --time=4:00:00
#SBATCH --output=/data/scratch-fast/ngom/cmp_csail/logs/smoke_%j.out
# CSAIL H200 smoke of compaction stage 1 (2026-11-17), one GPU, runs in order inside one allocation:
#   1. SMOKE=1 on the s012 packet (full continuations, 20 updates) from R_cmp_lcrec_final; its first step computes the corrected dev-75
#      reference (proxy u00000 --references on panel_dev75_tv -> data/teacher_dev75_tv.pt + data/proxy_ref_tv.json), its last proxies ckpt/final
#   2. SMOKE=1 on the same items with continuations truncated to T = 4k, 2k, 8k target tokens (truncate_packet.py), timing only
# Submit through csail_stage.sh sbatch (pre-job check): ... --gres=gpu:1 jobs/csail_smoke_job.sh BUNDLE
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/data/scratch-fast/ngom/cmp_csail; BUNDLE=${1:?BUNDLE}
export CMP_BUNDLE=$BUNDLE CMP_INIT=R_cmp_lcrec_final CMP_PANEL=panel_dev75_tv CMP_TEACHER=teacher_dev75_tv.pt CMP_REF=proxy_ref_tv.json SMOKE=1
bash $W/jobs/cmp_stage1_job.sh smoke_h200_s012_full s1kl_both_c20_s012_v1 || echo "full smoke failed"
for t in 4k 2k 8k; do CMP_FINAL_PROXY=0 bash $W/jobs/cmp_stage1_job.sh smoke_h200_s012_t$t smoke_s012_t$t || echo "t$t smoke failed"; done
echo "$(date -u +%FT%TZ) smoke job done"
