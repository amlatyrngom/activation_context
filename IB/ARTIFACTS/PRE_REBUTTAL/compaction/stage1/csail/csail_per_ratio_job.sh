#!/bin/bash
#SBATCH --job-name=cmp_ratio
# A per-ratio stage-1 run on CSAIL (2026-11-17), H200 or GH200. CSAIL holds jobs submitted with `--export=NIL,VAR=...` ("user env retrieval
# failed requeued held"), so they are submitted with plain --export=NIL and this script sets the recipe's options (run M's recipe; the
# per-ratio packet fixes r): start pair R_cmp_lcrec_final, bundle cmp_csail_code_v2, corrected dev-75 panel every 80 updates, recovery
# slot every 40, final proxy against proxy_ref_tv. x86_64 (H200): cmp_stage1_job.sh with the checked venv; aarch64 (GH200):
# csail_gh200_job.sh (fresh aarch64 venv built in the job first).
#   csail_stage.sh sbatch cmp_csail_code_v2 --export=NIL --account=quanta --partition=quanta-h200|quanta-gh200 --qos=quanta-main --gpus=1 \
#     --requeue --time=4-00:00:00 jobs/csail_per_ratio_job.sh RUN PACKET
set -uo pipefail
export PATH=/usr/local/bin:/usr/bin:/bin
W=/data/scratch-fast/ngom/cmp_csail
export CMP_INIT=R_cmp_lcrec_final CMP_BUNDLE=cmp_csail_code_v2 CMP_PANEL=panel_dev75_tv CMP_TEACHER=teacher_dev75_tv.pt CMP_REF=proxy_ref_tv.json \
  CMP_REPORT_EVERY=80 CMP_CKPT_EVERY=40
if [ "$(uname -m)" = aarch64 ]; then exec bash $W/jobs/csail_gh200_job.sh "$@"; else exec bash $W/jobs/cmp_stage1_job.sh "$@"; fi
