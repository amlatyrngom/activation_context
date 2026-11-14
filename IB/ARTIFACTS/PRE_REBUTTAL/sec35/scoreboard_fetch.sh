#!/bin/bash
# Scoreboard inputs (2026-11-13): copy every per-question output (*.jsonl) the scoreboard reads into IB/TMP/PRE_REBUTTAL/scoreboard/outs/<source>/,
# from ORCD (side scorer, injected-set reads, 20I baselines, final reads) and S3 (in-run evals, EC2 zero-training reads). Idempotent (rsync / s3 sync).
#   bash IB/ARTIFACTS/PRE_REBUTTAL/sec35/scoreboard_fetch.sh && python3 -I IB/ARTIFACTS/PRE_REBUTTAL/sec35/scoreboard.py
set -uo pipefail
cd /home/ubuntu/activation_context
O=IB/TMP/PRE_REBUTTAL/scoreboard/outs; mkdir -p $O
SR=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/sec35/runs
source IB/ARTIFACTS/PRE_REBUTTAL/sec35/jobs/cluster_lib.sh; cluster orcd
for d in rq_eval20i rq_eval_kI_ac rq_eval_kI_text rq_eval_s2 rq_eval_fin rq_eval_fin20i rq_eval_e4 rq_eval_e4x/T_res_rq9e4fr_20I_KL_u05600 rq_eval_e4x/T_res_rq9e4_20I_KL_u05600; do
  n=orcd_${d//\//_}; mkdir -p $O/$n; xfer -a --include='*.jsonl' --include='*.summary.json' --exclude='*' $H:$W/$d/out/ $O/$n/ 2>/dev/null || true
done
for r in T_res_rq9e4_e2 T_res_rq9e4_20I_KL; do   # ORCD-trained runs' in-run evals
  mkdir -p $O/orcd_run_$r; xfer -a --include='*.jsonl' --include='*.summary.json' --exclude='*' $H:$W/runs/$r/eval/ $O/orcd_run_$r/ 2>/dev/null || true; done
for d in kI_eval_text kI_eval_ac; do aws s3 sync --only-show-errors --exclude '*' --include '*.jsonl' --include '*.summary.json' $SR/$d/out/ $O/s3_$d/; done
for r in T_res_rq9L_20I T_res_rq9fr_20I T_res_rq9L_20I_KL T_res_rq9fr_20I_KL X_rq9_text_20I \
         T_res_rq9L_kI T_res_rq9fr_kI T_res_rq9L_kI_KL T_res_rq9fr_kI_KL X_rq9_text_t2 T_res_rq9e4fr_20I_KL; do
  aws s3 sync --only-show-errors --exclude '*' --include '*.jsonl' --include '*.summary.json' $SR/$r/eval/ $O/s3_run_$r/
done
echo "fetched: $(find $O -name '*.jsonl' | wc -l) outputs"
