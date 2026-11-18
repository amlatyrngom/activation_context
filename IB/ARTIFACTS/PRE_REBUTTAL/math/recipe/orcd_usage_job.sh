#!/bin/bash
#SBATCH --job-name=musage
#SBATCH --cpus-per-task=4 --mem=32G --time=01:00:00
# ORCD storage + recovery-slot check for the math recipe (2026-11-18), a CPU job (the login-node rule forbids du/quota/find there).
# ORCD scratch has a per-user quota of about 1.0 TB that df does not show (ROOT_CANON "Quotas you cannot see"): sum ~/orcd/scratch/*.
#   sbatch ... jobs/orcd_usage_job.sh [RUN ...]    RUN: also check runs/RUN's published recovery slot (record, sizes, torch.load of each file)
export PATH=/usr/local/bin:/usr/bin:/bin
S=/home/ngom/orcd/scratch; M=$S/activation/math1; PY=$S/activation/rq_v1/code/.venv/bin/python
date -u
echo "== scratch total (quota about 1.0 TB)"; du -sh --apparent-size $S/ 2>/dev/null; du -sh $S/ 2>/dev/null
echo "== $S/*"; du -sh $S/* 2>/dev/null | sort -h | tail -12
echo "== activation/*"; du -sh $S/activation/* 2>/dev/null | sort -h | tail -12
echo "== math1/runs"; du -sh $M/runs/* $M/runs/*/sync/* 2>/dev/null | sort -h | tail -20
for RUN in "$@"; do
  R=$M/runs/$RUN/sync; echo "== slot $RUN"; cat $R/AC_MODELS/rag_poc/partial_epoch.json 2>&1; echo
  ls -la $R/AC_MODELS/rag_poc/ $R/LORAS/rag_reader/ 2>&1 | head -20
  step=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" $R/AC_MODELS/rag_poc/partial_epoch.json 2>/dev/null)
  for d in $R/AC_MODELS/rag_poc/epoch_001_step_$(printf %05d ${step:-0}) $R/LORAS/rag_reader/epoch_001_step_$(printf %05d ${step:-0}); do
    ls -la $d 2>&1
    for f in $d/*.pt $d/*.safetensors; do [ -f "$f" ] || continue
      $PY -I -c "
import sys, torch
p = sys.argv[1]
if p.endswith('.safetensors'):
    from safetensors.torch import load_file; t = load_file(p); print('ok', p, len(t), 'tensors')
else:
    t = torch.load(p, map_location='cpu', weights_only=False); print('ok', p, type(t).__name__, len(t) if hasattr(t, '__len__') else '')
" $f 2>&1 | tail -1; done; done
done
