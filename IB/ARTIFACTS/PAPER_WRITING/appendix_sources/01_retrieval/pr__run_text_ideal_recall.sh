#!/bin/bash
# Perfect-recall evals (user, Oct 27): base Qwen3.5-9B reading BM25 top-K as text with golds injected (kit_pr rq_text_eval --inject-golds,
# same placement as the AC runs), K = 20, 100, then 200; eval500, thinking off, 3 passes, cap 12,288 = the frontier's text settings.
# Re-running skips finished arms.
set -u; P=/root/pr; cd ~/sky_workdir
export PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=$P/code/kit_pr
mkdir -p $P/out/text_i
echo "=== $(date -u +%FT%TZ) text start"
.venv/bin/python $P/code/kit_pr/rq_text_eval.py --data $P/data --set eval500 --out-dir $P/out/text_i --passes 3 --ks 20 100 200 --arms top \
   --inject-golds --no-thinking --kv-cache-dtype auto --max-tokens 12288 > $P/out/text_i.log 2>&1
echo "=== $(date -u +%FT%TZ) text exit $?"
