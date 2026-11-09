#!/bin/bash
# Select-then-answer v2 (Oct 27 15:20Z), after run_select.sh: the selection prompt asks for the most relevant first (rq_select_eval2.py).
#   sel{K}r   2B selector, at most 2 per call (the AC budget)
#   sel{K}r9  9B selector (the reader selecting for itself), at most 2 per call
#   sel{K}u   2B selector, uncapped (at most 20 per call; the reader budget grows)
# First (user, Oct 27 15:30Z): the section 4.5 twin, select4b_sec45.py (4B selects/summarizes, 4B reads; dev200 top20i).
set -u; P=/root/pr; cd $P
export PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONPATH=$P/code/kit_pr RQ_THINKING=0 HF_HUB_OFFLINE=1
PY="/root/sky_workdir/.venv/bin/python $P/code/kit_pr/rq_select_eval2.py --data $P/data --set eval500 --out-dir $P/out/select --ks 100 200 --modes sel --passes 3"
run() { echo "=== $(date -u +%FT%TZ) $*"; $PY "$@" >> $P/out/select2.log 2>&1 || echo "=== exit $? $*"; }
echo "=== $(date -u +%FT%TZ) sec45 4B select/summarize (dev200 top20i)"
/root/sky_workdir/.venv/bin/python $P/code/kit_pr/select4b_sec45.py --data $P/sec45data --out-dir $P/out/select4b_sec45 >> $P/out/select4b_sec45.log 2>&1 || echo "=== sec45 exit $?"
run --phase select --tag r
run --phase read --tag r
run --phase select --tag r9 --selector Qwen/Qwen3.5-9B
run --phase read --tag r9 --selector Qwen/Qwen3.5-9B
run --phase select --tag u --max-select 20 --select-tokens 128
run --phase read --tag u
echo "=== $(date -u +%FT%TZ) done"
