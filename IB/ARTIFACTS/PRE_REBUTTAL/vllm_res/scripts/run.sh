#!/bin/bash
# One validation stage on the node (sky job):  bash /root/vllm_res/run.sh hf|stock|res|agent|compare [extra args]
# hf: HF reference (rq_eval_job.sh environment); stock / res: vllm_run.py; compare: compare.py. Outputs under /root/vllm_res/out.
set -uo pipefail
V=/root/vllm_res; C=$V/code/sec35_code_v5; PY=${UV_PROJECT_ENVIRONMENT:-/opt/activation/.venv}/bin/python
export PYTHONUNBUFFERED=1 TZ=UTC TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/root/sky_workdir CQA_BENCH=$C/bench SEC45_RQ_KIT=$C/rq_kit RQ_THINKING=0 QA4B_READER=Qwen/Qwen3.5-9B QA4B_SIDE=Qwen/Qwen3.5-2B QA4B_RANK=128
export SEC45_ENCODER=Qwen/Qwen3.5-2B SEC45_RES_LAYERS=all SEC45_K=20,100,200 SEC45_LINEAR_SKIP=1 QA4B_LAYERS=3,7,11,15,19,23,27,31,35 QA4B_KV=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cd $V
stage=$1; shift
case $stage in
  hf)      $PY $V/hf_reference.py --code $C --data $V/data --pair $V/pair/ac $V/pair/reader --out $V/out/ref "$@" 2>&1 | tee $V/out/hf.log ;;
  stock|res) $PY $V/vllm_run.py --engine $stage --ref $V/out/ref --out $V/out/vllm "$@" 2>&1 | tee $V/out/$stage.log ;;
  agent)   $PY $V/agent_smoke.py $C $V/data $V/pair/ac $V/pair/reader 2>&1 | tee $V/out/agent.log ;;
  compare) $PY $V/compare.py $V/out/ref $V/out/vllm 2>&1 | tee $V/out/compare.log ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
exit ${PIPESTATUS[0]}
