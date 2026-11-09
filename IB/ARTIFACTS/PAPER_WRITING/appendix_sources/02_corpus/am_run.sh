#!/bin/bash
# AM reference runs on EC2 GPU 1, sequential. Mirrors scripts/main/qwen-longhealth5.sh flag logic.
# usage: am_run.sh "<name n_articles start compute_stats methods target_size query_config algorithm_config use_budget dataset>" ...
B=/root/am_v1; cd $B/compaction
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$B/hf HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=$B/cache/triton VLLM_CACHE_ROOT=$B/cache/vllm
MODEL=Qwen/Qwen3-4B-Instruct-2507
budget_path="head_budget_optimization/head_budgets/Qwen3-4B-Instruct-2507/optimized_agnostic.json"
for config in "$@"; do
  read -r name n_articles start_article compute_stats methods target_size query_config algorithm_config use_budget dataset export <<< "$config"
  entry="-m evaluation.run_qa_evaluation"; [ "$export" = "1" ] && { export AM_EXPORT_DIR=$B/export/$name; entry="/root/am_v1/kit/am_export.py"; } || unset AM_EXPORT_DIR
  budget_flag=""; [ "$use_budget" = "1" ] && budget_flag="--precomputed-budget-path $budget_path --max-ratio-per-head 0.75"
  log_dir=$B/results/${name}_a${start_article}; mkdir -p $log_dir
  methods_formatted=$(echo "$methods" | tr ',' ' ')
  echo "$(date -u +%FT%TZ) START $name ($dataset)"
  s=$(date +%s)
  $B/venv/bin/python -u $entry --name "$name" --dataset-name $dataset --n-articles "$n_articles" --start-article "$start_article" \
    --compute-stats "$compute_stats" --methods $methods_formatted --target-size "$target_size" --query-config "$query_config" \
    --algorithm-config "$algorithm_config" --model-name $MODEL $budget_flag --chunking longhealth --batch-size 5 --log-dir $log_dir --max-model-len 131072 \
    > $B/logs/${name}_a${start_article}.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) END $name rc=$rc elapsed=$(( $(date +%s) - s ))s"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
done
