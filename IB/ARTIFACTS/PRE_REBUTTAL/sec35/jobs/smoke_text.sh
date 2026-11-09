#!/bin/bash
#SBATCH --job-name=s35_smoke
#SBATCH --cpus-per-task=8 --mem=96G --time=2:00:00
#SBATCH --output=s35_smoke_%j.out
# Smoke: text_lora_train resume (6 updates, slot every 2, killed after the update-4 slot, then resumed) vs an uninterrupted run,
# and text_lora_eval_hf on 64 dev1000 questions.
set -uo pipefail
export PATH=${PATH:-/usr/local/bin:/usr/bin:/bin}
W=/data/scratch-fast/ngom/sec35; PY=/data/scratch/ngom/activation/venv/bin/python; C=$W/code/sec35_code_v1; TD=$W/data/hotpot_tasks_v2
export HOME=$W/home HF_HOME=/data/scratch-fast/ngom/sec45/hf PYTHONUNBUFFERED=1 PYTHONPATH=$C:$C/rq_kit RQ_KIT=$C/rq_kit RQ_THINKING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ ! -f $C/MANIFEST.json ]; then (cd $W/code && sha256sum -c sec35_code_v1.tar.gz.sha256) && mkdir -p $C.tmp && tar -xzf $W/code/sec35_code_v1.tar.gz -C $C.tmp && mv -T $C.tmp $C; fi
S=$W/smoke; rm -rf $S; mkdir -p $S
A=(--data $TD --steps 6 --epu 2 --ckpt-every 2 --held 4 --report-every 2 --warmup 1)
$PY $C/text_lora/text_lora_train.py "${A[@]}" --out $S/full --name full > $S/full.log 2>&1; echo "full exit $?"
$PY $C/text_lora/text_lora_train.py "${A[@]}" --out $S/res --name full > $S/res1.log 2>&1 &
P=$!; while kill -0 $P 2>/dev/null; do grep -q '"update": 4' $S/res/resume/state.json 2>/dev/null && { kill $P; break; }; sleep 2; done; wait $P
echo "killed at slot: $(cat $S/res/resume/state.json)"
$PY $C/text_lora/text_lora_train.py "${A[@]}" --out $S/res --name full > $S/res2.log 2>&1; echo "resumed exit $?"; grep RESUME $S/res2.log
$PY - <<'PY'
import json
a=json.load(open("/data/scratch-fast/ngom/sec35/smoke/full/full.stats.json")); b=json.load(open("/data/scratch-fast/ngom/sec35/smoke/res/full.stats.json"))
print("full loss", a["loss"]); print("res  loss", b["loss"]); print("same length", len(a["loss"])==len(b["loss"]))
PY
cat $S/full/full.held.jsonl; echo; cat $S/res/full.held.jsonl
$PY $C/text_lora/text_lora_eval_hf.py --data $TD --split dev1000 --adapter $S/res/ckpt/step_00006 --out $S/eval.jsonl --limit 64 > $S/eval.log 2>&1; echo "eval exit $?"; tail -2 $S/eval.log
head -c 600 $S/eval.jsonl
