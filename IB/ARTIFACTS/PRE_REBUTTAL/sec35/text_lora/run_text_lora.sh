#!/bin/bash
# Text + reader-LoRA baseline (T_text_lora) on one EC2 RTX PRO 6000 node: fetch the HotpotQA task data from S3, train
# (text_lora_train.py), evaluate every saved adapter (text_lora_eval.py), and upload everything (adapters, held curve, stats,
# scores, logs) to S3 sec45_20261026/runs/T_text_lora/ every 15 minutes and at the end.
#   bash run_text_lora.sh            the full run (4,000 x 8, ~ the T runs' recipe; see text_lora_train.py)
#   SMOKE=1 bash run_text_lora.sh    4 updates x 2 items, 4 held items, eval on the saved adapters (no S3 upload)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S3=s3://general-bucket-aln/activation-context-data/sec45_20261026
DATA=/root/sec45/data/hotpot_tasks_v1
NAME=T_text_lora; [ "${SMOKE:-0}" = 1 ] && NAME=T_text_lora_smoke
OUT=/root/sec45/runs/$NAME
if [ -x "$HOME/sky_workdir/.venv/bin/python" ]; then PY=("$HOME/sky_workdir/.venv/bin/python"); else PY=(uv run --frozen --project "$HOME/sky_workdir" python); fi   # as the T job scripts
S3PY=(uv run --no-project --with boto3 python)
say() { echo "$(date -u +%FT%TZ) [$NAME] $*"; }
mkdir -p $OUT $DATA
s3sync() {   # upload OUT (skipping files already there with the same size); never optimizer state (none is saved)
  "${S3PY[@]}" - "$OUT" "$S3/runs/$NAME" <<'EOF'
import boto3, os, sys
out, dest = sys.argv[1], sys.argv[2]; bucket, _, prefix = dest[5:].partition("/"); s3 = boto3.client("s3"); n = 0
have = {o["Key"]: o["Size"] for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/") for o in page.get("Contents", [])}
for root, _, files in os.walk(out):
    for f in files:
        p = os.path.join(root, f); key = prefix + "/" + os.path.relpath(p, out)
        if have.get(key) != os.path.getsize(p):
            s3.upload_file(p, bucket, key); n += 1
print(f"s3: {n} files uploaded to {dest}/")
EOF
}
if [ ! -f $DATA/train.jsonl ]; then
  "${S3PY[@]}" -c "
import boto3; s3 = boto3.client('s3'); b, p = 'general-bucket-aln', 'activation-context-data/sec45_20261026/data/hotpot_tasks_v1/'
for o in s3.list_objects_v2(Bucket=b, Prefix=p)['Contents']: s3.download_file(b, o['Key'], '$DATA/' + o['Key'][len(p):])
print('data fetched')" || { say "data fetch failed"; exit 1; }
fi
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX PRO 6000" || { say "not an RTX PRO 6000"; exit 1; }
export RQ_KIT=$HERE RQ_THINKING=0
if [ "${SMOKE:-0}" = 1 ]; then
  ARGS=(--steps 4 --epu 2 --held 4 --report-every 2 --warmup 1)
else
  ARGS=(); ( while true; do sleep 900; s3sync; done ) & SYNC=$!
fi
if [ ! -f $OUT/TRAINED ]; then
  say "train"; "${PY[@]}" $HERE/text_lora_train.py --data $DATA --out $OUT "${ARGS[@]}" > $OUT/train.log 2>&1 || { say "training failed (tail below)"; tail -30 $OUT/train.log; [ "${SMOKE:-0}" = 1 ] || { kill $SYNC; s3sync; }; exit 1; }
fi
say "eval"; "${PY[@]}" $HERE/text_lora_eval.py --data $DATA --out $OUT --base $([ "${SMOKE:-0}" = 1 ] && echo --max-tokens 256) > $OUT/eval.log 2>&1 || { say "eval failed (tail below)"; tail -30 $OUT/eval.log; }
[ "${SMOKE:-0}" = 1 ] || { kill $SYNC 2>/dev/null; date -u +%FT%TZ > $OUT/DONE; s3sync; }
say "done"; cat $OUT/T_text_lora.dev200.scores.json 2>/dev/null
