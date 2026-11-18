#!/bin/bash
# snap.sh RUN UPDATE NAME : make S3 stage1/runs/NAME/sync/ckpt/final/{ac,reader,meta.json} from RUN's recovery slot at UPDATE
set -euo pipefail
RUN=$1; U=$(printf %05d $2); NAME=$3
S=s3://general-bucket-aln/activation-context-data/pre_rebuttal_20261109/compaction/stage1/runs
SRC=$S/$RUN/sync; DST=$S/$NAME/sync/ckpt/final
aws s3 ls $SRC/AC_MODELS/rag_poc/epoch_001_step_$U/ac_modules.pt > /dev/null
aws s3 ls $SRC/LORAS/rag_reader/epoch_001_step_$U/adapter_model.safetensors > /dev/null
aws s3 cp --recursive --quiet $SRC/AC_MODELS/rag_poc/epoch_001_step_$U/ $DST/ac/ --exclude optimizer.pt --exclude teacher_targets.pt
aws s3 cp --recursive --quiet $SRC/LORAS/rag_reader/epoch_001_step_$U/ $DST/reader/
echo "{\"name\": \"$NAME\", \"source_run\": \"$RUN\", \"update\": $2, \"kind\": \"eval_snapshot\", \"mode\": \"res\", \"valid\": true}" | aws s3 cp - $DST/meta.json
aws s3 ls --recursive $DST/ | awk '{print $3, $4}' | sed "s#.*/ckpt/final/##"
