#!/bin/bash
# Full H8 (Oct 17): quota check on both channel roots, material md5 check, then the build and the H200x2 training (afterok on the
# build, afterany on M1 2689361). Submits only if (compaction now + 17G) + TQA <= 195G. Same builder flags, init and recipe as H8a.
set -uo pipefail
W=/data/scratch-fast/ngom/activation/paper_compaction_v1; T=/data/scratch-fast/ngom/activation/paper_traj_qa_v1
cd $W
du -sh $W $T
C=$(du -s --block-size=1G $W | cut -f1); Q=$(du -s --block-size=1G $T | cut -f1)
echo "compaction ${C}G + 17G + TQA ${Q}G = $((C + 17 + Q))G (limit 195G)"
[ $((C + 17 + Q)) -le 195 ] || { echo "NOT SUBMITTED: over 195G"; exit 3; }
(cd material && md5sum -c material_h8full/MD5SUMS) || { echo "NOT SUBMITTED: md5 mismatch"; exit 4; }
[ ! -e packets/h8full_from_h5 ] || { echo "NOT SUBMITTED: packets/h8full_from_h5 exists"; exit 5; }
B=$(MATERIAL="material_h8full/h8full_material_pca.jsonl material_h8full/h8full_material_pcb.jsonl" OUT=h8full_from_h5 \
    PAIR=$W/pairs/h5_final KITF=kit_h7 \
    BUILD="--loss sft --held-repos 2 --nominal-prefix-tokens 0 --epochs 1 --sides training evaluation --exclude-tasks $W/material/prefixes_eval9c_200_index.json $W/material/prefixes_dev9_75_index.json $W/material/prefixes_eval_100_index.json --slices 3 --train deltas_off_training=true --train checkpoint_every_updates=null" \
    sbatch --parsable -J h8full_build jobs_h7/build_packets_h8.sbatch) || { echo "build sbatch failed"; exit 6; }
echo "build $B"
J=$(RUN=packets/h8full_from_h5 PAIR=$W/pairs/h5_final CHANNEL_QUOTA_GIB=170 \
    sbatch --parsable -d afterok:$B,afterany:2689361 --time=12:00:00 -J h8full_h200x2 jobs_h7/train_continuation_h200x2_h7.sbatch) || { echo "train sbatch failed"; exit 7; }
echo "train $J"
squeue -j $B,$J -o "%.10i %.16j %.10T %.20R" 2>/dev/null
