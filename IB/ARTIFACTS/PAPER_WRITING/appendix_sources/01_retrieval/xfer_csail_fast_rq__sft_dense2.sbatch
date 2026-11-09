#!/bin/bash
#SBATCH --account=quanta --partition=quanta-h200 --qos=quanta-main --gpus=2 --cpus-per-task=32 --mem=400G
#SBATCH --time=1-00:00:00 --requeue --open-mode=append --job-name=rq_sftd2 --output=/data/scratch-fast/ngom/rq_v1/out/sftd_%j.out
# sbatch --export=NIL sft_dense.sbatch NAME [extra rq_train args] (user, Oct 20 ≈21:05Z: "Go with dense checkpoint. 1/10th is good. Use 2xgpus"):
# from dense_pow1_tgt48_256_lc3i (lc3i + 4,000 updates of passage / lookup / span reconstruction; interleaved; 1/10), question-first interleaved
# encoding, SFT on unsynthesized HotpotQA pairs (target "Final answer: <answer>"), each question's own distractor-setting context (2 gold + 8
# distractors) as ONE call at 1/10; 20,000 training questions + a disjoint 200-question held panel from hotpot_train_native; 2x H200 DDP;
# kit_s13: held rows add answer-only numbers (answer_nll, p_exact, prefix_nll, end_nll; user ≈21:30Z: most target tokens are not the answer).
# enc 5e-5 / reader 2e-5 constant, linear cooldown over the last 250 updates (canon); held NLL every 250 updates; no intermediate evals.
source /data/scratch-fast/ngom/rq_v1/env9.sh
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin RQ_THINKING=0 PYTHONPATH=${PYTHONPATH/kit_s7/kit_s13}
name=$1; shift; extra="$*"
P=$R/long_recon_csail/dense_pow1_tgt48_256_lc3i
export ACTIVATION_SYNC_ROOT=$F/sync/sftd_$name
echo "=== $(date -u +%FT%TZ) job $SLURM_JOB_ID restart $SLURM_RESTART_COUNT on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' '): sftd_$name"
[ "${SLURM_RESTART_COUNT:-0}" != 0 ] && [ -d $ACTIVATION_SYNC_ROOT ] && mv $ACTIVATION_SYNC_ROOT $ACTIVATION_SYNC_ROOT.r$SLURM_RESTART_COUNT
$PY -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 $F/kit_s13/rq_train.py --data $D --pair $P/AC_MODELS/rag_poc/epoch_001 $P/LORAS/rag_reader/epoch_001 \
   --loss sft --questions $D/hotpot_train_native.questions.jsonl --native --question-first --layout interleaved --items 20000 --held 200 --held-panel 200 \
   --name sftd_$name --ratio 0.1 --encoder-lr 5e-5 --reader-lr 2e-5 --schedule cooldown --cooldown-updates 250 \
   --report-every 250 --ckpt-every 250 --micro-batch 2 $extra; code=$?
echo "=== $(date -u +%FT%TZ) train exit $code"; exit $code
