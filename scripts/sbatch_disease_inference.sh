#!/bin/bash
#SBATCH --job-name=ccp_disease_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/kkolpetinou/slurm-%x-%j.out
#SBATCH --error=/home/kkolpetinou/slurm-%x-%j.err

# Inference-only re-run of the CCP Disease pipeline. The 14.5h job 312 trained everything
# (bi-encoder, top-50 candidates, cross-encoder all saved on disk) then crashed at the LAST
# step (run_bio_benchmark+.py) on a missing `flair` import. ner.py is now patched to make
# flair optional (NER path is never hit -- mentions are given). This re-runs ONLY inference
# to produce the end-to-end test-NIL InR@k. Driver copy has train/rep/eval/cross flags = false.

set -uo pipefail
source /home/kkolpetinou/miniconda3/etc/profile.d/conda.sh
conda activate onto38
export DEEPONTO_JVM_MEM=8g
export TOKENIZERS_PARALLELISM=false

REPO=/home/kkolpetinou/calibrated-concept-placement/repo
cd "$REPO"

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date -Iseconds)"
LOG="$REPO/results_log_Disease_INFERENCE.txt"
chmod +x ./Edge-Bi-enc+Cross-enc_INFERONLY.sh
stdbuf -oL -eL ./Edge-Bi-enc+Cross-enc_INFERONLY.sh Disease true 50 25 16 false true 200000 > "$LOG" 2>&1
rc=$?
echo "Inference driver exit code: $rc  ($(date -Iseconds))  log: $LOG"
exit $rc
