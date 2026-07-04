#!/bin/bash
#SBATCH --job-name=ccp_disease_bienc_top50
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/kkolpetinou/slurm-%x-%j.out
#SBATCH --error=/home/kkolpetinou/slurm-%x-%j.err

# Calibrated Concept Placement -- Phase 0 reproduction.
# Full Edge-Bi-encoder train + top-50 edge search + edge enrichment + Edge-Cross-encoder
# + inference, on MM-S14-Disease. Faithful to the repo's published run-example:
#   ./Edge-Bi-enc+Cross-enc.sh Disease true 50 25 16 false true 200000
# (context=true, top_k=50, seeds=25, batch=16, full bi-enc train, 200k pairs for cross-enc data)
#
# Queued to start automatically when a GPU frees (seg jobs own both GPUs now).
# Under Slurm --gres=gpu:...:1 the allocated GPU is exposed as index 0, which matches the
# driver's hardcoded CUDA_VISIBLE_DEVICES=0 -- no clash with the seg training.

set -uo pipefail  # NOT -e: the upstream driver is not written for errexit.

source /home/kkolpetinou/miniconda3/etc/profile.d/conda.sh
conda activate onto38

# deeponto JVM memory, read non-interactively by the patched deeponto/onto/ontology.py
# (eval_biencoder -> nn_prediction imports deeponto; without this it would prompt + abort).
export DEEPONTO_JVM_MEM=8g
export TOKENIZERS_PARALLELISM=false

REPO=/home/kkolpetinou/calibrated-concept-placement/repo
cd "$REPO"

echo "=========================================="
echo "Job:     $SLURM_JOB_NAME ($SLURM_JOB_ID)  Node: $(hostname)"
echo "Started: $(date -Iseconds)"
echo "GPU:     ${CUDA_VISIBLE_DEVICES:-unset} | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "Driver:  Edge-Bi-enc+Cross-enc.sh Disease true 50 25 16 false true 200000"
echo "=========================================="

LOG="$REPO/results_log_Disease_edge_biencoder_top50_200k_final.txt"
chmod +x ./Edge-Bi-enc+Cross-enc.sh
stdbuf -oL -eL ./Edge-Bi-enc+Cross-enc.sh Disease true 50 25 16 false true 200000 > "$LOG" 2>&1
rc=$?

echo "=========================================="
echo "Driver exit code: $rc"
echo "Finished: $(date -Iseconds)"
echo "Detailed log: $LOG"
echo "=========================================="
exit $rc
