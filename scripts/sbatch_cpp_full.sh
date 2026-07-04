#!/bin/bash
#SBATCH --job-name=ccp_cpp_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --output=/home/kkolpetinou/slurm-%x-%j.out
#SBATCH --error=/home/kkolpetinou/slurm-%x-%j.err

# CCP -- MM-S14-CPP full pipeline (2nd benchmark, for generalization). Mirrors the Disease run:
# Edge-Bi-encoder + top-50 search + enrichment + Edge-Cross-encoder + inference, top-50/25 seeds.
# CPP is bigger than Disease (1.4M train pairs, 626k edges) -> expect ~24-36h. Uses the CPP driver
# copy with the corrected NIL_ent_ind=173177 (= CPP atomic-concept count, analogous to Disease 64076).
# All env patches already applied (ner.py flair-optional, matplotlib, deeponto JVM env-var).
# Dumps go to a SEPARATE dir (dump tag = args.dataname = mm-{test,valid}-NIL collides with Disease).

set -uo pipefail
source /home/kkolpetinou/miniconda3/etc/profile.d/conda.sh
conda activate onto38
export DEEPONTO_JVM_MEM=8g
export TOKENIZERS_PARALLELISM=false
export CCP_DUMP_DIR=/home/kkolpetinou/calibrated-concept-placement/phase1/dumps_cpp

REPO=/home/kkolpetinou/calibrated-concept-placement/repo
cd "$REPO"
mkdir -p "$CCP_DUMP_DIR"
echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-?}, $(date -Iseconds)"
LOG="$REPO/results_log_CPP_edge_biencoder_top50_200k_final.txt"
chmod +x ./Edge-Bi-enc+Cross-enc_CPP.sh
stdbuf -oL -eL ./Edge-Bi-enc+Cross-enc_CPP.sh CPP true 50 25 16 false true 200000 > "$LOG" 2>&1
rc=$?
echo "CPP driver exit code: $rc ($(date -Iseconds)) log: $LOG  dumps: $CCP_DUMP_DIR"
exit $rc
