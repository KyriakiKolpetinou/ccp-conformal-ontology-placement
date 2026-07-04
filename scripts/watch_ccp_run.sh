#!/bin/bash
# Clean one-screen monitor for the CCP Disease reproduction (Slurm job 312).
# Usage: scripts/watch_ccp_run.sh            # one snapshot
#        scripts/watch_ccp_run.sh -w         # refresh every 30s (Ctrl-C to stop)
JOB=${JOB:-312}
LOG=/home/kkolpetinou/calibrated-concept-placement/repo/results_log_Disease_edge_biencoder_top50_200k_final.txt

snapshot() {
  echo "================ CCP job $JOB @ $(date '+%H:%M:%S') ================"
  # --- Slurm state ---
  line=$(squeue -j "$JOB" -h -o "%T | run %M | left %L | %R" 2>/dev/null)
  if [ -z "$line" ]; then
    echo "STATE : not in queue (finished or cancelled)"
    sa=$(sacct -j "$JOB" -n -o State,Elapsed,ExitCode 2>/dev/null | head -1)
    [ -n "$sa" ] && echo "sacct : $sa"
  else
    echo "STATE : $line"
  fi
  # --- GPU 0 = the card this job holds ---
  echo "GPU0  : $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0 2>/dev/null)"
  # --- current stage, inferred from which tqdm total is live + presence of cross-enc/inference ---
  last_total=$(grep -aoE "/[0-9]{4,} \[" "$LOG" 2>/dev/null | tail -1 | tr -dc '0-9')
  if   grep -aqiE "run_bio_benchmark|inference" "$LOG" 2>/dev/null; then stage="INFERENCE (final InR@k coming)"
  elif grep -aqiE "crossencoder|train_cross" "$LOG" 2>/dev/null;    then stage="CROSS-ENCODER (4 epochs)"
  elif grep -aqiE "enrich"   "$LOG" 2>/dev/null;                    then stage="EDGE ENRICHMENT (deeponto)"
  elif [ "$last_total" = "237826" ] || [ "$last_total" = "237825" ]; then stage="ENCODING 238k EDGE CATALOGUE / SEARCH"
  else stage="BI-ENCODER train/eval (live loop total=${last_total:-?})"; fi
  echo "STAGE : $stage"
  # --- live progress bar for the current loop (tqdm) ---
  bar=$(grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+(it/s|s/it)\]" "$LOG" 2>/dev/null | tail -1)
  echo "LOOP  : ${bar:-<no active progress bar>}"
  echo "LOG   : $(wc -l < "$LOG" 2>/dev/null) lines | $(du -h "$LOG" 2>/dev/null | cut -f1)"
  # --- final metrics if present ---
  metrics=$(grep -aiE "InR_(any|all)@|recall@[0-9]|insertion rate|acc@[0-9]" "$LOG" 2>/dev/null | grep -avE "tensor|device" | tail -8)
  if [ -n "$metrics" ]; then
    echo "----- METRICS (found in log) -----"
    echo "$metrics"
  fi
  echo "=================================================================="
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; snapshot; sleep 30; done
else
  snapshot
fi
