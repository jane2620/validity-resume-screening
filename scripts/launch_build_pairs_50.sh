#!/usr/bin/env bash
set -u

INPUT_DIR=${1:-/home/zs7353/resume_validity/data/harvest_pool_t25}
OUT_DIR=${2:-/home/zs7353/resume_validity/data/pairs_from_harvest}
SHARDS=${3:-50}

mkdir -p "$OUT_DIR"

for i in $(seq 0 $((SHARDS-1))); do
  sess="pairs_${i}_of_${SHARDS}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux kill-session -t "$sess"
  fi
  cmd="python3 -u -m resume_validity.build.build_pairs_from_harvest --harvest_dir $INPUT_DIR --out_dir $OUT_DIR --shard_index $i --shard_total $SHARDS"
  # Use double quotes so $cmd expands
  tmux new-session -d -s "$sess" -- bash -lc "export GOOGLE_API_KEY=${GOOGLE_API_KEY:-AIzaSyB8qql3bwNxc3mi50uictjQLuRVVSRpD8k}; $cmd"
  echo "launched $sess"
done

echo "Active sessions:"
tmux ls -F '#S' 2>/dev/null | grep '^pairs_' || true
