#!/usr/bin/env bash

# NOTE: per user rules, do not add -euo pipefail; show progress; do not cancel jobs.

ROOT="${1:-/home/zs7353/resume_validity/data/harvest_pool_t25}"
OUTDIR="${2:-/home/zs7353/resume_validity/data/pairs_from_harvest_claude_rel6}"
SHARDS="${3:-80}"
MODEL="${4:-anthropic/claude-sonnet-4}"

mkdir -p "$OUTDIR"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY not set" >&2
  exit 1
fi

for i in $(seq 0 $((SHARDS-1))); do
  sess="claude_pairs_${i}_of_${SHARDS}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux kill-session -t "$sess"
  fi
  cmd="python3 -u -m resume_validity.build.build_pairs_claude_from_harvest --harvest_dir $ROOT --out_dir $OUTDIR --shard_index $i --shard_total $SHARDS --model $MODEL"
  tmux new-session -d -s "$sess" -- bash -lc "$cmd > ${OUTDIR}/log_shard_$(printf %02d $i).txt 2>&1"
  echo "launched $sess"
done

echo "Active sessions:"
tmux ls -F '#S' 2>/dev/null | grep '^claude_pairs_' || true


