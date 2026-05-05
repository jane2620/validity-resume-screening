#!/usr/bin/env bash

# Do not use -euo pipefail per user preference. Show progress; do not cancel jobs en masse.

INPUT="${1:-/home/zs7353/resume_validity/data/pairs_from_harvest_claude_rel6/pairs_all_rel6_with_jd.jsonl}"
SUFFIX="${2:-claude_rel6}"
SHARDS="${SHARDS:-20}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY not set" >&2
  exit 1
fi

MODELS=(
  "meta-llama/llama-3.1-8b-instruct"
  "meta-llama/llama-3.3-70b-instruct"
  "google/gemma-3-12b-it"
  "google/gemini-2.0-flash-001"
  "google/gemini-2.5-pro"
  "deepseek/deepseek-chat-v3.1"
  "openai/gpt-4o-mini"
  "openai/gpt-5"
  "anthropic/claude-sonnet-4"
)

for MODEL in "${MODELS[@]}"; do
  safe_model=$(echo "$MODEL" | tr '/.-' '___')
  echo "Launching (ensuring) $MODEL -> $SHARDS shards"
  for i in $(seq 0 $((SHARDS-1))); do
    sess="eval_${i}_${safe_model}_${SUFFIX}"
    if tmux has-session -t "$sess" 2>/dev/null; then
      continue
    fi
    tmux new-session -d -s "$sess" -- bash -lc "python3 -u -m resume_validity.eval.evaluate_model --model_name '$MODEL' --username '${USER:-zs7353}' --seed 42 --input '$INPUT' --batch_size 4 --num_samples 1 --format_suffix '_${SUFFIX}' --shard_index $i --shard_total $SHARDS"
    echo "  launched $sess"
  done
  # Count sessions for this model
  cnt=$(tmux ls -F '#S' 2>/dev/null | grep -c "^eval_[0-9]\+_${safe_model}_${SUFFIX}$")
  echo "  -> active sessions: $cnt"
done

echo
echo "All eval sessions with suffix '${SUFFIX}':"
tmux ls -F '#S' 2>/dev/null | grep "_${SUFFIX}$" || true


