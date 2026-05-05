#!/usr/bin/env bash

# Do not use -euo pipefail per user preference. Always show progress.

INPUT=${1:?"Usage: $0 <pairs_jsonl> <suffix>"}
SUFFIX=${2:-run}
SHARDS=${SHARDS:-20}

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
  echo "Launching forced-choice eval for $MODEL -> $((SHARDS*2)) sessions (validity + validity_demographics)"
  EXPERIMENTS=(
    "validity"
    "validity_demographics"
  )
  for EXP in "${EXPERIMENTS[@]}"; do
    for i in $(seq 0 $((SHARDS-1))); do
      sess="noab_${safe_model}_${EXP}_${i}_${SUFFIX}"
      if tmux has-session -t "$sess" 2>/dev/null; then
        continue
      fi
      tmux new-session -d -s "$sess" -- bash -lc "python3 -u -m resume_validity.eval.evaluate_model_no_abstain --model_name '$MODEL' --seed 42 --input '$INPUT' --batch_size 4 --num_samples 1 --format_suffix '_${SUFFIX}' --shard_index $i --shard_total $SHARDS --filter_experiment_type ${EXP}"
      echo "  launched $sess"
    done
  done
  cnt=$(tmux ls -F '#S' 2>/dev/null | grep -c "^noab_${safe_model}_.*_${SUFFIX}$")
  echo "  -> active sessions for ${MODEL}: $cnt"
done

echo
echo "All no-abstain eval sessions with suffix '${SUFFIX}':"
tmux ls -F '#S' 2>/dev/null | grep "_${SUFFIX}$" || true


