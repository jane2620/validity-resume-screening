#!/usr/bin/env bash

# NOTE: not a slurm script; do not add -euo pipefail per user preference.

INPUT="${2:-/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_with_names_and_jd.jsonl}"
MODEL="${1:-meta-llama/llama-3.1-8b-instruct}"  # pass model as arg; default llama-3.1-8b
USERNAME="${USER:-zs7353}"
SEED="${SEED:-42}"
SHARDS="${SHARDS:-10}"
SUFFIX="${3:-named}"  # used in session naming

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY not set" >&2
  exit 1
fi

for i in $(seq 0 $((SHARDS-1))); do
  safe_model=$(echo "$MODEL" | tr '/.-' '___')
  sess="eval_${i}_${safe_model}_${SUFFIX}"
  if tmux has-session -t "$sess" 2>/dev/null; then
    tmux kill-session -t "$sess"
  fi
  tmux new-session -d -s "$sess" -- bash -lc "python3 -u -m resume_validity.eval.evaluate_model --model_name '$MODEL' --username '$USERNAME' --seed $SEED --input '$INPUT' --batch_size 4 --num_samples 1 --format_suffix '_${SUFFIX}' --shard_index $i --shard_total $SHARDS"
  echo "launched $sess"
done

# list our sessions
tmux ls | sed -n 's/^\(eval_[^:]*\):.*/\1/p'


