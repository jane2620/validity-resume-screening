#!/usr/bin/env bash

# Launch evaluation for Reddit pairs across all 9 models with maximum parallelization
# 9 models × 20 shards = 180 parallel sessions
#
# NOTE: per user rules, do not add -euo pipefail; show progress; do not cancel jobs.

INPUT="${1:-/scratch/gpfs/KOROLOVA/zs7353/resume_validity/data/pairs_from_reddit/combined/pairs_all.jsonl}"
SUFFIX="${2:-reddit}"
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

total_sessions=0

for MODEL in "${MODELS[@]}"; do
    safe_model=$(echo "$MODEL" | tr '/.-' '___')
    echo "Launching $MODEL -> $SHARDS shards"
    
    for i in $(seq 0 $((SHARDS-1))); do
        sess="eval_reddit_${i}_${safe_model}_${SUFFIX}"
        
        # Kill existing session if any
        tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
        
        tmux new-session -d -s "$sess" -- bash -lc "cd /scratch/gpfs/KOROLOVA/zs7353 && export OPENROUTER_API_KEY=${OPENROUTER_API_KEY}; python3 -u -m resume_validity.eval.evaluate_model --model_name '$MODEL' --username '${USER:-zs7353}' --seed 42 --input '$INPUT' --batch_size 4 --num_samples 1 --format_suffix '_${SUFFIX}' --shard_index $i --shard_total $SHARDS"
        ((total_sessions++))
    done
    
    cnt=$(tmux ls -F '#S' 2>/dev/null | grep -c "^eval_reddit_[0-9]\+_${safe_model}_${SUFFIX}$")
    echo "  -> active sessions: $cnt"
done

echo ""
echo "========================================="
echo "Total sessions launched: $total_sessions"
echo ""
echo "Active eval sessions:"
tmux ls -F '#S' 2>/dev/null | grep "^eval_reddit_" | head -20
echo "... (showing first 20)"
echo ""
echo "To monitor progress:"
echo "  tmux ls | grep eval_reddit_ | wc -l"
echo ""
echo "Results will be in: /home/${USER:-zs7353}/resume_validity/results/"

