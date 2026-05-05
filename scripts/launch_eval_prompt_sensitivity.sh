#!/bin/bash
# Launch prompt sensitivity evaluations across 9 models and 2 prompt variants (human, llm)
# Total: 9 models × 2 variants × 40 shards = 720 tmux sessions

INPUT="${1:-/scratch/gpfs/KOROLOVA/zs7353/resume_validity/data/pairs_from_harvest/pairs_all_rel6_with_jd.jsonl}"
SHARDS="${SHARDS:-20}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set"
    exit 1
fi

# 9 models
MODELS=(
    "anthropic/claude-sonnet-4"
    "openai/gpt-5"
    "openai/gpt-4o-mini"
    "google/gemini-2.5-pro"
    "google/gemini-2.0-flash-001"
    "deepseek/deepseek-chat-v3-0324"
    "meta-llama/llama-3.3-70b-instruct"
    "meta-llama/llama-3.1-8b-instruct"
    "google/gemma-3-12b-it"
)

# 2 prompt variants
VARIANTS=(
    "human"
    "llm"
)

ROOT_DIR="/scratch/gpfs/KOROLOVA/zs7353"

total_sessions=0

for MODEL in "${MODELS[@]}"; do
    safe_model=$(echo "$MODEL" | tr '/.-' '___')
    
    for VARIANT in "${VARIANTS[@]}"; do
        echo "Launching $MODEL with prompt variant $VARIANT -> $SHARDS shards"
        
        for i in $(seq 0 $((SHARDS-1))); do
            sess="ps_${safe_model}_${VARIANT}_${i}"
            
            # Kill existing session if present
            tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
            
            tmux new-session -d -s "$sess" -- bash -lc "cd ${ROOT_DIR} && export OPENROUTER_API_KEY=${OPENROUTER_API_KEY}; python3 -u -m resume_validity.eval.evaluate_model_prompt_sensitivity --model_name '$MODEL' --prompt_variant '$VARIANT' --seed 42 --input '$INPUT' --batch_size 4 --num_samples 1 --format_suffix '_prompt_${VARIANT}' --shard_index $i --shard_total $SHARDS"
            ((total_sessions++))
        done
    done
done

echo ""
echo "=== Launched $total_sessions total tmux sessions ==="
echo "Monitor with: tmux ls | grep ps_"
echo "Check progress: tmux ls | grep ps_ | wc -l"

