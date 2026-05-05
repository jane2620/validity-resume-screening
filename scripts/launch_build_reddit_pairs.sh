#!/usr/bin/env bash

# Launch script for building resume pairs from Reddit resumes with MAXIMUM parallelization
# Each resume-job combination runs in its own tmux session
#
# NOTE: per user rules, do not add -euo pipefail; show progress; do not cancel jobs.

# Configuration
ROOT="/scratch/gpfs/KOROLOVA/zs7353/resume_validity"
OUT_BASE="${ROOT}/data/pairs_from_reddit"
LOG_DIR="${ROOT}/logs/reddit_runs"

# Parameters
JOBS_PER_RESUME="${1:-3}"       # Number of job descriptions to randomly draw for each resume
MODEL="${2:-anthropic/claude-sonnet-4}"
USE_GEMINI="${3:-false}"        # Set to "true" to use Gemini instead of OpenRouter

mkdir -p "$OUT_BASE" "$LOG_DIR"

# Check API keys
if [[ "$USE_GEMINI" == "true" ]]; then
    if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "ERROR: GOOGLE_API_KEY not set (required for Gemini)" >&2
        exit 1
    fi
    API_FLAG="--use_gemini"
    echo "Using Gemini API"
else
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "ERROR: OPENROUTER_API_KEY not set" >&2
        exit 1
    fi
    API_FLAG="--model $MODEL"
    echo "Using OpenRouter with model: $MODEL"
fi

# Define runs: (category, resume_file, job_file, num_resumes)
# Finance: 10 resumes x 3 jobs = 30 combos
# SWE: 10 resumes x 3 jobs = 30 combos
# Total: 60 combos -> 60 parallel sessions

declare -A RUNS
RUNS["finance"]="real_world/reddit_finance_resumes.jsonl|top_jobs/financial_analyst/passing_financial_analyst.jsonl|30"
RUNS["swe"]="real_world/reddit_swe_resumes.jsonl|data/harvest_pool_t25/software_engineer/passing_software_engineer.jsonl|30"

total_sessions=0

for category in "${!RUNS[@]}"; do
    IFS='|' read -r resume_file job_file num_combos <<< "${RUNS[$category]}"
    
    out_dir="${OUT_BASE}/${category}"
    mkdir -p "$out_dir"
    
    echo ""
    echo "=== Launching $num_combos sessions for $category ==="
    echo "  Resume file: $resume_file"
    echo "  Job file: $job_file"
    echo "  Output: $out_dir"
    
    for shard_idx in $(seq 0 $((num_combos - 1))); do
        sess="reddit_${category}_${shard_idx}"
        log_file="${LOG_DIR}/${category}_shard_${shard_idx}.log"
        
        # Kill existing session if any
        tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
        
        # Build the command
        cmd="cd $ROOT && python3 -u -m build.build_pairs_from_reddit \
            --resume_file ${resume_file} \
            --job_file ${job_file} \
            --out_dir ${out_dir} \
            --jobs_per_resume ${JOBS_PER_RESUME} \
            --shard_index ${shard_idx} \
            --shard_total ${num_combos} \
            ${API_FLAG} \
            --seed 42"
        
        # Launch in tmux
        if [[ "$USE_GEMINI" == "true" ]]; then
            tmux new-session -d -s "$sess" -- bash -lc "export GOOGLE_API_KEY=${GOOGLE_API_KEY}; $cmd > ${log_file} 2>&1"
        else
            tmux new-session -d -s "$sess" -- bash -lc "export OPENROUTER_API_KEY=${OPENROUTER_API_KEY}; $cmd > ${log_file} 2>&1"
        fi
        
        ((total_sessions++))
    done
    
    echo "  Launched $num_combos sessions for $category"
done

echo ""
echo "========================================="
echo "Total sessions launched: $total_sessions"
echo ""
echo "Active Reddit pair-building sessions:"
tmux ls -F '#S' 2>/dev/null | grep '^reddit_' | head -20
echo "... (showing first 20)"
echo ""
echo "To monitor progress:"
echo "  tmux ls | grep reddit_ | wc -l    # count active sessions"
echo "  tail -f ${LOG_DIR}/<category>_shard_<N>.log"
echo ""
echo "To check completion:"
echo "  ls -la ${OUT_BASE}/finance/pairs_shard_*.jsonl | wc -l"
echo "  ls -la ${OUT_BASE}/swe/pairs_shard_*.jsonl | wc -l"
