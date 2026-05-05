#!/usr/bin/env bash
set -eo pipefail

TITLES_FILE="/home/zs7353/resume_validity/data/selected_titles_25.txt"

OUT_BASE="/home/zs7353/resume_validity/data/harvest_pool_t25"
LOG_DIR="/home/zs7353/resume_validity/logs/harvest"
COMPANIES_FILE="/home/zs7353/resume_validity/data/companies_greenhouse.txt"
MAX_JOBS=400
TARGET=200

mkdir -p "$OUT_BASE" "$LOG_DIR"

# Require GOOGLE_API_KEY for Gemini qualification extraction
if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "GOOGLE_API_KEY is not set. Export it before running this script."
  exit 1
fi

PY_BIN="/home/zs7353/.conda/envs/resume_fting/bin/python"
SETUP='export PYTHONUNBUFFERED=1'
SESS_PREFIX="harvest_t25"

if [[ ! -f "$TITLES_FILE" ]]; then
  echo "Missing titles file: $TITLES_FILE" >&2
  exit 1
fi

if [[ ! -f "$COMPANIES_FILE" ]]; then
  echo "Missing companies file: $COMPANIES_FILE" >&2
  exit 1
fi
COMPANIES=$(tr '\n' ',' < "$COMPANIES_FILE" | sed 's/,$//')

idx=0
while IFS= read -r title || [[ -n "$title" ]]; do
  [[ -z "$title" ]] && continue
  slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
  sess=$(printf "%s_%02d" "$SESS_PREFIX" "$idx")
  out_dir="$OUT_BASE/$slug"
  log_file="$LOG_DIR/${slug}.log"
  mkdir -p "$out_dir"
  tmux new-session -d -s "$sess" bash -lc "export GOOGLE_API_KEY=${GOOGLE_API_KEY}; ${SETUP}; ${PY_BIN} -u -m resume_validity.scrape.harvest_pass_postings --companies ${COMPANIES} --title_filter \"${title}\" --max_jobs_per_company ${MAX_JOBS} --per_role_target ${TARGET} --require_basic 0 --require_bonus 0 --out_dir ${out_dir} --no_progress > ${log_file} 2>&1"
  echo "Launched $sess -> $out_dir (title=\"$title\")"
  idx=$((idx+1))
done < "$TITLES_FILE"

tmux ls | grep harvest_ || true


