#!/usr/bin/env bash

ROOT="/home/zs7353/resume_validity/data/harvest_pool_t25"
NAMES="/home/zs7353/resume-fting/data/names.json"
ROLES_FILE="/home/zs7353/resume_validity/data/tail5_roles.txt"
OUTDIR_DEFAULT="/home/zs7353/resume_validity/data/pairs_from_harvest_gpt5_tail5_test"
OUTDIR="${OUTDIR:-$OUTDIR_DEFAULT}"
SESS_PREF="${SESSION_PREFIX:-gpt5_tail5}"
LOGDIR="$OUTDIR/logs"

# Basic checks
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set; export it before running." >&2
  exit 1
fi
if [[ ! -f "$NAMES" ]]; then
  echo "ERROR: names.json not found at $NAMES" >&2
  exit 1
fi
if [[ ! -f "$ROLES_FILE" ]]; then
  echo "ERROR: tail-5 roles file not found at $ROLES_FILE" >&2
  exit 1
fi

mkdir -p "$OUTDIR" "$LOGDIR"

idx=0
while IFS= read -r role; do
  [[ -z "$role" ]] && continue
  sess="${SESS_PREF}_${idx}"
  outfile="$OUTDIR/validity_demographics_${role}.jsonl"
  logfile="$LOGDIR/${sess}.log"
  seed=$((200 + idx))
  # Avoid stale sessions
  tmux has-session -t "$sess" 2>/dev/null && tmux kill-session -t "$sess"
  echo "Launching $sess for role=$role -> $outfile (log=$logfile)"
  tmux new-session -d -s "$sess" \
    "OPENROUTER_API_KEY=$OPENROUTER_API_KEY python3 -u -m resume_validity.build.build_pairs_gpt5_validity_demographics \
      --harvest_root $ROOT \
      --names_path $NAMES \
      --output $outfile \
      --role $role \
      --num_posts 5 \
      --independent_names \
      --seed $seed \
      2>&1 | tee -a $logfile"
  idx=$((idx+1))
done < "$ROLES_FILE"

tmux ls | grep "^${SESS_PREF}_" || true
