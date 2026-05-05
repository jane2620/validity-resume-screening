#!/usr/bin/env bash
set -u

ROOT="/home/zs7353/resume_validity/data/harvest_pool_t25"
OUTDIR_DEFAULT="/home/zs7353/resume_validity/data/pairs_from_harvest_gpt5"
OUTDIR="${OUTDIR:-$OUTDIR_DEFAULT}"
NAMES="/home/zs7353/resume-fting/data/names.json"
SESS_PREF="${SESSION_PREFIX:-gpt5_vdem}"

mkdir -p "$OUTDIR"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set; export it before running." 1>&2
  exit 1
fi

# Enumerate roles
mapfile -t ROLES < <(ls -1 "$ROOT" | sort)

idx=0
for role in "${ROLES[@]}"; do
  sess="${SESS_PREF}_${idx}"
  outfile="$OUTDIR/validity_demographics_${role}.jsonl"
  tmux new-session -d -s "$sess" "OPENROUTER_API_KEY=$OPENROUTER_API_KEY python3 -u -m resume_validity.build.build_pairs_gpt5_validity_demographics --harvest_root $ROOT --names_path $NAMES --output $outfile --role $role --seed $((42+idx))"
  echo "Launched tmux session $sess for role=$role -> $outfile"
  idx=$((idx+1))
done

tmux ls || true
