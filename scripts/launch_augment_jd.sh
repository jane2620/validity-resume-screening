#!/usr/bin/env bash

IN="/home/zs7353/resume_validity/data/pairs_from_harvest/pairs_all.jsonl"
OUT_DIR="/home/zs7353/resume_validity/data/pairs_from_harvest/jd_augmented"
SHARDS=40

mkdir -p "$OUT_DIR/shards"

# Split into shards deterministically by line number modulo
# Create shard files
for i in $(seq 0 $((SHARDS-1))); do
  : > "$OUT_DIR/shards/shard_$(printf "%02d" $i).jsonl"
done

# Demux lines into shard files
lineno=0
while IFS= read -r line; do
  idx=$((lineno % SHARDS))
  printf "%s\n" "$line" >> "$OUT_DIR/shards/shard_$(printf "%02d" $idx).jsonl"
  lineno=$((lineno+1))
done < "$IN"

echo "Launching $SHARDS tmux sessions for augmentation..."
for i in $(seq 0 $((SHARDS-1))); do
  shard=$(printf "%02d" $i)
  sess="augment_jd_${shard}"
  in_shard="$OUT_DIR/shards/shard_${shard}.jsonl"
  out_shard="$OUT_DIR/shards/shard_${shard}_with_jd.jsonl"
  tmux new-session -d -s "$sess" -- bash -lc "python3 -u -m resume_validity.analysis.augment_job_descriptions --input '$in_shard' --output '$out_shard' --timeout 20"
  echo "launched $sess"
done

echo "Active sessions:"
tmux ls -F '#S' 2>/dev/null | grep -E '^augment_jd_' || true
