#!/usr/bin/env bash
set -eo pipefail

OUT_DIR="/home/zs7353/resume_validity/data/reconstructed_spec_harvest"
HARV_DIR="/home/zs7353/resume_validity/data/harvest_pool_t25"
SHARDS=${SHARDS:-20}

mkdir -p "$OUT_DIR"

# Optional: load modules and conda if available
PY_BIN="/home/zs7353/.conda/envs/resume_fting/bin/python"
SETUP='export PYTHONUNBUFFERED=1; export GOOGLE_API_KEY='"${GOOGLE_API_KEY}"

for i in $(seq 0 $((SHARDS-1))); do
  LOG=$(printf "%02d" "$i")
  tmux new-session -d -s recon_h_${i} "bash -lc '$SETUP; ${PY_BIN} -u -m resume_validity.build.reconstruct_dataset_spec --harvest_dir ${HARV_DIR} --offline --out_dir ${OUT_DIR} --shard_index ${i} --shard_total ${SHARDS} > ${OUT_DIR}/log_shard_${LOG}.txt 2>&1'"
  echo "Launched recon_h_${i} -> ${OUT_DIR}/log_shard_${LOG}.txt"
done

tmux ls | grep recon_h_ || true
