#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 SRC_ROOT DST_ROOT NUM_GROUPS [START_INDEX]"
  exit 1
fi

SRC_ROOT="$1"
DST_ROOT="$2"
NUM_GROUPS="$3"
START_INDEX="${4:-0}"

if ! [[ "$NUM_GROUPS" =~ ^[0-9]+$ ]] || [ "$NUM_GROUPS" -lt 1 ]; then
  echo "NUM_GROUPS must be a positive integer"
  exit 1
fi

mkdir -p "$DST_ROOT"

idx="$START_INDEX"

link_pair() {
  local tar_path="$1"
  local idx="$2"
  local dst_dir="$3"

  local tar_name="shard_$(printf '%08d' "$idx").tar"
  local counts_name="shard_$(printf '%08d' "$idx")_counts.pt"

  local counts_path="${tar_path%.tar}_counts.pt"
  if [ ! -f "$counts_path" ]; then
    echo "Missing counts file for $tar_path" >&2
    exit 1
  fi

  ln -s "$tar_path" "$dst_dir/$tar_name"
  ln -s "$counts_path" "$dst_dir/$counts_name"
}

mapfile -t chunk_dirs < <(find "$SRC_ROOT" -maxdepth 1 -mindepth 1 -type d)
if [ ${#chunk_dirs[@]} -eq 0 ]; then
  chunk_dirs=("$SRC_ROOT")
fi

IFS=$'\n' sorted_chunks=($(printf '%s\n' "${chunk_dirs[@]}" | sort -V))
unset IFS

total_chunks=${#sorted_chunks[@]}
pad_width=${#NUM_GROUPS}

base_size=$(( total_chunks / NUM_GROUPS ))
remainder=$(( total_chunks % NUM_GROUPS ))

offset=0
for (( g=0; g<NUM_GROUPS; g++ )); do
  this_size=$base_size
  if [ $g -lt $remainder ]; then
    this_size=$((this_size + 1))
  fi

  group_dir="$DST_ROOT/group_$(printf '%0*d' "$pad_width" "$g")"
  mkdir -p "$group_dir"

  if [ $this_size -eq 0 ]; then
    continue
  fi

  start=$offset
  end=$((offset + this_size))

  for (( i=start; i<end; i++ )); do
    chunk="${sorted_chunks[$i]}"
    shopt -s nullglob
    for tar_file in "$chunk"/shard_*.tar; do
      echo "Linking $(basename "$chunk")/$(basename "$tar_file") -> $(basename "$group_dir")/shard_$(printf '%08d' "$idx").tar"
      link_pair "$tar_file" "$idx" "$group_dir"
      idx=$((idx + 1))
    done
    shopt -u nullglob
  done

  offset=$end
done

count=$(find "$DST_ROOT" -type l -name 'shard_*.tar' | wc -l)
if [ "$count" -gt 0 ]; then
  echo "Done. Created $count symlinked shard pairs in $DST_ROOT across $NUM_GROUPS group directories."
else
  echo "No shards were linked. Check that $SRC_ROOT contains chunk directories with shard_*.tar files." >&2
  exit 1
fi


