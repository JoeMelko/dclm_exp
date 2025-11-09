#!/bin/bash

# Parameters
INPUT_DIR=/mnt/denver/home/jmelko/curric/mgd_avg/iter2/docs_tok_merged
OUT_DIR=/mnt/denver/home/jmelko/curric/mgd_avg/iter2/docs_tok_ordered
SHARD_SIZE=64
TRUNCATE_MOD=512
TOTAL_TOKENS=1131944704
SCRIPT=greedy_order_sparse_gpu_curric.py
#RATIO_FILE=/mnt/denver/home/jmelko/curric/mgd_avg/iter2/schedule_gpu.json

# Loop ranges
I_MAX=1
J_MAX=8

for i in $(seq 0 $((I_MAX-1))); do
  for j in $(seq 0 $((J_MAX-1))); do
    GPU_ID=$((j))
    SUB_ID=$((8*i+j))
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python $SCRIPT \
      --input-dir $INPUT_DIR/sub$SUB_ID \
      --out-dir $OUT_DIR/sub$SUB_ID \
      --shard-size $SHARD_SIZE \
      --truncate-mod $TRUNCATE_MOD \
      --total-tokens $TOTAL_TOKENS &
  done
  wait
done