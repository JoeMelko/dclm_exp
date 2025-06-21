#!/bin/bash

# Run collect_grads_dc.py on all 8 GPUs simultaneously
for gpu in {0..7}; do
    CUDA_VISIBLE_DEVICES=$gpu python clustering/collect_grads_dc.py \
        --uuid 400m_3x-d=1024_l=24_h=8-warm=2000-lr=0p003-wd=0p033-cd=3e-05-bs=512-mult=3p0-seed=124-tokens=24696975360 \
        --wds-dir /home/jmelko/dclm_exp/rust_processing/tokshuf-rs/dclm_tokshuf \
        --lora-rank 128 &
done

# Wait for all background processes to complete
wait

echo "All GPU processes completed"
