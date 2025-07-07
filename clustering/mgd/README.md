## Multi-GPU clustering (MGD) workflow

The `mgd` toolkit processes large WebDataset archives in **four successive stages**. Each stage is implemented as a standalone script so you can pause, resume, or tweak parameters in-between.

1. **Collect projected features**  – `launch_collect_features.sh`
   • Spawns one `collect_features_dc.py` worker per GPU (default: 8 GPUs).
   • Each worker streams shards from `--wds-dir`, projects them to low-rank blocks and writes the result into a shared *memory-mapped* file created beforehand by `create_mmap_features.py`.

   ```bash
   ./launch_collect_features.sh \
       --wds-dir /data/tokenised_wds \
       --uuid 123e4567-e89b-12d3-a456-426614174000 \
       --shards-per-gpu 15 \
       --shard-size 1000 \
       --num-blocks 60 \
       --rank 32 \
       --out clustering/mgd/features.fp16
   ```

2. **Compute block-wise inverse-Fisher "whitener" matrices**  – `hessian.py`
   • Reads the mem-mapped gradients/features from Stage 1.
   • Clips each sample to the `--clip-percentile` L2-norm.
   • Solves a small ridge-regularised eigenproblem per block to obtain the inverse square root of the Fisher matrix.

   ```bash
   python -m dclm_exp.clustering.mgd.hessian \
       --mmap-path clustering/mgd/features.fp16 \
       --rank 32 \
       --num-blocks 60 \
       --dtype fp16 \
       --out-path clustering/mgd/whiteners.npy \
       --cond 1e4 \
       --clip-percentile 99.9 \
       --verbose
   ```

3. **Generate target representations**  – `launch_get_target.sh`
   • Again spawns one process per GPU, this time running `get_target.py`.
   • Each worker loads the whiteners from Stage 2, applies them to its chunk of data and stores the final target vectors (e.g. cosine-normalised, PCA-reduced) as WebDataset shards.

   ```bash
   ./launch_get_target.sh \
       --wds-dir /data/tokenised_wds \
       --whiteners-path clustering/mgd/whiteners.npy \
       --chunk-size 15 \
       --shard-size 1000 \
       --out-dir clustering/mgd/targets
   ```

4. **Compute pairwise cosine similarities for every dataset**  – `run_collect_all.sh`
   • Iterates over *all* first-level sub-directories in `PARENT_DIR` and launches `collect_cosine_sim_dc.py` on an available GPU, ensuring at most one job per GPU.
   • The simple PID-based queue makes it easy to saturate all GPUs without relying on `nvidia-smi` polling.

   ```bash
   ./run_collect_all.sh clustering/mgd/targets \
       --top-k 500 \
       --batch-size 8192 \
       --fp16
   ```

---

### Quick checklist
1. Prepare your WebDataset shards (tokenised, batched).
2. Run **Stage 1** to create the shared feature memmap.
3. Run **Stage 2** to compute whiteners (a few minutes per block).
4. Run **Stage 3** to obtain target vectors for clustering.
5. Run **Stage 4** to collect cosine-similarity statistics across all datasets.

All intermediate artefacts are plain `.npy` or `.mmap` files so you can inspect them with NumPy or reload them in PyTorch at any point.
