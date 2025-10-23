### clustering/ – brief workflow guide

This directory contains disjoint but composable tools for cluster-based data weighting and ordering. Below are the minimal usage patterns and goals for how they fit together.

### MGD pipeline (run_full_workflow.sh)

- Goal: given the latest training checkpoint and clustering setup, compute updated sampling weights for clusters.
- Pattern: sequential stages – collect features → compute whiteners (regularised Fisher) → aggregate/whiten target → collect per-dataset cosine sims → update logits/weights.
- Inputs: model checkpoint/UUID, tokenised WDS, OpenHermes target corpus, LoRA dims, iteration index.
- Key artifacts (paths are configured in the script):
  - grads.mmap → whiteners.npy → targets/sum_*.npy → hw_target.npy
  - OUT_DIR/dataset{i}.npz (cosine metrics) → updated_counts_iter{ITER}.json (new weights)
- Output: updated counts/weights JSON for the current iteration (plus diagnostics). This is the hand-off for dataset construction in the next section.

### From weights to a new dataset/run

1) Convert weights → ratios for `line_ratio_resampler.py` (manual)
- Produce a ratios JSON keyed by dataset<i>.
- Interpret ratios as per-cluster retention factors ("how much of this cluster to keep"), not as global mixture proportions. Manual conversion from weights/logits to these ratios is required at the moment.
- Input: updated weights JSON from the MGD pipeline. Output: ratios.json for resampling.

2) Sample and tokenize
- Use `line_ratio_resampler.py` to materialize sampled lines per cluster. Output is a processed directory of evenly sized shards (with optional cluster indices).
- Tokenize the sampled text in parallel via `ordered_tokenize_chunks.sh` to produce `shard_*.tar` token archives (and `*_counts.pt`).

3) Group/symlink token shards
- Use `link_chunked_ordered_tokens.sh` to symlink chunked token tars (and counts) into contiguous group directories (sequential `shard_%08d` naming) for downstream ordering.
- Input: chunk roots; Output: `group_*` directories with symlinked token/count shard pairs.

4) Order sequences by cluster target
- Current: `greedy_order_sparse_gpu.py` to produce ordered token shards + counts, aligned to the target ratios.
- Moving to: `greedy_order_sparse_gpu_clean.py` (preferred; better sparse handling and memory). Same goal/outputs.
- Inputs: grouped token shards + counts, optional ratio file; Outputs: `out_dir/tokens/shard_*.tar`, `out_dir/counts/shard_*.tar`, `manifest.jsonl`, and logs.

5) Concatenate if produced in chunks
- Use `merge_ds_symlink.py` to concatenate multiple ordered groups into a single dataset directory via symlinks and a fresh manifest.
- Input: one or more ordered group directories; Output: merged dataset with sequential shard naming and consolidated `manifest.jsonl`.

6) Register dataset and train
- Manually create the dataset artifact under `exp_data/datasets/` pointing to the final dataset directory. Then launch the next model run consuming it (configs in `training/`).
- Output: a new run UUID with metrics and checkpoints under `exp_data/models/`.

7) Set next-step checkpoint
- After the run finishes, manually update the checkpoint field in the resulting run UUID JSON under `exp_data/models/` to seed the next MGD iteration.

### Notes and pitfalls

- Ratios semantics: ratios are retention factors per cluster (relative keep amounts), not a full-dataset mixture distribution. Converting weights/logits → ratios is manual and impacts sampling volume directly.
- Chunking: ordering can be run per group/chunk to fit memory; merge later via symlinks to avoid copying.
- Counts alignment: ensure `*_counts.pt` tensors correspond 1:1 with token sequences when ordering. Mismatches will fail fast.
- GPU memory: prefer the `greedy_order_sparse_gpu_clean.py` variant for large corpora; it avoids dense copies and computes norms from CSR on-device.
- Determinism: where order matters across iterations, fix seeds in resampling and tokenization steps.

### Planned

- `run_full_workflow.sh` will be extended to accept multiple checkpoints to learn a curriculum (instead of a single set of static data weights). The downstream steps remain the same; only the weight update stage produces curriculum-aware weights.


