# Cluster-Aware Data-Curation & Model-Guided Data (MGD) Pipeline

> Sub-directory: `dclm_exp/clustering/`
>
> This pipeline turns **one Datacomp-LM local shard** (~ 40 B tokens / 40 M docs) into **tokenised per-cluster WebDataset shards** and then iteratively re-weights those clusters with a model-guided signal.

---
## 0 .  Requirements

* **Hardware**  • 1 node, 8 × H100-80 GB (other multi-GPU cards also work; tune batch sizes accordingly).
* **Python**    • 3.10 +
* **Packages**  • install once:

```bash
pip install "transformers>=4.51" sentence-transformers>=2.7.0 \
            accelerate datasets faiss-gpu numpy tqdm zstandard \
            webdataset blake3 torch scikit-learn scipy
```

---
## 1 .  Directory Layout (convention)

```
dclm_exp/
├── data/              # raw input shards
├── embeddings/        # dense vectors (.fp16)
├── clusters.tsv       # row_id <TAB> cluster_id
├── cluster_jsonl/     # raw docs per cluster
├── balanced/          # size-balanced jsonl per cluster (optional)
├── tok/               # tokenised WebDataset shards
├── openhermes/        # OpenHermes jsonl / features
└── ckpt/              # model checkpoints & whitening stats
```

---
## 2 .  Download the Shard (Datacomp-LM)

```bash
aws s3 cp --recursive \
  s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_01_of_10/local-shard_1_of_10/ \
  data/gs01_ls1
```

---
## 3 .  Embed Every Document

```bash
python embed.py \
  --shard-dir data/gs01_ls1 \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-size 2048 \
  --fp16 \
  --out embeddings/embeddings.fp16
```

---
## 4 .  K-Means Clustering

```bash
python kmeans.py \
  --embeddings embeddings/embeddings.fp16 \
  --n-clusters 10000 \
  --sample 3000000 \
  --out clusters.tsv
```
`clusters.tsv` holds `row_id<TAB>cluster_id` for **all** documents.

---
## 5 .  Extract Raw Docs per Cluster

```bash
export SHARD_ROOT=data/gs01_ls1
parallel -j 32 "\
  awk -v k={1} '\$2==k {print \$1}' clusters.tsv > tmp_ids_k.txt && \
  python extract_cluster.py \
    --row-ids tmp_ids_k.txt \
    --shard-root $SHARD_ROOT \
    --out-jsonl cluster_jsonl/cluster_{1}.jsonl" ::: $(seq 0 9999)
```

---
## 6 .  (Optional) Balance Cluster Sizes

```bash
python balanced_resample.py \
  --indir  cluster_jsonl/cluster_{k}.jsonl \
  --outdir balanced/cluster_{k} \
  --n      250_000      # target rows per cluster
```
Repeat for each `k` (GNU parallel works well).

---
## 7 .  Tokenise & Shuffle (WebDataset)

```bash
cargo run --release -- \
   --input balanced/cluster_{k} \
   --output tok/cluster_{k} \
   --tokenizer EleutherAI/gpt-neox-20b \
   --seqlen 2049 --wds-chunk-size 8192
```
Run once per cluster (`k`).

---
## 8 .  Process OpenHermes for MGD

```bash
# download
wget https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations/raw/main/openhermes_2.jsonl.gz
gzip -d openhermes_2.jsonl.gz -c > openhermes/openhermes.jsonl

# embed (same model as above)
python embed.py \
  --shard-dir openhermes \
  --input-json openhermes.jsonl \
  --out openhermes/features.fp16
```

---
## 9 .  Model-Guided Data (MGD) Loop

Below is **one** outer iteration.  Wrap B→E in a bash or Python loop for multiple rounds.

### A.  Initialise Cluster Logits

```python
import numpy as np, pandas as pd

df = pd.read_csv("clusters.tsv", sep="\t", names=["row", "cid"])
counts = df.groupby("cid").size().reindex(range(10000), fill_value=0)
logits = np.log(counts + 1e-6)  # avoid -inf
np.save("mgd_logits.npy", logits)
```

### B.  Train with Current Weights

```bash
torchrun --nproc_per_node 8 train.py \
  --train-shards "tok/cluster_{cid}/**/*.tar" \
  --cluster-logits mgd_logits.npy \
  --output-dir ckpt/iter0 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 32 \
  --lr 2e-5 --warmup 0.03 --epochs 1
```
`train.py` must softmax `mgd_logits.npy` and sample proportionally.

### C.  Compute Whitening Matrix + Mean Direction

```bash
python compute_whitening.py \
  --features openhermes/features.fp16 \
  --output  ckpt/iter0/whitening.npz
```

### D.  Score Cluster Agreement

```bash
python cluster_agreement.py \
  --whitening ckpt/iter0/whitening.npz \
  --cluster-feats-dir embeddings/ \
  --clusters-tsv clusters.tsv \
  --out scores.npy
```

### E.  Update Logits

```python
import numpy as np
logits  = np.load("mgd_logits.npy")
scores  = np.load("scores.npy")
# scale to −1 … 1
scores  = 2 * (scores - scores.min()) / (scores.ptp() + 1e-9) - 1
logits += 0.5 * scores   # temperature 0.5; tune as needed
np.save("mgd_logits.npy", logits)
```

### F.  Repeat
Use the updated `mgd_logits.npy` in the next call to **B**.

---
## Notes

* Every script referenced lives in `dclm_exp/clustering/`.
* Adjust `--nproc_per_node`, `--batch-size`, etc. for other hardware.
* When training with enforced ordering, set `--workers 1` to preserve order.
* For production-scale runs, move large intermediates to S3/GCS and stream.


