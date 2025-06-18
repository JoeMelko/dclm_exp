# Cluster‑Aware Data Curation Pipeline  
*(sub‑directory `cluster/`)*

This mini‑pipeline turns one **Datacomp‑LM local shard** (~ 40 B tokens / 40 M
docs) into **per‑cluster tokenised datasets** that you can mix and weight freely
for DCLM training.

## 0 .  Requirements

* **Hardware**  • 1 node, 8 × H100‑80 GB (other multi‑GPU cards also work).  
* **Python**    • 3.10 + the packages below:

pip install "transformers>=4.51" sentence-transformers>=2.7.0 \
            accelerate datasets faiss-gpu numpy tqdm zstandard \
            webdataset blake3 torch

# Download the shard
aws s3 cp --recursive \
  s3://commoncrawl/contrib/datacomp/DCLM-baseline/global-shard_01_of_10/local-shard_1_of_10/ \
  data/gs01_ls1

# embed the documents
python embed.py \
       --shard-dir data/gs01_ls1

# run kmeans
python kmeans.py \
       --embeddings embeddings.fp16 \
       --n-clusters 10000 \
       --sample 3000000 \
       --out clusters.tsv
**→ clusters.tsv  (row_id <TAB> cluster_id)**

# Extract raw docs for cluster k

awk '$2==k {print $1}' clusters.tsv > ids_k.txt

python extract_cluster.py \
       --row-ids ids_k.txt \
       --shard-root data/gs01_ls1 \
       --out-jsonl cluster_k.jsonl

# Tokenise & shuffle cluster docs
cargo run --release -- \
   --input cluster_k.jsonl \
   --output tok/cluster_k \
   --tokenizer EleutherAI/gpt-neox-20b \
   --seqlen 2049 --wds-chunk-size 8192

# (Optional) balance cluster sizes
Duplicate or down‑sample shards so that each cluster supplies exactly Nₖ examples:

python balanced_resample.py \
       --indir  tok/cluster_k \
       --outdir tok/cluster_k_bal \
       --n      $TARGET_SAMPLES_K


