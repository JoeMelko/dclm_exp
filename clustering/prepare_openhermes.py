#!/usr/bin/env python
"""
prepare_openhermes.py
---------------------
Download *OpenHermes‑2.5*, format it so that loss / gradients apply **only to
assistant answers**, and emit tokenised WebDataset shards ready for DCLM
(400 M model) training or gradient extraction.

Usage
-----
python prepare_openhermes.py \
       --out-dir tok/openhermes25 \
       --seqlen 2049 \
       --shard-size 8192           # samples per .tar shard
"""
import argparse, os, numpy as np, webdataset as wds, tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def iter_docs():
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    for rec in ds:
        conv = rec["conversations"]
        prompt_parts, answer = [], None
        for turn in conv:
            role = turn["from"]
            if role == "gpt":
                answer = turn["value"]
                break
            prompt_parts.append(f"<|im_start|>{role}\n{turn['value']}\n<|im_end|>")
        prompt = "\n".join(prompt_parts) + "\n<|im_start|>assistant\n"
        if answer is None:
            continue                 # skip malformed
        yield prompt, answer

def shard_writer(out_dir, shard_size):
    return wds.ShardWriter(f"{out_dir}/openhermes-%06d.tar", maxcount=shard_size)

def main(args):
    tok = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b",
        padding_side="right",
        truncation_side="right"
    )
    tok.pad_token = tok.eos_token
    os.makedirs(args.out_dir, exist_ok=True)
    sink   = shard_writer(args.out_dir, args.shard_size)
    seq    = 0

    for prompt, answer in tqdm.tqdm(iter_docs(), desc="tokenise"):
        prompt_ids  = tok(prompt)["input_ids"]
        answer_ids  = tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

        # 1. Concatenate prompt + answer
        input_ids   = prompt_ids + answer_ids

        # 2. Copy to labels
        labels      = input_ids.copy()

        # 3. Mask prompt positions
        labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

        # 4. Pad/truncate to fixed length for both tokens and labels
        if len(input_ids) < args.seqlen:
            tokens_list = input_ids + [tok.pad_token_id] * (args.seqlen - len(input_ids))
        else:
            tokens_list = input_ids[:args.seqlen]

        # Manually pad / truncate labels to match the fixed sequence length.
        if len(labels) < args.seqlen:
            labels_list = labels + [-100] * (args.seqlen - len(labels))
        else:
            labels_list = labels[:args.seqlen]

        # Convert to NumPy arrays for efficient downstream processing.
        tokens = np.asarray(tokens_list, dtype=np.int32)
        labels = np.asarray(labels_list, dtype=np.int32)

        # sanity check
        labels[tokens == tok.pad_token_id] = -100

        # 5. Write to shard 
        sink.write({
            "__key__"     : f"{seq:012d}",
            "tokens.npy"  : tokens,
            "labels.npy"  : labels
        })
        seq += 1
    sink.close()
    print(f"wrote {seq:,} samples to {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seqlen", type=int, default=2049)
    ap.add_argument("--shard-size", type=int, default=8192)
    main(ap.parse_args())
