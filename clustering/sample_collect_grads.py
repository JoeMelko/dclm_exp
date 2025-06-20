#!/usr/bin/env python
"""
collect_grads.py
----------------
Iterate over a *tokenised* WebDataset (tokens.npy / labels.npy) and save
per‑example gradients dL/dW for the model's lm_head.weight at a fixed 400 M
checkpoint.

Outputs
-------
grads.fp16        mem‑map  (N_samples, 1_024 * vocab)  ~ float16
sample_index.tsv  row_id <TAB> sample_key   (key == tokens.npy basename)
"""

import argparse, numpy as np, torch, webdataset as wds, tqdm
from transformers import AutoModelForCausalLM

# -------------------- configuration -------------------- #
TARGET_PARAMS = ["lm_head.weight"]             # <- change to LoRA names if needed
DTYPE_OUT     = np.float16
BATCH_SIZE    = 8                              # fits in 80 GB per H100
# ------------------------------------------------------- #

def load_fixed_model(ckpt_path):
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval().requires_grad_(False)         # freeze everything
    # re‑enable grad only on target params
    for n, p in model.named_parameters():
        if n in TARGET_PARAMS:
            p.requires_grad_(True)
    return model

def collate(batch):
    toks   = [b["tokens.npy"] for b in batch]
    labels = [b["labels.npy"] for b in batch]
    L      = max(len(t) for t in toks)
    pad    = lambda x: np.pad(x, (0, L-len(x)), constant_values=0)
    toks   = torch.tensor([pad(t) for t in toks], dtype=torch.long)
    labels = torch.tensor([pad(l) for l in labels], dtype=torch.long)
    return toks.cuda(), labels.cuda(), [b["__key__"] for b in batch]

def prepare_model(model):
    model_parameters = sum(p.numel() for p in model.parameters())
    handler = LoRAHandler(rank=args.lora_rank)
    logger = Logger(model_parameters)
    handler.add_lora(model, logger, train_loader.batch_size)

    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    for name, param in model.named_parameters():
        if "logix_lora_B" not in name:
            param.requires_grad = False
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()
    # TODO: add the removal of randomness

def main(args):
    # ---------- model & projection vector size ----------
    model   = load_fixed_model(args.ckpt)
    head    = dict(model.named_parameters())[TARGET_PARAMS[0]]
    V, D    = head.shape                     # vocab, 1024
    vec_dim = V * D
    # ---------- dataset ----------
    ds = (wds.WebDataset(f"{args.wds_dir}/*.tar")
          .decode("numpy")
          .to_tuple("tokens.npy", "labels.npy", "__key__"))
    loader = wds.WebLoader(ds, batch_size=BATCH_SIZE, num_workers=4,
                           collate_fn=collate)

    N_est = int(open(f"{args.wds_dir}/manifest.jsonl").read().split(",")[1])
    grads = np.memmap(args.out, mode="w+", dtype=DTYPE_OUT,
                      shape=(N_est, vec_dim))
    index = open(args.index, "w")
    row   = 0

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    for toks, labels, keys in tqdm.tqdm(loader, total=N_est//BATCH_SIZE):
        with torch.enable_grad():
            out = model(input_ids=toks).logits
            loss = loss_fn(out.view(-1, V), labels.view(-1))
            g    = torch.autograd.grad(loss, head)[0]   # shape V×D
        g   = g.flatten().half().cpu().numpy()
        grads[row] = g
        index.write(f"{row}\t{keys[0]}\n")   # one key per row (no micro-batching)
        row += 1

    grads.flush(); index.close()
    print(f"Saved {row} gradients to {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True, help="tokenised WebDataset dir")
    ap.add_argument("--ckpt",    required=True, help="400m checkpoint path")
    ap.add_argument("--out",     default="grads.fp16")
    ap.add_argument("--index",   default="sample_index.tsv")
    main(ap.parse_args())
