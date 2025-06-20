#!/usr/bin/env python
"""
collect_grads.py
----------------
Iterate over a *tokenised* WebDataset (tokens.npy / labels.npy) and save
per-example gradients dL/dW for the model's ``lm_head.weight`` at a fixed 400 M
checkpoint.

Two loading modes
-----------------
* **--uuid**  : give the run UUID produced by Datacomp-LM/Open-LM training;
                the script resolves the JSON in ``exp_data/models`` and loads
                the exact checkpoint the evaluation code would.
* **--ckpt**  : (legacy) path or HF Hub ID pointing to a standard HuggingFace
                directory with ``pytorch_model*.bin`` etc.

Outputs
-------
grads.fp16        mem-map  (N_samples, vocab * hidden)  ~ float16
sample_index.tsv  row_id <TAB> sample_key   (key == tokens.npy basename)
"""

import argparse, json, uuid
from pathlib import Path
import numpy as np
import torch, tqdm, webdataset as wds
from transformers import AutoModelForCausalLM
from lora.lora import LoRAHandler
from lora.logger import Logger

# --- optional: only needed when you use --uuid -----------------------------
try:
    from open_lm.utils import download_cached          # S3/HTTP cache helper
    from open_lm.factory import build_open_lm          # model builder
except ImportError:
    download_cached = build_open_lm = None             # fallback for --ckpt
# ---------------------------------------------------------------------------

# -------------------- configuration -------------------- #
DTYPE_OUT     = np.float16
BATCH_SIZE    = 8                        # fits in 80 GB on an H100
# ------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                                Loaders                                      #
# --------------------------------------------------------------------------- #
def load_hf_model(ckpt_path: str):
    """Standard HuggingFace loader (unchanged)."""
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    return model


def load_openlm_model_from_uuid(run_uuid: str):
    """
    Resolve a Datacomp-LM / Open-LM run UUID to an actual checkpoint and
    instantiate the model exactly the way evaluation does.
    """
    if download_cached is None or build_open_lm is None:
        raise ImportError("open_lm.* unavailable -- install open-lm to use --uuid")

    # 1) locate the metadata JSON emitted by training
    exp_root = Path(__file__).resolve().parent / "exp_data" / "models"
    meta_path = next(exp_root.rglob(f"*{run_uuid}*.json"), None)
    if meta_path is None:
        raise FileNotFoundError(f"could not find exp_data/models/*{run_uuid}*.json")

    meta = json.loads(meta_path.read_text())

    # 2) download / cache the blobs
    ckpt_local   = download_cached(meta["checkpoint_url"])
    params_local = download_cached(meta["params_url"])

    # 3) build the model (same path used by eval scripts)
    model, _, _ = build_open_lm(
        ckpt_path     = ckpt_local,
        params_path   = params_local,
        config_name   = meta["open_lm_config"],
        dtype         = torch.bfloat16,
        device        = "cuda",
        compile_model = False,         # need autograd-friendly version
    )
    return model


def get_model(args):
    """Choose the proper load routine based on the flags."""
    if args.uuid is not None:
        return load_openlm_model_from_uuid(args.uuid)
    if args.ckpt is not None:
        return load_hf_model(args.ckpt)
    raise ValueError("must supply either --uuid or --ckpt")

def prepare_model(model):
    model_parameters = sum(p.numel() for p in model.parameters())
    handler = LoRAHandler(rank=args.lora_rank)
    # init logger with model parameters and number of blocks
    logger = Logger(model.parameters(), 8)
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


# --------------------------------------------------------------------------- #
#                              Data helpers                                   #
# --------------------------------------------------------------------------- #
def collate(batch):
    toks   = [b["tokens.npy"]  for b in batch]
    labels = [b["labels.npy"]  for b in batch]
    L      = max(len(t) for t in toks)
    pad    = lambda x: np.pad(x, (0, L - len(x)), constant_values = 0)
    toks   = torch.tensor([pad(t) for t in toks],   dtype = torch.long)
    labels = torch.tensor([pad(l) for l in labels], dtype = torch.long)
    return toks.cuda(), labels.cuda(), [b["__key__"] for b in batch]


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main(args):
    # ---------- model ----------
    model   = get_model(args)
    head    = dict(model.named_parameters())[TARGET_PARAMS[0]]
    V, D    = head.shape
    vec_dim = V * D

    # ---------- dataset ----------
    ds = (
        wds.WebDataset(f"{args.wds_dir}/*.tar")
        .decode("numpy")
        .to_tuple("tokens.npy", "labels.npy", "__key__")
    )
    loader  = wds.WebLoader(
        ds, batch_size = BATCH_SIZE, num_workers = 64, collate_fn = collate
    )

    with open(f"{args.wds_dir}/manifest.jsonl") as mf:
        N_est = int(mf.read().split(",")[1])

    grads = np.memmap(args.out, mode = "w+", dtype = DTYPE_OUT,
                      shape = (N_est, vec_dim))
    index = open(args.index, "w")
    row   = 0
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100, reduction = "mean")

    for toks, labels, keys in tqdm.tqdm(loader, total = N_est // BATCH_SIZE):
        with torch.enable_grad():
            out  = model(input_ids = toks).logits
            loss = loss_fn(out.view(-1, V), labels.view(-1))
            g    = torch.autograd.grad(loss, head)[0]     # V × D

        grads[row] = g.flatten().half().cpu().numpy()
        index.write(f"{row}\t{keys[0]}\n")                # 1 row per sample
        row += 1

    grads.flush()
    index.close()
    print(f"Saved {row} gradients to {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required = True,
                    help = "directory containing tokenised WebDataset shards")
    ap.add_argument("--uuid",
                    help = "Datacomp-LM run UUID (overrides --ckpt)")
    ap.add_argument("--ckpt",
                    help = "HuggingFace checkpoint path or Hub ID")
    ap.add_argument("--out",   default = "grads.fp16",
                    help = "output memmap filename")
    ap.add_argument("--index", default = "sample_index.tsv",
                    help = "tsv mapping row_id → sample key")
    args = ap.parse_args()

    # sanity: enforce mutually exclusive args
    if (args.uuid is None) == (args.ckpt is None):
        raise SystemExit("exactly one of --uuid or --ckpt is required")

    main(args)
