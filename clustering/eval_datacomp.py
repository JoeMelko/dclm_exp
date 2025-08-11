#!/usr/bin/env python
"""
eval_datacomp.py
----------------
Evaluate an Open-LM checkpoint (by Datacomp-LM run UUID) on a held-out portion
of the Datacomp tokenised WebDataset.

Format expectations (matching collect_features_dc.py):
- Each sample contains either `tokens.npy` (legacy) or a JSON/JSON.GZ with a
  top-level list or dict field named `tokens` or `input_ids`.
- We build labels identical to input ids, with padding positions ignored (-100).

This script:
- Loads the model via the UUID using `load_openlm_model_from_uuid`.
- Loads local WebDataset shards from `--wds-dir` using an optional glob `--pattern`.
- Pads and batches sequences, computes token-weighted cross-entropy loss.
- Prints running average loss every `--log-interval` batches and final perplexity.

Example
-------
python -m dclm_exp.clustering.eval_datacomp \
  --uuid <RUN_UUID> \
  --wds-dir /path/to/heldout_wds \
  --pattern "*.tar" \
  --batch-size 16 \
  --log-interval 500 \
  --total-samples 100000
"""
# --------------------------------------------------------------------------- #
import argparse, io, json, gzip, math
from pathlib import Path

import numpy as np
import torch, tqdm
import webdataset as wds
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

# Enable TF-32 for safe speedups
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#                         Model loader (reuse from eval_openhermes)           #
# --------------------------------------------------------------------------- #
try:
    from .eval_openhermes import load_openlm_model_from_uuid  # type: ignore
except ImportError:  # nocov
    import sys as _sys
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore


# --------------------------------------------------------------------------- #
#                              Tokenizer                                      #
# --------------------------------------------------------------------------- #
# Use GPT-NeoX-20B tokenizer to obtain EOS token id for padding
_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


# --------------------------------------------------------------------------- #
#                         Data loading helpers                                #
# --------------------------------------------------------------------------- #
def _load_tokens_from_sample(sample: dict):
    """Return the token list from a WebDataset sample as a 1-D int32 array.

    Supports either legacy ``tokens.npy`` payloads **or** ``*.json`` / ``*.json.gz``
    files containing a top-level list (tokens) or dict with ``tokens`` / ``input_ids``
    field.
    """
    if "tokens.npy" in sample:  # fast path
        return np.load(io.BytesIO(sample["tokens.npy"]))

    # Otherwise expect exactly one JSON payload key
    json_keys = [k for k in sample if k.endswith("json") or k.endswith("json.gz")]
    if not json_keys:
        return None  # no usable payload
    key = json_keys[0]
    raw = sample[key]
    if key.endswith(".gz"):
        raw = gzip.decompress(raw)
    obj = json.loads(raw)

    if isinstance(obj, list):
        tokens_list = obj
    elif isinstance(obj, dict):
        if "tokens" in obj:
            tokens_list = obj["tokens"]
        elif "input_ids" in obj:
            tokens_list = obj["input_ids"]
        else:
            raise KeyError("JSON object missing 'tokens'/'input_ids' field")
    else:
        return None  # unsupported payload type

    return np.asarray(tokens_list, dtype=np.int32) if tokens_list else None


def _collate(batch):
    """Pad variable-length token sequences and build corresponding labels."""
    toks = []
    for b in batch:
        arr = _load_tokens_from_sample(b)
        if arr is not None and len(arr) > 0:
            toks.append(arr)
    if not toks:
        raise ValueError("Empty batch after filtering invalid samples")

    L = max(len(t) for t in toks)

    pad_id = _tokenizer.eos_token_id
    pad_toks   = lambda x: np.pad(x, (0, L - len(x)), constant_values=pad_id)
    pad_labels = lambda x: np.pad(x, (0, L - len(x)), constant_values=-100)

    toks_padded = torch.tensor([pad_toks(t)   for t in toks],   dtype=torch.long)
    labels      = torch.tensor([pad_labels(t) for t in toks], dtype=torch.long)
    return toks_padded, labels


# --------------------------------------------------------------------------- #
#                               Evaluation                                    #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             log_interval: int = 500,
             total_samples: int | None = None) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct = 0
    total_batches = math.ceil(total_samples / loader.batch_size) if total_samples else None
    for toks, labels in tqdm.tqdm(loader, desc="Evaluating", total=total_batches):
        toks, labels = toks.to(device), labels.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels)
        n_tokens       = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        if log_interval > 0 and ct % log_interval == 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset ---------------------------------------------------------- #
    wds_dir = Path(args.wds_dir)
    if not wds_dir.is_absolute() and not wds_dir.exists():
        alt = Path(__file__).resolve().parent / wds_dir
        wds_dir = alt if alt.exists() else wds_dir
    if not wds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {wds_dir}")

    shards = sorted(wds_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching pattern '{args.pattern}' found in {wds_dir}")

    ds = wds.WebDataset([str(p) for p in shards], handler=wds.handlers.ignore_and_continue)
    loader = wds.WebLoader(ds,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           collate_fn=_collate)

    # ----- evaluation ------------------------------------------------------- #
    loss = evaluate(model,
                    loader,
                    device,
                    log_interval=args.log_interval,
                    total_samples=args.total_samples)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--wds-dir", required=True,
                        help="Directory containing held-out Datacomp WebDataset shards")
    parser.add_argument("--pattern", default="*.tar",
                        help="Glob pattern to select shards within wds-dir (default: *.tar)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader workers (default: 8)")
    parser.add_argument("--log-interval", type=int, default=500,
                        help="Print running average loss every N batches (default: 500; 0 disables)")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Total number of samples to enable tqdm progress bar. If omitted, tqdm shows speed only.")
    main(parser.parse_args()) 