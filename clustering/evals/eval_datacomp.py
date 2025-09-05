#!/usr/bin/env python
"""
eval_datacomp.py
----------------
Evaluate an Open-LM checkpoint (by Datacomp-LM run UUID) on a held-out portion
of the Datacomp tokenised dataset stored as WebDataset tar shards.

Shard expectations (aligned with collect_features_dc.py payload parsing):
- Each sample contains either:
  - legacy `tokens.npy`, or
  - JSON/JSON.GZ payload with fields `tokens` or `input_ids`.

This script:
- Loads the model via UUID using `load_openlm_model_from_uuid`.
- Globs `.tar` shards under `--wds-dir` (configurable via `--pattern`).
- Streams samples via WebDataset and extracts token arrays per example.
- Builds batches, padding to max length in-batch and creating labels where pad
  positions are `-100`.
- Computes token-weighted cross-entropy loss with periodic logging and final perplexity.

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
from typing import List, Optional

import numpy as np
import torch, tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.cuda.amp import autocast
import webdataset as wds

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
_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
SEP_TOKEN_ID = 0  # document separator id; attention is blocked across this token


# --------------------------------------------------------------------------- #
#                        WebDataset sample parsing                             #
# --------------------------------------------------------------------------- #
def _load_tokens_from_sample(sample: dict) -> Optional[np.ndarray]:
    """Extract the token list from a WebDataset sample and return as int32 array.
    Accepts either legacy `tokens.npy` or generic `*.json[.gz]` payloads with
    fields `tokens` or `input_ids`.
    """
    if "tokens.npy" in sample:
        return np.load(io.BytesIO(sample["tokens.npy"]))

    json_keys = [k for k in sample if k.endswith("json") or k.endswith("json.gz")]
    if not json_keys:
        return None
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
            return None
    else:
        return None

    return np.asarray(tokens_list, dtype=np.int32) if tokens_list else None


# --------------------------------------------------------------------------- #
#                         Collate and evaluation                              #
# --------------------------------------------------------------------------- #
def collate(batch: List[dict]):
    toks: List[np.ndarray] = []
    for b in batch:
        arr = _load_tokens_from_sample(b)
        if arr is not None and len(arr) > 0:
            toks.append(arr)
    if not toks:
        # Fallback to a dummy example to avoid DataLoader errors; caller will ignore tokens
        raise ValueError("Empty batch after filtering invalid samples")

    L = max(len(t) for t in toks)
    # Pad input_ids with the separator id (0) so doc mask also blocks pads
    pad_id_inputs = SEP_TOKEN_ID

    def pad_toks(x: np.ndarray) -> np.ndarray:
        return np.pad(x, (0, L - len(x)), constant_values=pad_id_inputs)

    def pad_labels(x: np.ndarray) -> np.ndarray:
        # Mask only padded positions with -100; keep true tokens intact (including EOS if present)
        return np.pad(x, (0, L - len(x)), constant_values=-100)

    toks_padded = torch.tensor([pad_toks(t) for t in toks], dtype=torch.long)
    labels = torch.tensor([pad_labels(t) for t in toks], dtype=torch.long)
    # Do not compute loss on artificial separator tokens
    labels[toks_padded == SEP_TOKEN_ID] = -100
    return toks_padded, labels


@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             log_interval: int = 500,
             total_samples: int | None = None,
             n_batches: int | None = 5000) -> float:
    """Evaluate and return token-weighted average cross-entropy over up to n_batches."""
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    ct, seen_examples = 0, 0
    if n_batches is not None:
        total_batches = n_batches
    else:
        total_batches = math.ceil(total_samples / loader.batch_size) if total_samples else None
    for toks, labels in tqdm.tqdm(loader, desc="Evaluating", total=total_batches):
        toks, labels = toks.to(device), labels.to(device)
        # Avoid passing attention_mask when document-aware masking is enabled,
        # to prevent shape mismatches in the attention implementation.
        use_doc_mask = getattr(getattr(model, "model", None), "params", None) is not None and getattr(model.model.params, "doc_causal_mask", False)
        attn_mask = None if use_doc_mask else (labels != -100)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels, attention_mask=attn_mask)
        n_tokens = (labels != -100).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        ct += 1
        seen_examples += toks.size(0)
        if log_interval > 0 and ct % log_interval == 0 and tmp_tokens > 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
        if n_batches is not None and ct >= n_batches:
            break
        if n_batches is None and total_samples is not None and seen_examples >= total_samples:
            break
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_openlm_model_from_uuid(args.uuid)
    # Enable document-aware causal masking in the underlying Transformer.
    # This prevents attention across segments separated by token id 0.
    try:
        model.model.params.doc_causal_mask = True
    except Exception:
        pass
    model.to(device, dtype=torch.bfloat16).eval()
    breakpoint()

    # ----- shards ----------------------------------------------------------- #
    wds_dir = Path(args.wds_dir)
    if not wds_dir.is_absolute() and not wds_dir.exists():
        alt = Path(__file__).resolve().parent / wds_dir
        wds_dir = alt if alt.exists() else wds_dir
    if not wds_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {wds_dir}")

    files = sorted(wds_dir.rglob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern '{args.pattern}' found under {wds_dir}")

    ds = wds.WebDataset([str(p) for p in files], handler=wds.handlers.ignore_and_continue)
    loader = wds.WebLoader(ds,
                           batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           collate_fn=collate)

    # ----- evaluation ------------------------------------------------------- #
    loss = evaluate(model,
                    loader,
                    device,
                    log_interval=args.log_interval,
                    total_samples=args.total_samples,
                    n_batches=args.n_batches)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")
    return loss


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--wds-dir", required=True,
                        help="Directory containing held-out WebDataset shards (.tar)")
    parser.add_argument("--pattern", default="*.tar",
                        help="Glob pattern relative to wds-dir (default: *.tar)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader workers (default: 8)")
    parser.add_argument("--log-interval", type=int, default=500,
                        help="Print running average loss every N batches (default: 500; 0 disables)")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Cap total number of samples to iterate (default: None)")
    parser.add_argument("--n-batches", type=int, default=5000,
                        help="Return mean loss after N batches (default: 5000). If set, overrides total-samples for stopping criterion.")
    main(parser.parse_args()) 