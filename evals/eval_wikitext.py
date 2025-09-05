#!/usr/bin/env python
"""
eval_wikitext.py
----------------
Evaluate an Open-LM checkpoint (by Datacomp-LM run UUID) on the
EleutherAI/wikitext_document_level dataset (wikitext-103-v1, train split).

Pretraining evaluation protocol:
- Do not truncate individual documents.
- Tokenize every document end-to-end without special tokens.
- Concatenate all tokenized documents into a single token buffer,
  inserting an end-of-text separator token id 0 between documents.
- After full concatenation, chunk the buffer into fixed-length sequences
  (default: 2048 tokens) and evaluate cross-entropy/perplexity.

Dataset reference: https://huggingface.co/datasets/EleutherAI/wikitext_document_level

Example
-------
python -m dclm_exp.clustering.eval_wikitext --uuid <RUN_UUID> \
  --batch-size 16 --block-size 2048 --log-interval 200

Optional arguments
------------------
--batch-size     Evaluation batch size (default: 16)
--block-size     Sequence length for evaluation chunks (default: 2048)
--max-chunks     Limit the number of chunks used for evaluation (default: None)
--log-interval   Print running token-weighted average loss every N batches (default: 500)
"""
# --------------------------------------------------------------------------- #
import argparse, math
from typing import List, Optional

import numpy as np
import torch, tqdm
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

# Datasets
try:
    from datasets import load_dataset  # type: ignore
except Exception as _e:  # nocov
    load_dataset = None  # type: ignore

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
    from pathlib import Path as _Path
    _this_dir = _Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore


# --------------------------------------------------------------------------- #
#                                   Setup                                     #
# --------------------------------------------------------------------------- #
# Use the same tokenizer as other evaluators in this repo
TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
# Explicit document separator token id per request
DOC_SEP_TOKEN_ID = 0

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# --------------------------------------------------------------------------- #
#                         Tokenization and chunking                           #
# --------------------------------------------------------------------------- #
def build_concatenated_token_buffer(texts: List[str], sep_token_id: int) -> np.ndarray:
    """Tokenize texts without special tokens and concatenate with sep_token_id between docs.

    Returns a 1-D numpy array of dtype int64 containing the full token stream.
    """
    token_buffer: List[int] = []
    for idx, text in enumerate(tqdm.tqdm(texts, desc="Tokenizing docs")):
        ids: List[int] = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        if ids:
            token_buffer.extend(ids)
        # Append separator between documents (including after empties)
        token_buffer.append(sep_token_id)
    return np.asarray(token_buffer, dtype=np.int64)


def chunk_token_buffer(token_buffer: np.ndarray, block_size: int) -> np.ndarray:
    """Trim the token buffer to a multiple of block_size and reshape into 2-D chunks.

    Returns an array with shape (num_chunks, block_size), dtype int64.
    """
    total_full = (len(token_buffer) // block_size) * block_size
    if total_full == 0:
        raise ValueError("Token buffer is smaller than one block; nothing to evaluate.")
    trimmed = token_buffer[:total_full]
    return trimmed.reshape(-1, block_size)


# --------------------------------------------------------------------------- #
#                                   Eval                                       #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    batches: torch.utils.data.DataLoader,
    device: torch.device,
    log_interval: int = 500,
) -> float:
    total_loss, total_tokens = 0.0, 0
    tmp_loss, tmp_tokens = 0.0, 0
    for step, toks in enumerate(tqdm.tqdm(batches, desc="Evaluating"), start=1):
        labels = toks  # no padding; labels == input ids
        toks, labels = toks.to(device), labels.to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(input_ids=toks, labels=labels)
        # out.loss is mean over non-ignored labels internally
        n_tokens = labels.numel()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
        tmp_loss += out.loss.item() * n_tokens
        tmp_tokens += n_tokens
        if log_interval > 0 and step % log_interval == 0:
            print(tmp_loss / tmp_tokens)
            tmp_loss, tmp_tokens = 0.0, 0
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                                    Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    if load_dataset is None:
        raise RuntimeError("The 'datasets' library is required. Install via `pip install datasets`.")

    # ----- model ------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset ---------------------------------------------------------- #
    ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-103-v1")
    train = ds["train"]
    # Use 'text' if present, otherwise fall back to 'page' as provided by the dataset
    col = "text" if "text" in train.column_names else "page"
    texts: List[str] = train[col]  # type: ignore[index]

    # ----- tokenize and chunk ---------------------------------------------- #
    token_buffer = build_concatenated_token_buffer(texts, DOC_SEP_TOKEN_ID)
    chunks_np = chunk_token_buffer(token_buffer, args.block_size)

    if args.max_chunks is not None:
        chunks_np = chunks_np[: args.max_chunks]

    # Convert to a tensor dataset for efficient batching
    chunks = torch.from_numpy(chunks_np)  # shape: (num_chunks, block_size)
    loader = torch.utils.data.DataLoader(
        chunks,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ----- evaluation ------------------------------------------------------- #
    loss = evaluate(model, loader, device, log_interval=args.log_interval)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uuid",
        required=True,
        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--block-size", type=int, default=2048, help="Sequence length (default: 2048)")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks to evaluate (default: None)")
    parser.add_argument("--log-interval", type=int, default=500, help="Log average loss every N batches (default: 500)")
    main(parser.parse_args()) 