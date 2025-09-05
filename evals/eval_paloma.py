#!/usr/bin/env python
"""
eval_paloma.py
---------------
Compute the token-level cross-entropy loss (and perplexity) of an Open-LM
checkpoint – identified by its Datacomp-LM run UUID – on the pre-tokenised
Paloma validation set.

The four WebDataset shards (00000001.tar … 00000004.tar) are automatically
retrieved from the HuggingFace dataset "mlfoundations/paloma_validation" if
not already present locally.

Example
-------
python eval_paloma.py --uuid <RUN_UUID>

Optional arguments
------------------
--wds-dir   Directory to cache the downloaded shards (default: ./paloma_validation)
--batch-size  Evaluation batch size (default: 16)
"""
# --------------------------------------------------------------------------- #
import argparse, io, json, gzip, math
from pathlib import Path

import numpy as np
import torch, tqdm
import webdataset as wds
from transformers import AutoTokenizer
from torch.cuda.amp import autocast

# --------------------------------------------------------------------------- #
#                     Optional dependency: huggingface_hub                    #
# --------------------------------------------------------------------------- #
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

# Enable TF-32 for safe speedups
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# --------------------------------------------------------------------------- #
#                         Model loader (reuse from eval_openhermes)           #
# --------------------------------------------------------------------------- #
# When executed within a package (e.g. `python -m dclm_exp.clustering.eval_paloma`) the
# relative import below works.  When run as a standalone script, the package
# context is absent, so we fall back to adding this file's directory to
# `sys.path` and performing an absolute import instead.
try:
    from .eval_openhermes import load_openlm_model_from_uuid  # type: ignore
except ImportError:  # nocov
    import sys as _sys
    _this_dir = Path(__file__).resolve().parent
    if str(_this_dir) not in _sys.path:
        _sys.path.insert(0, str(_this_dir))
    from eval_openhermes import load_openlm_model_from_uuid  # type: ignore


# --------------------------------------------------------------------------- #
#                         Data loading helpers                                #
# --------------------------------------------------------------------------- #
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# --------------------------------------------------------------------------- #
#                  Utility: load tokens from WebDataset sample               #
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


def collate(batch):
    """Pad variable-length token sequences and build corresponding labels."""
    toks = []
    for b in batch:
        arr = _load_tokens_from_sample(b)
        if arr is not None and len(arr) > 0:
            toks.append(arr)
    if not toks:
        raise ValueError("Empty batch after filtering invalid samples")

    L = max(len(t) for t in toks)

    pad_id = tokenizer.eos_token_id
    pad_toks   = lambda x: np.pad(x, (0, L - len(x)), constant_values=pad_id)
    pad_labels = lambda x: np.pad(x, (0, L - len(x)), constant_values=-100)

    toks_padded = torch.tensor([pad_toks(t)   for t in toks],   dtype=torch.long)
    labels      = torch.tensor([pad_labels(t) for t in toks], dtype=torch.long)
    return toks_padded, labels


# --------------------------------------------------------------------------- #
#                               Evaluation                                    #
# --------------------------------------------------------------------------- #
# --------------------- evaluation loop ------------------------------------ #

@torch.inference_mode()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
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
        if ct % 500 == 0:
            print(tmp_loss / tmp_tokens)
    return total_loss / total_tokens


# --------------------------------------------------------------------------- #
#                           Shard acquisition                                 #
# --------------------------------------------------------------------------- #
HF_REPO = "mlfoundations/paloma_validation"
SHARD_FILENAMES = [f"{i:08d}.tar" for i in range(1, 5)]  # 00000001.tar … 00000004.tar

def ensure_shards(wds_dir: Path):
    """Download the required shards from HuggingFace if missing locally."""
    wds_dir.mkdir(parents=True, exist_ok=True)
    missing = [fn for fn in SHARD_FILENAMES if not (wds_dir / fn).exists()]
    if not missing:
        return

    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub is not installed – run `pip install huggingface_hub` "
            "or download the shards manually."
        )

    print(f"Downloading {len(missing)} shard(s) to {wds_dir}…")
    for fn in missing:
        path = hf_hub_download(repo_id=HF_REPO,
                               filename=fn,
                               repo_type="dataset",
                               local_dir=wds_dir,
                               local_dir_use_symlinks=False)
        print(f"  ✓ {fn} → {path}")


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def main(args):
    # ----- model ---------------------------------------------------------- #
    device = torch.device("cuda")

    model  = load_openlm_model_from_uuid(args.uuid)
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- dataset (download if needed) ----------------------------------- #
    wds_dir = Path(args.wds_dir).expanduser().resolve()
    ensure_shards(wds_dir)

    shards = [str(wds_dir / fn) for fn in SHARD_FILENAMES]
    ds = wds.WebDataset(shards, handler=wds.handlers.ignore_and_continue)
    loader = wds.WebLoader(ds,
                           batch_size=args.batch_size,
                           num_workers=8,
                           collate_fn=collate)

    # ----- evaluation ----------------------------------------------------- #
    loss = evaluate(model, loader, device, total_samples=args.total_samples)
    print(f"Cross-entropy loss: {loss:.6f}\nPerplexity       : {math.exp(loss):.6f}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--wds-dir", default="./paloma_validation",
                        help="Cache directory for Paloma validation shards (default: ./paloma_validation)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Evaluation batch size (default: 16)")
    parser.add_argument("--total-samples", type=int, default=None,
                        help="Total number of samples in the dataset – enables tqdm progress bar. If omitted, tqdm shows speed only.")
    main(parser.parse_args()) 