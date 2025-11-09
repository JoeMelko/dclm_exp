#!/usr/bin/env python
"""
eval_grad_svd.py
----------------
Compute average gradients for simulated batches (accumulated over microbatches)
and record SVD-based stats per layer on the gradient matrices of attention/FFN
weights.

Shard expectations (aligned with collect_features_dc.py payload parsing):
- Each sample contains either:
  - legacy `tokens.npy`, or
  - JSON/JSON.GZ payload with fields `tokens` or `input_ids`.

This script:
- Loads the model via UUID using `load_openlm_model_from_uuid`.
- Globs `.tar` shards under `--wds-dir` (configurable via `--pattern`).
- Streams samples via WebDataset and extracts token arrays per example.
- Builds batches via a custom collate.
- Accumulates gradients over microbatches of size `--mini-bz` to simulate a
  batch of `--bz` examples (simple average by example count).
- For the resulting average gradient, computes stable rank and max singular
  value of these matrices per layer:
    - `layers[i].attention.in_proj.weight`
    - `layers[i].attention.out_proj.weight`
    - `layers[i].feed_forward.w12.weight`
    - `layers[i].feed_forward.w3.weight`
- Appends results as JSON Lines (NDJSON), one JSON object per line.

Example
-------
python -m dclm_exp.evals.eval_grad_svd \
  --uuid <RUN_UUID> \
  --wds-dir /path/to/heldout_wds \
  --pattern "*.tar" \
  --bz 512 \
  --mini-bz 32 \
  --n-batches 100 \
  --log-json /path/to/grad_svd_logs.jsonl
"""
# --------------------------------------------------------------------------- #
import argparse, io, json, gzip, math
from pathlib import Path
from typing import List, Optional
import random

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


def _append_jsonl(log_path: Path, record: dict):
    """Append a JSON object to a file as a single line (JSONL/NDJSON)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _consume_examples(data_iter,
                      num_examples_to_skip: int) -> int:
    """Advance the iterator by approximately num_examples_to_skip examples.

    Returns the number of examples actually skipped (can be >= requested if the
    last microbatch overshoots). Raises StopIteration if the underlying
    iterator is exhausted.
    """
    skipped = 0
    while skipped < num_examples_to_skip:
        toks, _ = next(data_iter)  # may raise StopIteration
        # Only inspect shape on CPU; do not move to device.
        skipped += int(getattr(toks, "size")(0))
    return skipped


def _compute_grad_svd_for_simulated_batch(model: torch.nn.Module,
                                           data_iter,
                                           device: torch.device,
                                           target_bz: int,
                                           log_layers: Optional[int] = None) -> dict:
    """Accumulate gradients over microbatches until at least target_bz examples.

    Performs simple example-count averaging: each microbatch loss contributes
    with weight (microbatch_examples / target_bz).
    Returns a dict of SVD stats for the gradient matrices.
    """
    model.zero_grad(set_to_none=True)
    model.eval()  # keep deterministic behavior; grads are still enabled

    accumulated_examples = 0
    steps = 0
    while accumulated_examples < target_bz:
        toks, labels = next(data_iter)  # may raise StopIteration upstream
        toks, labels = toks.to(device), labels.to(device)
        use_doc_mask = getattr(getattr(model, "model", None), "params", None) is not None and getattr(model.model.params, "doc_causal_mask", False)
        attn_mask = None if use_doc_mask else (labels != -100)

        # Forward + backward; rely on model's built-in mean loss, scale by examples/target_bz
        out = model(input_ids=toks, labels=labels, attention_mask=attn_mask)
        micro_examples = toks.size(0)
        scale = float(micro_examples) / float(target_bz)
        (out.loss * scale).backward()
        accumulated_examples += micro_examples
        steps += 1

    # Gather gradients and compute SVD-based stats
    try:
        layers = model.model.layers  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError("Model structure does not have model.layers as expected") from e
    num_layers = len(layers)
    if log_layers is not None:
        num_layers = min(num_layers, int(log_layers))

    def sv_max_and_stable_rank(t: torch.Tensor) -> tuple[float, float]:
        if t is None:
            return float("nan"), float("nan")
        # Move to GPU float32 for SVD
        g = t.detach()
        if g.ndim != 2:
            g = g.view(g.shape[0], -1)
        g = g.to(device=device, dtype=torch.float32, non_blocking=True)
        # Sanitize non-finite values to avoid NaNs in SVD
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        # Early exit for zero gradients
        fro_sq_t = g.pow(2).sum()
        if float(fro_sq_t.item()) == 0.0:
            return 0.0, 0.0
        # Try GPU SVD; if it fails, fall back to CPU SVD
        try:
            svals = torch.linalg.svdvals(g)
        except RuntimeError:
            try:
                print("Using CPU SVD")
                svals = torch.linalg.svdvals(g.detach().cpu())
            except Exception:
                return float("nan"), float("nan")
        if svals.numel() == 0:
            return float("nan"), float("nan")
        sv_max_t = svals.max()
        eps_t = torch.tensor(1e-12, device=sv_max_t.device, dtype=sv_max_t.dtype)
        stable_rank_t = fro_sq_t / (sv_max_t * sv_max_t + eps_t)
        return float(sv_max_t.item()), float(stable_rank_t.item())

    # Compute global gradient L2 norm across all parameters (after accumulation)
    total_sq = torch.tensor(0.0, device=device, dtype=torch.float32)
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        g = g.to(device=device, dtype=torch.float32, non_blocking=True)
        g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        total_sq = total_sq + g.pow(2).sum()
    grad_l2_norm = float(total_sq.sqrt().item())

    per_layer = []
    for i in range(num_layers):
        layer = layers[i]
        attn_in = getattr(getattr(layer, "attention"), "in_proj").weight.grad if getattr(getattr(layer, "attention"), "in_proj", None) is not None else None
        attn_out = getattr(getattr(layer, "attention"), "out_proj").weight.grad if getattr(getattr(layer, "attention"), "out_proj", None) is not None else None
        ffn_in = getattr(getattr(layer, "feed_forward"), "w12").weight.grad if getattr(getattr(layer, "feed_forward"), "w12", None) is not None else None
        ffn_out = getattr(getattr(layer, "feed_forward"), "w3").weight.grad if getattr(getattr(layer, "feed_forward"), "w3", None) is not None else None

        attn_in_svmax, attn_in_sr = sv_max_and_stable_rank(attn_in)
        attn_out_svmax, attn_out_sr = sv_max_and_stable_rank(attn_out)
        ffn_in_svmax, ffn_in_sr = sv_max_and_stable_rank(ffn_in)
        ffn_out_svmax, ffn_out_sr = sv_max_and_stable_rank(ffn_out)

        per_layer.append({
            "layer": i,
            "attn_in": {"sv_max": attn_in_svmax, "stable_rank": attn_in_sr},
            "attn_out": {"sv_max": attn_out_svmax, "stable_rank": attn_out_sr},
            "ffn_in": {"sv_max": ffn_in_svmax, "stable_rank": ffn_in_sr},
            "ffn_out": {"sv_max": ffn_out_svmax, "stable_rank": ffn_out_sr},
        })

    # Clear grads for next batch
    model.zero_grad(set_to_none=True)

    return {
        "simulated_bz": int(target_bz),
        "num_micro_steps": int(steps),
        "num_layers": int(num_layers),
        "grad_l2_norm": grad_l2_norm,
        "per_layer": per_layer,
    }


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
    # Optional full-dataset sample-level shuffle (streaming with bounded buffer)
    if getattr(args, "shuffle", False):
        buffer_size = int(getattr(args, "shuffle_buffer", 10000) or 10000)
        rng = random.Random(int(getattr(args, "shuffle_seed", 1337) or 1337))
        try:
            # Newer WebDataset (may accept buffer_size and rng)
            ds = ds.shuffle(buffer_size=buffer_size, initial=buffer_size, rng=rng)
        except TypeError:
            try:
                # Alternate kw name used in some versions
                ds = ds.shuffle(bufsize=buffer_size, initial=buffer_size, rng=rng)
            except TypeError:
                try:
                    # Fallback: no rng kw supported, keep deterministic default elsewhere if needed
                    ds = ds.shuffle(buffer_size, buffer_size)
                except TypeError:
                    # Last resort positional with initial as kw
                    ds = ds.shuffle(buffer_size, initial=buffer_size)
    loader = wds.WebLoader(ds,
                           batch_size=args.mini_bz,
                           num_workers=args.num_workers,
                           collate_fn=collate)

    data_iter = iter(loader)

    # ----- gradient SVD accumulation --------------------------------------- #
    total_batches = args.n_batches
    processed = 0
    batch_index = 0
    for _ in tqdm.tqdm(range(total_batches), desc="Simulated batches"):
        try:
            record = _compute_grad_svd_for_simulated_batch(
                model=model,
                data_iter=data_iter,
                device=device,
                target_bz=args.bz,
                log_layers=args.num_layers,
            )
        except StopIteration:
            print("Data exhausted before reaching requested number of batches.")
            break
        record.update({
            "batch_index": int(batch_index),
        })
        _append_jsonl(Path(args.log_json), record)
        processed += 1

        # Skip subsequent batches so that we only process every-kth batch
        k = max(1, int(getattr(args, "every_k", 1) or 1))
        if k > 1:
            to_skip_examples = (k - 1) * args.bz
            try:
                _ = _consume_examples(data_iter, to_skip_examples)
            except StopIteration:
                print("Data exhausted while skipping between batches.")
                break

        batch_index += 1
    return 0.0


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True,
                        help="Datacomp-LM / Open-LM run UUID identifying the checkpoint to evaluate")
    parser.add_argument("--wds-dir", required=True,
                        help="Directory containing held-out WebDataset shards (.tar)")
    parser.add_argument("--pattern", default="*.tar",
                        help="Glob pattern relative to wds-dir (default: *.tar)")
    parser.add_argument("--bz", type=int, required=True,
                        help="Simulated batch size to average gradients over")
    parser.add_argument("--mini-bz", type=int, default=16,
                        help="Microbatch size for gradient accumulation")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of DataLoader workers (default: 8)")
    parser.add_argument("--n-batches", type=int, default=100,
                        help="Number of simulated batches to process")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Optional cap on number of layers to log (default: all)")
    parser.add_argument("--log-json", type=str, required=True,
                        help="Path to a JSONL file to append results to (will create if missing)")
    parser.add_argument("--every-k", dest="every_k", type=int, default=1,
                        help="Process only every k-th simulated batch; k=1 processes all (default: 1)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Enable sample-level dataset shuffling with a bounded buffer")
    parser.add_argument("--shuffle-buffer", type=int, default=10000,
                        help="Shuffle buffer size (larger approximates fuller shuffle; default: 10000)")
    parser.add_argument("--shuffle-seed", type=int, default=1337,
                        help="Random seed for shuffling (default: 1337)")
    main(parser.parse_args()) 