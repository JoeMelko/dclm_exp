#!/usr/bin/env python
"""
eval_grad_cosine.py
-------------------
Two-pass gradient analysis to measure drift/bias:

Pass 1:
- Stream the dataset and simulate batches of size --bz via microbatch accumulation.
- Compute the example-weighted average gradient across all processed examples.

Pass 2:
- Stream the dataset again with the same batching.
- For each simulated batch, compute cosine similarity between that batch's
  gradients and the average gradient from Pass 1:
    - Total cosine across all selected parameters (no per-layer logging)

Notes:
- Targets the same parameter groups as eval_grad_svd:
    layers[i].attention.in_proj.weight
    layers[i].attention.out_proj.weight
    layers[i].feed_forward.w12.weight
    layers[i].feed_forward.w3.weight
- Streaming shuffling (bounded buffer) is available via --shuffle flags.

Outputs:
- Appends one JSON line per simulated batch in Pass 2 to --log-json with:
    {"batch_index": int, "simulated_bz": int, "num_micro_steps": int, "cosine_total": float}
"""
# --------------------------------------------------------------------------- #
import argparse, io, json, gzip, math, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
#                         Collate and utilities                               #
# --------------------------------------------------------------------------- #
def collate(batch: List[dict]):
    toks: List[np.ndarray] = []
    for b in batch:
        arr = _load_tokens_from_sample(b)
        if arr is not None and len(arr) > 0:
            toks.append(arr)
    if not toks:
        raise ValueError("Empty batch after filtering invalid samples")

    L = max(len(t) for t in toks)
    pad_id_inputs = SEP_TOKEN_ID

    def pad_toks(x: np.ndarray) -> np.ndarray:
        return np.pad(x, (0, L - len(x)), constant_values=pad_id_inputs)

    def pad_labels(x: np.ndarray) -> np.ndarray:
        return np.pad(x, (0, L - len(x)), constant_values=-100)

    toks_padded = torch.tensor([pad_toks(t) for t in toks], dtype=torch.long)
    labels = torch.tensor([pad_labels(t) for t in toks], dtype=torch.long)
    labels[toks_padded == SEP_TOKEN_ID] = -100
    return toks_padded, labels


def _append_jsonl(log_path: Path, record: dict):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _build_loader(args) -> wds.WebLoader:
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
    if getattr(args, "shuffle", False):
        buffer_size = int(getattr(args, "shuffle_buffer", 10000) or 10000)
        rng = random.Random(int(getattr(args, "shuffle_seed", 1337) or 1337))
        try:
            ds = ds.shuffle(buffer_size=buffer_size, initial=buffer_size, rng=rng)
        except TypeError:
            try:
                ds = ds.shuffle(bufsize=buffer_size, initial=buffer_size, rng=rng)
            except TypeError:
                try:
                    ds = ds.shuffle(buffer_size, buffer_size)
                except TypeError:
                    ds = ds.shuffle(buffer_size, initial=buffer_size)

    loader = wds.WebLoader(ds,
                           batch_size=args.mini_bz,
                           num_workers=args.num_workers,
                           collate_fn=collate)
    return loader


def _iter_simulated_batches(model: torch.nn.Module,
                            data_iter,
                            device: torch.device,
                            target_bz: int):
    """Yield (num_micro_steps, num_examples_in_sim_batch) after setting grads on model."""
    accumulated_examples = 0
    steps = 0
    model.zero_grad(set_to_none=True)
    model.eval()
    while accumulated_examples < target_bz:
        toks, labels = next(data_iter)  # may raise StopIteration
        toks, labels = toks.to(device), labels.to(device)
        use_doc_mask = getattr(getattr(model, "model", None), "params", None) is not None and getattr(model.model.params, "doc_causal_mask", False)
        attn_mask = None if use_doc_mask else (labels != -100)
        out = model(input_ids=toks, labels=labels, attention_mask=attn_mask)
        micro_examples = toks.size(0)
        scale = float(micro_examples) / float(target_bz)
        (out.loss * scale).backward()
        accumulated_examples += micro_examples
        steps += 1
    return steps, accumulated_examples


def _collect_layer_param_grads(model: torch.nn.Module) -> List[Dict[str, Optional[torch.Tensor]]]:
    """Return list indexed by layer, with selected param grads as tensors (can be None)."""
    try:
        layers = model.model.layers  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError("Model structure does not have model.layers as expected") from e
    out: List[Dict[str, Optional[torch.Tensor]]] = []
    for layer in layers:
        entry: Dict[str, Optional[torch.Tensor]] = {
            "attn_in": getattr(getattr(layer, "attention"), "in_proj").weight.grad if getattr(getattr(layer, "attention"), "in_proj", None) is not None else None,
            "attn_out": getattr(getattr(layer, "attention"), "out_proj").weight.grad if getattr(getattr(layer, "attention"), "out_proj", None) is not None else None,
            "ffn_in": getattr(getattr(layer, "feed_forward"), "w12").weight.grad if getattr(getattr(layer, "feed_forward"), "w12", None) is not None else None,
            "ffn_out": getattr(getattr(layer, "feed_forward"), "w3").weight.grad if getattr(getattr(layer, "feed_forward"), "w3", None) is not None else None,
        }
        out.append(entry)
    return out


def _ensure_like_zeros(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return torch.zeros_like(t, dtype=torch.float32) if t is not None else None


def _accumulate_weighted_sum(dest: Optional[torch.Tensor],
                             src: Optional[torch.Tensor],
                             weight: float) -> Optional[torch.Tensor]:
    if src is None:
        return dest
    src32 = src.detach().to(dtype=torch.float32)
    if dest is None:
        dest = torch.zeros_like(src32)
    dest += src32 * weight
    return dest


def _concat_nonnull_tensors(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for t in tensors:
        if t is None:
            continue
        f = t.detach().to(dtype=torch.float32).view(-1)
        if f.numel() > 0:
            flat.append(f)
    if not flat:
        return None
    return torch.cat(flat, dim=0)


def _cosine_sim(x: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> float:
    if x is None or y is None:
        return float("nan")
    x = x.view(-1)
    y = y.view(-1)
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    dot = torch.dot(x, y)
    nx = torch.norm(x)
    ny = torch.norm(y)
    eps = torch.tensor(1e-12, dtype=dot.dtype, device=dot.device)
    return float((dot / (nx * ny + eps)).item())


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #
def main(args):
    # ----- model ------------------------------------------------------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_openlm_model_from_uuid(args.uuid)
    try:
        model.model.params.doc_causal_mask = True
    except Exception:
        pass
    model.to(device, dtype=torch.bfloat16).eval()

    # ----- pass 1: compute example-weighted average gradient ---------------- #
    loader1 = _build_loader(args)
    data_iter1 = iter(loader1)

    # We'll accumulate: sum_grad_per_layer[param_name] weighted by examples, then divide by total_examples
    avg_grads_per_layer: List[Dict[str, Optional[torch.Tensor]]] = []
    total_examples = 0
    num_layers_capped: Optional[int] = None

    for _ in tqdm.tqdm(range(args.n_batches), desc="Pass 1: averaging gradients"):
        try:
            steps, sim_bz = _iter_simulated_batches(model, data_iter1, device, args.bz)
        except StopIteration:
            break
        # Collect current grads
        layer_grads = _collect_layer_param_grads(model)
        if num_layers_capped is None:
            num_layers_capped = len(layer_grads) if args.num_layers is None else min(len(layer_grads), int(args.num_layers))
            # Initialize accumulators
            avg_grads_per_layer = [{"attn_in": None, "attn_out": None, "ffn_in": None, "ffn_out": None} for _ in range(num_layers_capped)]
        # Accumulate weighted by target_bz to convert the batch-average grad
        # (scaled by 1/target_bz during accumulation) back to the sum over examples.
        for i in range(min(num_layers_capped, len(layer_grads))):
            for k in ("attn_in", "attn_out", "ffn_in", "ffn_out"):
                avg_grads_per_layer[i][k] = _accumulate_weighted_sum(avg_grads_per_layer[i][k], layer_grads[i][k], float(args.bz))
        total_examples += int(sim_bz)
        model.zero_grad(set_to_none=True)

        # Optional skipping between simulated batches
        k = max(1, int(getattr(args, "every_k", 1) or 1))
        if k > 1:
            to_skip_examples = (k - 1) * args.bz
            skipped = 0
            try:
                while skipped < to_skip_examples:
                    toks, _ = next(data_iter1)
                    skipped += int(getattr(toks, "size")(0))
            except StopIteration:
                break

    if total_examples == 0:
        raise RuntimeError("No examples processed in Pass 1; cannot compute average gradient.")
    # Convert sums to averages
    for layer_dict in avg_grads_per_layer:
        for k in ("attn_in", "attn_out", "ffn_in", "ffn_out"):
            if layer_dict[k] is not None:
                layer_dict[k] = layer_dict[k] / float(total_examples)

    # ----- pass 2: batch-wise cosines vs average gradient ------------------- #
    loader2 = _build_loader(args)
    data_iter2 = iter(loader2)

    batch_index = 0
    for _ in tqdm.tqdm(range(args.n_batches), desc="Pass 2: cosine similarities"):
        try:
            steps, sim_bz = _iter_simulated_batches(model, data_iter2, device, args.bz)
        except StopIteration:
            break
        layer_grads = _collect_layer_param_grads(model)

        # Compute total cosine by aggregating dot products and norms across layers
        total_dot = 0.0
        total_norm_sq_batch = 0.0
        total_norm_sq_avg = 0.0
        Lcap = len(avg_grads_per_layer)
        for i in range(min(Lcap, len(layer_grads))):
            g_batch_flat = _concat_nonnull_tensors([
                layer_grads[i]["attn_in"],
                layer_grads[i]["attn_out"],
                layer_grads[i]["ffn_in"],
                layer_grads[i]["ffn_out"],
            ])
            g_avg_flat = _concat_nonnull_tensors([
                avg_grads_per_layer[i]["attn_in"],
                avg_grads_per_layer[i]["attn_out"],
                avg_grads_per_layer[i]["ffn_in"],
                avg_grads_per_layer[i]["ffn_out"],
            ])
            if g_batch_flat is not None and g_avg_flat is not None:
                # Accumulate for total cosine across layers without full concat
                # Move to same device if needed
                x = g_batch_flat
                y = g_avg_flat
                total_dot += float(torch.dot(x, y).item())
                total_norm_sq_batch += float(torch.dot(x, x).item())
                total_norm_sq_avg += float(torch.dot(y, y).item())

        eps = 1e-12
        cosine_total = float(total_dot / (math.sqrt(total_norm_sq_batch) * math.sqrt(total_norm_sq_avg) + eps)) if total_norm_sq_batch > 0.0 and total_norm_sq_avg > 0.0 else float("nan")

        record = {
            "batch_index": int(batch_index),
            "simulated_bz": int(sim_bz),
            "num_micro_steps": int(steps),
            "cosine_total": cosine_total,
        }
        _append_jsonl(Path(args.log_json), record)
        batch_index += 1
        model.zero_grad(set_to_none=True)

        # Optional skipping between simulated batches
        k = max(1, int(getattr(args, "every_k", 1) or 1))
        if k > 1:
            to_skip_examples = (k - 1) * args.bz
            skipped = 0
            try:
                while skipped < to_skip_examples:
                    toks, _ = next(data_iter2)
                    skipped += int(getattr(toks, "size")(0))
            except StopIteration:
                break

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
                        help="Number of simulated batches to process in each pass")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Optional cap on number of layers to consider (default: all)")
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


