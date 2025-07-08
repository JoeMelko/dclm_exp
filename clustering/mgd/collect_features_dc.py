#!/usr/bin/env python
"""
collect_features_dc.py
----------------------
Compute *per-example* **raw** LoRA gradient features (no whitening) for a tokenised WebDataset and store them
in an *existing* NumPy memmap.

Each per-sample feature tensor has shape ``(2 * num_blocks, lora_rank * lora_rank)`` and is stored
verbatim (no flattening / whitening). The memmap therefore has shape::

    (num_shards * shard_size, 2 * num_blocks, lora_rank * lora_rank)

This matches what `create_mmap_features.py` allocates.

Launch pattern remains identical – one GPU process writes a contiguous slice
defined via ``--start-offset``.

The workflow mimics `collect_grads.py` / `launch_collect_grads.sh` so that you
can spawn 8 independent GPU workers that each append their slice of the dataset
into the shared mmap without race conditions:

1. Pre-allocate the mmap once (see `create_mmap_features.py`).
2. Launch N parallel instances of this script (one per GPU) with unique
   ``--gpu-id`` / ``--start-offset`` arguments so every worker writes to its own
   segment.

Compared to `collect_cosine_sim_dc.py` this version
* **does not** compute cosine similarities – only the feature vectors,
* writes directly to a memmap instead of an ``.npz`` file,
* follows the sharding / worker logic of `collect_grads.py` so it can be used
  from `launch_collect_grads.sh`-style launchers.
"""
import argparse, json, gzip, io, os, sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure parent directory (clustering/) is on PYTHONPATH **before** we attempt
# to import `lora.*` so the modules are resolvable regardless of the cwd.
# ---------------------------------------------------------------------------
PARENT_DIR = Path(__file__).resolve().parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import numpy as np
import torch, tqdm, webdataset as wds
from transformers import AutoTokenizer

from lora.lora import LoRAHandler
from lora.logger import Logger

from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from open_lm.main import load_model

# allow TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------- configuration -------------------- #
DTYPE_OUT  = np.float16   # on-disk dtype for the feature vectors
BATCH_SIZE = 16            # must match Logger initialisation
# ------------------------------------------------------- #

# --- optional: only needed when you use --uuid -----------------------------
try:
    from open_lm.utils import download_cached          # S3/HTTP cache helper
    from open_lm.factory import build_open_lm          # model builder
except ImportError:                                    # fallback for --ckpt
    download_cached = build_open_lm = None

# ──────────────────────────── Data helpers ────────────────────────────────
# We only need the tokenizer for padding-id lookup.
_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

def _load_tokens_from_sample(sample: dict) -> np.ndarray:
    """Extract the token list from a WebDataset sample and return it as a
    1-D ``np.ndarray`` of dtype ``int32``.
    Accepts either legacy ``tokens.npy`` or generic ``*.json[.gz]`` payloads
    with fields ``tokens`` or ``input_ids``.
    """
    if "tokens.npy" in sample:  # legacy path
        return np.load(io.BytesIO(sample["tokens.npy"]))

    # otherwise assume JSON (optionally gzipped)
    key = next(k for k in sample if k.endswith("json") or k.endswith("json.gz"))
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
        raise TypeError(f"Unsupported JSON payload type: {type(obj)}")

    return np.asarray(tokens_list, dtype=np.int32)

# ──────────────────────────── Model loading ───────────────────────────────

def _load_openlm_model_from_uuid(run_uuid: str):
    """Resolve an Open-LM run UUID to a checkpoint and build the model."""
    from types import SimpleNamespace
    from open_lm.model import create_params
    from open_lm.utils.transformers.hf_config import OpenLMConfig

    # Resolve project root dynamically: ascend the directory tree until we find
    # an `exp_data` directory.
    _here = Path(__file__).resolve()
    project_root = next((p for p in _here.parents if (p / "exp_data").exists()), _here.parent.parent.parent)
    exp_root = project_root / "exp_data" / "models"
    meta_path = next(exp_root.rglob(f"*{run_uuid}*.json"), None)
    if meta_path is None:
        raise FileNotFoundError(
            f"could not find metadata matching '*{run_uuid}*.json' in {exp_root}"
        )

    meta = json.loads(meta_path.read_text())
    ckpt_path = Path(meta["checkpoint_url"]).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint file not found: {ckpt_path}")

    cfg_rel = meta["hyperparameters"].get("model")
    if cfg_rel is None:
        raise KeyError("Missing 'hyperparameters.model' entry in metadata JSON")
    cfg_path = (project_root / cfg_rel).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"model config file not found: {cfg_path}")

    dummy_args = SimpleNamespace(
        model=str(cfg_path),
        model_norm=meta["hyperparameters"].get("norm", "gain_only_lp_layer_norm"),
        attn_name="torch_attn",
        attn_activation=None,
        attn_seq_scalar=None,
        attn_seq_scalar_alpha=None,
        qk_norm=meta["hyperparameters"].get("qk_norm", False),
        positional_embedding_type="rotary",
        ffn_type="swiglu_torch",
        moe_num_experts=meta["hyperparameters"].get("moe_num_experts", 8),
        moe_loss_weight=0.1,
        moe_expert_model_parallelism=False,
        moe_weight_parallelism=False,
        moe_capacity_factor=1.25,
        moe_freq=0,
        moe_top_k=2,
    )

    params = create_params(dummy_args)
    wrapper = OpenLMforCausalLM(OpenLMConfig(params))

    load_args = SimpleNamespace(resume=str(ckpt_path), fsdp=False, distributed=False, seed=0)
    load_model(load_args, wrapper.model, different_seed=True)
    return wrapper


def _get_model(args):
    if args.uuid is None:
        raise ValueError("--uuid is required (HuggingFace checkpoints no longer supported)")
    return _load_openlm_model_from_uuid(args.uuid)


# ──────────────────────────── LoRA helpers ────────────────────────────────

def _prepare_model(model, args):
    model_parameters = sum(p.numel() for p in model.parameters())
    handler = LoRAHandler(rank=args.lora_rank)
    logger = Logger(model_parameters, args.num_blocks)
    # LoRAHandler expects the bare Transformer (with attribute n_layers).
    core_model = model.model if hasattr(model, "model") else model
    handler.add_lora(core_model, logger, BATCH_SIZE)
    logger.init_grads(BATCH_SIZE, args.lora_rank)

    # freeze all except LoRA params
    for name, param in model.named_parameters():
        if "logix_lora_B" not in name:
            param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()
    return model, handler, logger

# ───────────────────────────── Data collation ─────────────────────────────

def _collate(batch):
    toks = [_load_tokens_from_sample(b) for b in batch]
    L   = max(len(t) for t in toks)

    pad_id  = _tokenizer.eos_token_id
    pad_fn  = lambda arr: np.pad(arr, (0, L - len(arr)), constant_values=pad_id)
    toks_padded = torch.tensor([pad_fn(t) for t in toks], dtype=torch.long)

    labels = toks_padded.clone()
    labels[labels == pad_id] = -100  # ignore pad

    return toks_padded, labels

# ─────────────────────────────── Main ─────────────────────────────────────

def main(args):
    # ---------- model ----------
    model, handler, logger = _prepare_model(_get_model(args), args)

    device = next(model.parameters()).device

    # ---------- feature dimensions ----------
    blocks = args.num_blocks * 2
    dim2   = args.lora_rank * args.lora_rank

    # ---------- dataset ----------
    wds_dir_path = Path(args.wds_dir)
    if not wds_dir_path.is_absolute() and not wds_dir_path.exists():
        alt_path = Path(__file__).resolve().parent / wds_dir_path
        if alt_path.exists():
            wds_dir_path = alt_path
        else:
            raise FileNotFoundError(f"dataset directory not found: {args.wds_dir}")

    tar_files = sorted(wds_dir_path.glob("*.tar"))
    if len(tar_files) == 0:
        raise FileNotFoundError(f"No .tar shards found in {wds_dir_path}")

    # ---------------- shard selection: multi-GPU ----------------
    # Each GPU processes a contiguous slice of `tar_files` determined by its id.
    start_idx = args.chunk_size * args.gpu_id
    end_idx   = args.chunk_size * (args.gpu_id + 1)

    # Guard against overshooting the number of available shards.
    if start_idx >= len(tar_files):
        raise RuntimeError(
            f"GPU {args.gpu_id} received no shards – check --chunk-size and --total-gpus"
        )

    selected_indices = range(start_idx, min(end_idx, len(tar_files)))
    selected_files   = [tar_files[i] for i in selected_indices]

    ds = wds.WebDataset([str(p) for p in selected_files])
    # Use a single worker to preserve exact sample order → deterministic index
    loader = wds.WebLoader(
        ds,
        batch_size=BATCH_SIZE,
        num_workers=args.workers,
        collate_fn=_collate,
        persistent_workers=args.workers > 0,
    )

    # ---------- mmap ----------
    N_est = args.num_shards * args.shard_size  # must match pre-allocation
    feats = np.memmap(args.out, mode="r+", dtype=DTYPE_OUT, shape=(N_est, blocks, dim2))

    row_offset = args.start_offset

    # ---------- processing loop ----------
    for toks, labels in tqdm.tqdm(loader):
        # Stop at incomplete batch because Logger assumes fixed batch size
        if toks.shape[0] < BATCH_SIZE:
            print(f"Encountered incomplete batch of size {toks.shape[0]} < {BATCH_SIZE}; stopping.")
            break

        toks = toks.to(device)
        labels = labels.to(device)

        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=toks, labels=labels)
                out.loss.mul_(1024)
                out.loss.backward()
            # collect grads → (num_blocks, batch, rank)
            features = (
                logger.grads.detach().cpu().to(torch.float32).numpy().transpose(1, 0, 2)
            )

        # Reset gradient buffer before further ops
        logger.grads.zero_()

        # ------- write to mmap -------
        feats[row_offset:row_offset + features.shape[0]] = features
        row_offset += features.shape[0]

        # Flush periodically to avoid data loss
        if (row_offset - args.start_offset) % (BATCH_SIZE * 100) == 0:
            feats.flush()

    feats.flush()

# ------------------------------ CLI ---------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wds-dir", required=True,
                    help="directory containing tokenised WebDataset shards")
    ap.add_argument("--uuid", required=True, help="Datacomp-LM run UUID")

    # mmap / sharding related flags (identical semantics to collect_grads.py)
    ap.add_argument("--out",   default="clustering/mgd/features.fp16",
                    help="output memmap filename")
    ap.add_argument("--lora-rank",  type=int, default=128)
    ap.add_argument("--num-blocks", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=15,
                    help="contiguous shard block size assigned in one go (default 15)")
    ap.add_argument("--gpu-id", type=int, default=int(os.environ.get("GPU_ID", 0)),
                    help="numeric id of the current GPU / process [0-7]")
    ap.add_argument("--total-gpus", type=int, default=8,
                    help="how many independent GPU workers will run in parallel")
    ap.add_argument("--num-shards", type=int, default=120,
                    help="total number of dataset shards (default 120)")
    ap.add_argument("--shard-size", type=int, required=True,
                    help="number of samples per shard")
    ap.add_argument("--start-offset", type=int, required=True,
                    help="row offset in the mmap where this worker starts writing")

    # Number of background WebDataset loader workers (0 = main process only).
    ap.add_argument("--workers", type=int, default=0,
                    help="Number of subprocess workers for WebDataset loader (default: 0).")

    args = ap.parse_args()

    main(args) 